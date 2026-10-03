#!/usr/bin/env python3
"""
compile_migraphx.py — Compile ONNX models to MIGraphX GPU binaries.

Why MIGraphX instead of PyTorch directly on NucBox:
  PyTorch ROCm wheels don't yet support gfx1151 (Strix Halo / AMD Radeon 890M).
  MIGraphX is AMD's native inference compiler: it ingests an ONNX graph and
  emits optimised GCN/RDNA kernels targeting the exact installed GPU.

  PyTorch DINOv2: ~350 ms/frame (CPU)
  MIGraphX DINOv2 target: ~20-30 ms/frame (GPU gfx1151)
  Full pipeline reduction: 9.5s → <300 ms/frame

What this script does:
  1. Loads an ONNX model.
  2. Compiles it with MIGraphX to a binary (.mxr file).
  3. Caches the compiled binary alongside the ONNX file.
  4. Optionally runs a benchmark (migraphx-driver perf equivalent).

Blocked on: ROCm gfx1151 kernel conflict (see docs/SPRINT3_ROCM_ISSUE.md).
When unblocked, run:
  python3 perception/tools/compile_migraphx.py \
    --onnx models/dino_vitb14_patches.onnx \
    --output models/dino_vitb14_patches.mxr \
    --benchmark

Then update run_eval.py to use the MIGraphX runtime path:
  PYTHONPATH=perception python3 perception/eval/run_eval.py \
    --detector sam2_amg --amg-points 20 --confidence 0.2 \
    --negative-weight 1.0 --nms-iou 0.5 \
    --migraphx-dino models/dino_vitb14_patches.mxr

[NUCBOX] only — MIGraphX is not available on Mac.
Sprint 4: E8 MIGraphX deployment path.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_INPUT_SIZE = 518
_N_PATCHES  = 37 * 37   # 1369
_EMBED_DIM  = 768


def _require_migraphx():
    """Import migraphx or exit with a clear message."""
    try:
        import migraphx  # noqa: F401
        return migraphx
    except ImportError:
        logger.error(
            "migraphx Python bindings not found. "
            "This script requires ROCm + MIGraphX (AMD NucBox / NUCBOX only). "
            "See docs/SPRINT3_ROCM_ISSUE.md for the current blocking issue."
        )
        sys.exit(1)


def compile_onnx(
    onnx_path: Path,
    output_path: Path,
    gpu: bool = True,
    fp16: bool = False,
) -> None:
    """Compile an ONNX model to a MIGraphX binary (.mxr).

    The .mxr file contains GPU-native code for the target device. Compilation
    takes 30–120 seconds (first time); subsequent loads take <1 second via
    migraphx.load().

    fp16: Enables FP16 quantisation. Halves memory bandwidth and often doubles
          throughput on RDNA3. Reduces numerical precision slightly — acceptable
          for cosine similarity scoring.
    """
    migraphx = _require_migraphx()

    logger.info("Loading ONNX from %s...", onnx_path)
    prog = migraphx.parse_onnx(str(onnx_path))

    if fp16:
        logger.info("Applying FP16 quantisation...")
        migraphx.quantize_fp16(prog)

    target = "gpu" if gpu else "cpu"
    logger.info("Compiling for target=%s (this may take 60–120s)...", target)
    t0 = time.perf_counter()
    prog.compile(migraphx.get_target(target))
    elapsed = time.perf_counter() - t0
    logger.info("Compilation complete in %.1fs.", elapsed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prog.save(str(output_path))
    size_mb = output_path.stat().st_size / 1e6
    logger.info("Saved compiled binary to %s (%.1f MB).", output_path, size_mb)


def benchmark(mxr_path: Path, n_warmup: int = 5, n_runs: int = 50) -> None:
    """Run latency benchmark on a compiled .mxr model."""
    migraphx = _require_migraphx()

    logger.info("Loading compiled binary from %s...", mxr_path)
    prog = migraphx.load(str(mxr_path))

    dummy = np.random.randn(1, 3, _INPUT_SIZE, _INPUT_SIZE).astype(np.float32)

    logger.info("Warming up (%d runs)...", n_warmup)
    for _ in range(n_warmup):
        result = prog.run({"image": migraphx.argument(dummy)})

    logger.info("Benchmarking (%d runs)...", n_runs)
    latencies_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        prog.run({"image": migraphx.argument(dummy)})
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    latencies_ms.sort()
    mean_ms = sum(latencies_ms) / len(latencies_ms)
    p50_ms  = latencies_ms[len(latencies_ms) // 2]
    p99_ms  = latencies_ms[int(0.99 * len(latencies_ms))]

    logger.info("── MIGraphX Latency ──")
    logger.info("  Model:  %s", mxr_path.name)
    logger.info("  Mean:   %.2f ms", mean_ms)
    logger.info("  p50:    %.2f ms", p50_ms)
    logger.info("  p99:    %.2f ms", p99_ms)
    logger.info("  Speedup vs CPU (~350ms): ~%.1fx", 350.0 / mean_ms)

    # Validate output shape
    out_np = np.array(result[0])
    if out_np.shape == (1, _N_PATCHES, _EMBED_DIM):
        logger.info("  Output shape: %s ✓", out_np.shape)
    else:
        logger.warning("  Unexpected output shape: %s", out_np.shape)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile ONNX models to MIGraphX GPU binaries for NucBox inference."
    )
    parser.add_argument(
        "--onnx", type=Path, required=True,
        help="Path to the ONNX model to compile (e.g. models/dino_vitb14_patches.onnx).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help=(
            "Output path for the compiled .mxr binary. "
            "Defaults to the same directory as --onnx with .mxr extension."
        ),
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Compile for CPU target instead of GPU (debugging only).",
    )
    parser.add_argument(
        "--fp16", action="store_true",
        help=(
            "Apply FP16 quantisation before compilation. "
            "~2x throughput on RDNA3 with minor numerical precision trade-off."
        ),
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="After compilation, run latency benchmark.",
    )
    parser.add_argument(
        "--benchmark-only", action="store_true",
        help="Skip compilation; load existing .mxr and benchmark only.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(repo_root)

    onnx_path = args.onnx if args.onnx.is_absolute() else repo_root / args.onnx

    if args.output:
        output_path = args.output if args.output.is_absolute() else repo_root / args.output
    else:
        output_path = onnx_path.with_suffix(".mxr")

    if not args.benchmark_only:
        if not onnx_path.exists():
            logger.error("ONNX file not found: %s", onnx_path)
            logger.error(
                "Run first: PYTHONPATH=perception python3 perception/tools/export_dino_onnx.py"
            )
            sys.exit(1)
        compile_onnx(onnx_path, output_path, gpu=not args.cpu, fp16=args.fp16)

    if args.benchmark or args.benchmark_only:
        if not output_path.exists():
            logger.error("Compiled binary not found: %s. Run without --benchmark-only first.", output_path)
            sys.exit(1)
        benchmark(output_path)

    logger.info("")
    logger.info("To use in eval, add --migraphx-dino %s to run_eval.py.", output_path)
    logger.info("(--migraphx-dino support: see SAM2AMGDetector._load_models — Sprint 4 E8)")


if __name__ == "__main__":
    main()
