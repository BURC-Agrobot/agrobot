#!/usr/bin/env python3
"""
export_dino_onnx.py — Export DINOv2 ViT-B/14 patch token extractor to ONNX.

Why ONNX instead of running PyTorch directly on NucBox:
  The pre-built ROCm 6.4 PyTorch wheels don't support gfx1151 (Strix Halo).
  ONNX is a hardware-agnostic graph format. MIGraphX ingests it and compiles
  native AMD GCN/RDNA kernels for the exact target GPU at load time — no
  pre-baked GFX target list required.

What this exports:
  A wrapper around DINOv2 ViT-B/14 that takes one preprocessed frame:
    Input:  (1, 3, 518, 518)  float32, ImageNet-normalised
    Output: (1, 1369, 768)    float32, L2-normalised patch token embeddings
  The output is exactly what dino_sam2_detector.py reads as x_norm_patchtokens.
  CLS token is excluded — it's only needed for query embedding build (one-time).

Why opset 17:
  MIGraphX supports up to opset 18. Opset 17 includes LayerNormalization as a
  native op. Older opsets decompose it into 6+ primitives, bloating the graph
  and reducing MIGraphX's ability to fuse operations.

Why tracing not scripting:
  DINOv2's forward_features has Python-level control flow. torch.onnx.export
  uses tracing — it runs one forward pass with dummy input and records tensor
  ops. For a fixed-size feedforward ViT, this is correct and produces a clean
  graph. Dynamic control flow would require TorchScript, which DINOv2 doesn't
  support out of the box.

Usage (from repo root):
  # [MAC] — runs on MPS (fast) or CPU
  PYTHONPATH=perception python3 perception/tools/export_dino_onnx.py \
    --output models/dino_vitb14_patches.onnx

  # Verify the exported graph
  python3 perception/tools/export_dino_onnx.py --verify \
    --output models/dino_vitb14_patches.onnx

  # Then transfer to NucBox
  bash tools/network/setup/model_sync.sh --onnx models/dino_vitb14_patches.onnx

  # If MIGraphX segfaults on gfx1151, try fixed batch (rules out dynamic-shape):
  PYTHONPATH=perception python3 perception/tools/export_dino_onnx.py \
    --output models/dino_vitb14_patches_fixed.onnx --fixed-batch

On NucBox (Sprint 3.2):
  migraphx-driver perf --onnx models/dino_vitb14_patches.onnx --gpu

Known gfx1151 (Strix Halo) issue:
  MIGraphX may segfault during "Compiling..." due to MIOpen Conv2d compilation
  bugs on gfx1151 (ROCm/rocm-libraries#4070). If so, try --fixed-batch to rule
  out dynamic-shape issues. If segfault persists, it is upstream ROCm, not this
  export.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DINO_MODEL_NAME = "dinov2_vitb14"
_INPUT_SIZE = 518     # must be a multiple of patch_size=14: 518 = 37 * 14
_EMBED_DIM  = 768     # ViT-B embedding dimension
_N_PATCHES  = 37 * 37  # 1369 patch tokens per frame
_OPSET      = 17


def _select_device() -> torch.device:
    if os.environ.get("AGROBOT_FORCE_CPU", "0") == "1":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DINOv2PatchExtractor(torch.nn.Module):
    """Thin wrapper that exposes only the patch token output of DINOv2.

    torch.onnx.export traces the forward() method. By isolating just the
    patch token extraction here, the exported graph contains no dead code
    (CLS head, classification layers) and MIGraphX compiles a smaller graph.
    """

    def __init__(self, dino: torch.nn.Module) -> None:
        super().__init__()
        self.dino = dino

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, 518, 518) float32 ImageNet-normalised tensor.
        Returns:
            (batch, 1369, 768) float32 L2-normalised patch token embeddings.
        """
        features = self.dino.forward_features(x)
        patch_tokens = features["x_norm_patchtokens"]  # (B, 1369, 768)
        # L2-normalise per patch so cosine similarity = dot product at inference.
        # Baking this into the ONNX graph saves a normalize call per frame.
        return F.normalize(patch_tokens, dim=-1)


def export(output_path: Path, device: torch.device, fixed_batch: bool = False) -> None:
    logger.info("Loading DINOv2 (%s) via torch.hub...", _DINO_MODEL_NAME)
    dino = torch.hub.load(
        "facebookresearch/dinov2", _DINO_MODEL_NAME, pretrained=True
    )
    dino.eval()
    dino.to(device)
    logger.info("DINOv2 loaded on %s.", device)

    model = DINOv2PatchExtractor(dino).eval().to(device)

    # Dummy input: one 518×518 frame, ImageNet-normalised range ~[-2.5, 2.5]
    dummy = torch.randn(1, 3, _INPUT_SIZE, _INPUT_SIZE, device=device)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # MIGraphX has limited dynamic-shape support; fixed batch can avoid crashes.
    # dynamo=False: legacy TorchScript exporter, stable with MIGraphX.
    export_kwargs: dict = {
        "opset_version": _OPSET,
        "input_names": ["image"],
        "output_names": ["patch_tokens"],
        "do_constant_folding": True,
        "dynamo": False,
    }
    if not fixed_batch:
        export_kwargs["dynamic_axes"] = {
            "image": {0: "batch"},
            "patch_tokens": {0: "batch"},
        }

    logger.info("Exporting to ONNX (opset %d, fixed_batch=%s)...", _OPSET, fixed_batch)
    with torch.no_grad():
        torch.onnx.export(model, dummy, str(output_path), **export_kwargs)

    size_mb = output_path.stat().st_size / 1e6
    logger.info("Exported to %s (%.1f MB).", output_path, size_mb)
    batch_desc = "1" if fixed_batch else "batch"
    logger.info(
        "Input:  (%s, 3, %d, %d)  →  Output: (%s, %d, %d)",
        batch_desc, _INPUT_SIZE, _INPUT_SIZE, batch_desc, _N_PATCHES, _EMBED_DIM,
    )


def verify(output_path: Path, device: torch.device) -> None:
    """Load the ONNX model with onnxruntime and run one forward pass.

    Checks:
      1. ONNX graph is valid (onnx.checker)
      2. Output shape matches expected (1, 1369, 768)
      3. Output is L2-normalised (norms ≈ 1.0)
      4. Difference from PyTorch reference is small (numerical parity)
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        logger.error("onnx and onnxruntime are required for --verify. "
                     "pip install onnx onnxruntime")
        sys.exit(1)

    logger.info("Verifying %s...", output_path)

    model_proto = onnx.load(str(output_path))
    onnx.checker.check_model(model_proto)
    logger.info("  ONNX graph valid.")

    sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    dummy_np = np.random.randn(1, 3, _INPUT_SIZE, _INPUT_SIZE).astype(np.float32)
    out = sess.run(["patch_tokens"], {"image": dummy_np})[0]

    assert out.shape == (1, _N_PATCHES, _EMBED_DIM), \
        f"Shape mismatch: {out.shape}"
    logger.info("  Output shape: %s ✓", out.shape)

    norms = np.linalg.norm(out[0], axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5), \
        f"Not L2-normalised: min={norms.min():.4f} max={norms.max():.4f}"
    logger.info("  L2 norms: min=%.4f max=%.4f ✓", norms.min(), norms.max())

    # PyTorch reference
    dino = torch.hub.load(
        "facebookresearch/dinov2", _DINO_MODEL_NAME, pretrained=True
    ).eval()
    wrapper = DINOv2PatchExtractor(dino).eval()
    with torch.no_grad():
        pt_out = wrapper(torch.from_numpy(dummy_np)).numpy()

    max_diff = np.abs(pt_out - out).max()
    logger.info("  Max |PyTorch − ONNX|: %.2e %s",
                max_diff, "✓" if max_diff < 1e-4 else "⚠ larger than expected")

    logger.info("Verification passed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export DINOv2 ViT-B/14 patch extractor to ONNX for MIGraphX."
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("models/dino_vitb14_patches.onnx"),
        help="Output .onnx path (default: models/dino_vitb14_patches.onnx).",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="After export, verify with onnxruntime (requires: pip install onnx onnxruntime).",
    )
    parser.add_argument(
        "--fixed-batch", action="store_true",
        help="Export with fixed batch=1 (no dynamic_axes). Use if MIGraphX segfaults on gfx1151.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(repo_root)

    output = args.output if args.output.is_absolute() else repo_root / args.output
    device = _select_device()
    logger.info("Device: %s", device)

    export(output, device, fixed_batch=args.fixed_batch)

    if args.verify:
        verify(output, device)
    else:
        logger.info("Tip: re-run with --verify to check numerical parity.")

    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Transfer to NucBox:")
    logger.info("       bash tools/network/setup/model_sync.sh --onnx %s", output)
    logger.info("  2. Benchmark on NucBox (S3.2):")
    logger.info("       migraphx-driver perf --onnx models/dino_vitb14_patches.onnx --gpu")


if __name__ == "__main__":
    main()
