"""Locate the repository so tools and nodes can find models and data."""

import os
from pathlib import Path


# Read the repository root from the environment set by setup_env.sh
def find_repo_root() -> Path:
    # Read the root once because installed code cannot infer the source path
    root_text = os.environ.get("AGROBOT_ROOT")
    # Refuse to guess a location when the root is unknown
    if not root_text:
        raise RuntimeError("AGROBOT_ROOT is not set. Run: source setup_env.sh")
    root = Path(root_text)
    # Reject a stale setting before it causes a confusing missing-file error
    if not root.is_dir():
        raise RuntimeError(f"AGROBOT_ROOT is not a directory: {root}")
    return root
