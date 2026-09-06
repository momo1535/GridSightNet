"""Verify a GridSight checkpoint against a model configuration on CPU."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gridsight.core import YAMLConfig


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()

    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")

    if checkpoint.get("ema") is not None:
        state, source = checkpoint["ema"]["module"], "ema.module"
    elif checkpoint.get("model") is not None:
        state, source = checkpoint["model"], "model"
    else:
        raise KeyError("Checkpoint has neither ema.module nor model state")

    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["DINOv3STAs"]["weights_path"] = None
    model = cfg.model
    state = {
        key: value
        for key, value in state.items()
        if key not in {"decoder.anchors", "decoder.valid_mask"}
    }
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing = {"decoder.anchors", "decoder.valid_mask"}
    if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {incompatible}")

    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"checkpoint={args.checkpoint.resolve()}")
    print(f"state_source={source}")
    print(f"parameters={parameters}")
    print(f"sha256={sha256(args.checkpoint)}")
    print("status=ok")


if __name__ == "__main__":
    main()
