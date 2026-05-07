from __future__ import annotations

from pathlib import Path

import torch


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return resolved


def device_report(device: torch.device) -> str:
    cuda_version = torch.version.cuda or "none"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    return (
        f"torch={torch.__version__} cuda_runtime={cuda_version} "
        f"cuda_available={torch.cuda.is_available()} selected_device={device} gpu={gpu_name}"
    )


def resolve_checkpoint(checkpoint: str | Path | None, run_dir: str | Path | None) -> Path:
    if checkpoint is not None:
        path = Path(checkpoint)
    elif run_dir is not None:
        path = Path(run_dir) / "checkpoints" / "latest.pt"
    else:
        raise ValueError("Provide either --checkpoint or --run-dir.")

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path
