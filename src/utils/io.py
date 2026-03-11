from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def _to_path(path: Path | str) -> Path:
    return Path(path)


def save_tensor(tensor: Any, path: Path | str) -> Path:
    """
    Save tensor-like object with torch.save and ensure parent directory exists.
    """
    target = _to_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, target)
    return target


def save_json(obj: Any, path: Path | str) -> Path:
    """
    Save JSON with utf-8 encoding, indent=2, ensure_ascii=False.
    """
    target = _to_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return target


def tensor_metadata_dict(
    *,
    model_name: str,
    prompt_name: str,
    layer_index: int | None,
    tensor_name: str,
    rope_stage: str | None,
    source_module: str | None,
    tensor: Any = None,
) -> dict[str, Any]:
    """
    Build a metadata dictionary for tensor export.
    If `tensor` is not a torch.Tensor, shape/dtype/device are set to None.
    """
    if isinstance(tensor, torch.Tensor):
        shape: list[int] | None = list(tensor.shape)
        dtype: str | None = str(tensor.dtype)
        device: str | None = str(tensor.device)
    else:
        shape = None
        dtype = None
        device = None

    return {
        "model_name": model_name,
        "prompt_name": prompt_name,
        "layer_index": layer_index,
        "tensor_name": tensor_name,
        "shape": shape,
        "dtype": dtype,
        "device": device,
        "rope_stage": rope_stage,
        "source_module": source_module,
    }
