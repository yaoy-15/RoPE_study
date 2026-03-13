from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SUPPORTED_TENSORS = {"q", "k", "v", "attn_weights"}
TENSOR_TO_MODULE_SUFFIX = {
    "q": "q_proj",
    "k": "k_proj",
    "v": "v_proj",
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture q/k/v projection outputs or attn_weights for Qwen attention modules."
    )
    parser.add_argument("--model_type", type=str, required=True, help="Model type tag (for run metadata).")
    parser.add_argument("--model_dir", type=Path, required=True, help="Local model directory.")
    parser.add_argument("--prompt", type=str, required=True, help="Single prompt for one forward pass.")
    parser.add_argument("--prompt_name", type=str, required=True, help="Prompt name for output directory.")
    parser.add_argument("--layer_idx", type=int, required=True, help="Target layer index.")
    parser.add_argument(
        "--tensor_name",
        type=str,
        required=True,
        choices=sorted(SUPPORTED_TENSORS),
        help="Tensor to capture: q / k / v / attn_weights.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device, e.g. cpu / cuda / cuda:0.")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to model/tokenizer loading.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sanitize_prompt_name(name: str) -> str:
    keep = []
    for ch in name.strip():
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    cleaned = "".join(keep).strip("_")
    return cleaned or "prompt"


def build_source_module_name(layer_idx: int, tensor_name: str) -> str:
    if tensor_name not in TENSOR_TO_MODULE_SUFFIX:
        raise ValueError(f"tensor_name {tensor_name} does not map to a projection module.")
    suffix = TENSOR_TO_MODULE_SUFFIX[tensor_name]
    return f"model.layers.{layer_idx}.self_attn.{suffix}"


def encode_inputs(tokenizer: Any, prompt: str, device: str) -> dict[str, Any]:
    encoded = tokenizer(prompt, return_tensors="pt")
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in encoded.items()}


def run_single_forward(model: Any, encoded: dict[str, Any]) -> None:
    with torch.no_grad():
        _ = model(**encoded)


def set_eager_attention_or_raise(model: Any) -> None:
    changed = False
    errors: list[str] = []
    for field in ("attn_implementation", "_attn_implementation"):
        if hasattr(model, "config") and hasattr(model.config, field):
            try:
                setattr(model.config, field, "eager")
                changed = True
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{field}:{type(exc).__name__}:{exc}")
    if not changed:
        detail = "; ".join(errors) if errors else "no configurable attn_implementation field found"
        raise RuntimeError(
            "Failed to switch attention implementation to eager. "
            "Current environment may be on sdpa, where output_attentions=True is unsupported. "
            f"Details: {detail}"
        )


def capture_projection_tensor(
    model: Any,
    encoded: dict[str, Any],
    layer_idx: int,
    tensor_name: str,
) -> tuple[torch.Tensor, str]:
    source_module = build_source_module_name(layer_idx=layer_idx, tensor_name=tensor_name)
    logging.info("Resolving source module: %s", source_module)
    module = model.get_submodule(source_module)

    capture_store: dict[str, torch.Tensor] = {}

    def _hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                f"Expected Tensor output from {source_module}, but got {type(output).__name__}."
            )
        # IMPORTANT: We capture the linear projection output of q_proj/k_proj/v_proj.
        # Whether it strictly equals RoPE-before tensors is still unconfirmed.
        capture_store["tensor"] = output.detach().cpu().clone()

    handle = module.register_forward_hook(_hook)
    try:
        logging.info("Running one forward pass for projection capture...")
        run_single_forward(model, encoded)
    finally:
        handle.remove()

    if "tensor" not in capture_store:
        raise RuntimeError(f"Hook did not capture any tensor from module: {source_module}")
    return capture_store["tensor"], source_module


def capture_attn_weights_tensor(
    model: Any,
    encoded: dict[str, Any],
    layer_idx: int,
) -> torch.Tensor:
    set_eager_attention_or_raise(model)
    logging.info("Running one forward pass with output_attentions=True (eager attention expected)...")
    with torch.no_grad():
        outputs = model(**encoded, output_attentions=True, return_dict=True)

    attentions = getattr(outputs, "attentions", None)
    if attentions is None or not isinstance(attentions, (tuple, list)) or len(attentions) == 0:
        raise RuntimeError(
            "outputs.attentions is empty after eager attention attempt. "
            "Current environment default sdpa does not reliably support output_attentions=True."
        )
    if layer_idx < 0 or layer_idx >= len(attentions):
        raise IndexError(
            f"layer_idx={layer_idx} is out of range for outputs.attentions (len={len(attentions)})."
        )
    target = attentions[layer_idx]
    if not isinstance(target, torch.Tensor):
        raise TypeError(
            f"outputs.attentions[{layer_idx}] is not a Tensor, got {type(target).__name__}."
        )
    return target.detach().cpu().clone()


def semantic_check_attn_weights(tensor: torch.Tensor, atol: float = 1e-2) -> dict[str, Any]:
    if tensor.ndim != 4:
        return {
            "semantic_check_passed": False,
            "semantic_check_summary": "expected_4d_tensor_for_attention_weights",
            "value_min": float(tensor.min().item()),
            "value_max": float(tensor.max().item()),
            "last_dim_sum_mean": None,
            "last_dim_sum_std": None,
        }

    t = tensor.float()
    sum_last = t.sum(dim=-1)
    sum_mean = float(sum_last.mean().item())
    sum_std = float(sum_last.std(unbiased=False).item())
    min_v = float(t.min().item())
    max_v = float(t.max().item())

    close_to_one = bool(torch.allclose(sum_last, torch.ones_like(sum_last), atol=atol, rtol=0.0))
    passed = close_to_one
    summary = (
        "looks_like_post_softmax_attention_weights"
        if passed
        else "last_dim_prob_sum_not_close_to_1_or_shape_unexpected"
    )
    return {
        "semantic_check_passed": passed,
        "semantic_check_summary": summary,
        "value_min": min_v,
        "value_max": max_v,
        "last_dim_sum_mean": sum_mean,
        "last_dim_sum_std": sum_std,
    }


def save_metadata(path: Path, metadata: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    setup_logging()
    args = parse_args()

    tensor_name = args.tensor_name
    layer_idx = args.layer_idx

    logging.info("Loading tokenizer from: %s", args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_dir),
        trust_remote_code=args.trust_remote_code,
    )

    logging.info("Loading model from: %s", args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir),
        trust_remote_code=args.trust_remote_code,
    )
    model = model.to(args.device)
    model.eval()

    encoded = encode_inputs(tokenizer, args.prompt, args.device)
    source_field: str | None = None
    semantic_info: dict[str, Any] | None = None
    if tensor_name in {"q", "k", "v"}:
        captured, source_module = capture_projection_tensor(
            model=model,
            encoded=encoded,
            layer_idx=layer_idx,
            tensor_name=tensor_name,
        )
        tensor_semantics = "projection_output"
        capture_method = "forward_hook"
        rope_stage = "uncertain"
    elif tensor_name == "attn_weights":
        captured = capture_attn_weights_tensor(
            model=model,
            encoded=encoded,
            layer_idx=layer_idx,
        )
        semantic_info = semantic_check_attn_weights(captured)
        source_module = None
        source_field = "outputs.attentions[layer_idx]"
        tensor_semantics = "attention_weights"
        capture_method = "forward_output_attentions_with_eager_config"
        rope_stage = "not_applicable"
    else:
        raise ValueError(f"Unsupported tensor_name: {tensor_name}")

    model_name = args.model_dir.name
    safe_prompt_name = sanitize_prompt_name(args.prompt_name)
    out_dir = Path("outputs") / "tensors" / model_name / safe_prompt_name
    tensor_path = out_dir / f"layer_{layer_idx}_{tensor_name}.pt"
    metadata_path = out_dir / "metadata.json"
    # NOTE: metadata is currently a single-file write pattern suited for smoke tests.
    # If capturing multiple layers/tensors under the same prompt_name, later writes overwrite earlier metadata.

    ensure_parent(tensor_path)
    torch.save(captured, tensor_path)

    metadata = {
        "prompt": args.prompt,
        "prompt_name": args.prompt_name,
        "model_name": model_name,
        "model_type": args.model_type,
        "layer": layer_idx,
        "tensor_name": tensor_name,
        "shape": list(captured.shape),
        "dtype": str(captured.dtype),
        "rope_stage": rope_stage,
        "tensor_semantics": tensor_semantics,
        "source_module": source_module,
        "source_field": source_field,
        "capture_method": capture_method,
        "tensor_file": str(tensor_path),
    }
    if semantic_info is not None:
        metadata.update(
            {
                "semantic_check_passed": semantic_info["semantic_check_passed"],
                "semantic_check_summary": semantic_info["semantic_check_summary"],
                "value_min": semantic_info["value_min"],
                "value_max": semantic_info["value_max"],
                "last_dim_sum_mean": semantic_info["last_dim_sum_mean"],
                "last_dim_sum_std": semantic_info["last_dim_sum_std"],
            }
        )
    save_metadata(metadata_path, metadata)

    logging.info("Captured tensor shape: %s", list(captured.shape))
    logging.info("Captured tensor dtype: %s", captured.dtype)
    if tensor_name == "attn_weights" and semantic_info is not None:
        logging.info(
            "attn_weights semantic check | shape=%s | dtype=%s | last_dim_sum_mean=%.6f | last_dim_sum_std=%.6f | passed=%s",
            list(captured.shape),
            captured.dtype,
            semantic_info["last_dim_sum_mean"] if semantic_info["last_dim_sum_mean"] is not None else float("nan"),
            semantic_info["last_dim_sum_std"] if semantic_info["last_dim_sum_std"] is not None else float("nan"),
            semantic_info["semantic_check_passed"],
        )
    logging.info("Saved tensor: %s", tensor_path)
    logging.info("Saved metadata: %s", metadata_path)


if __name__ == "__main__":
    main()
