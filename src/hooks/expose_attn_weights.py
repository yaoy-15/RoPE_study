from __future__ import annotations

import argparse
import json
import logging
import re
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


OUTPUT_JSON = Path("outputs/reports/expose_attn_weights.json")
OUTPUT_MD = Path("outputs/reports/expose_attn_weights.md")


@dataclass
class AttnModuleSpec:
    module_name: str
    module_type: str
    layer_idx: int | None


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimentally expose attention weights via non-invasive eager config / monkey patch / wrapper."
    )
    parser.add_argument("--model_dir", type=Path, required=True, help="Local model directory.")
    parser.add_argument("--prompt", type=str, required=True, help="Single prompt for one forward pass.")
    parser.add_argument("--device", type=str, default="cpu", help="Device, e.g. cpu / cuda / cuda:0.")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to model/tokenizer loading.",
    )
    parser.add_argument(
        "--max_modules",
        type=int,
        default=512,
        help="Max attention-like modules to wrap for monkey patch probing.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def maybe_extract_layer_idx(module_name: str) -> int | None:
    patterns = (
        r"(?:^|\.)layers\.(\d+)(?:\.|$)",
        r"(?:^|\.)blocks\.(\d+)(?:\.|$)",
        r"(?:^|\.)h\.(\d+)(?:\.|$)",
    )
    for pat in patterns:
        match = re.search(pat, module_name)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
    return None


def summarize_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def evaluate_weight_like_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    result: dict[str, Any] = {
        "is_weight_like": False,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "reasons": [],
    }
    if tensor.ndim != 4:
        result["reasons"].append("ndim_is_not_4")
        return result
    if int(tensor.shape[-1]) < 2 or int(tensor.shape[-2]) < 2:
        result["reasons"].append("last_two_dims_too_small")
        return result
    result["reasons"].append("shape_looks_like_[batch,heads,q_len,k_len]")
    result["is_weight_like"] = True
    return result


def walk_tensors(obj: Any, path: str = "output", max_items: int = 8) -> list[tuple[str, torch.Tensor]]:
    found: list[tuple[str, torch.Tensor]] = []
    if isinstance(obj, torch.Tensor):
        found.append((path, obj))
        return found
    if isinstance(obj, (tuple, list)):
        for idx, item in enumerate(obj[:max_items]):
            found.extend(walk_tensors(item, f"{path}[{idx}]", max_items=max_items))
        return found
    if isinstance(obj, dict):
        for key in list(obj.keys())[:max_items]:
            found.extend(walk_tensors(obj[key], f"{path}.{key}", max_items=max_items))
    return found


def find_attention_modules(model: Any, max_modules: int) -> list[AttnModuleSpec]:
    specs: list[AttnModuleSpec] = []
    for name, module in model.named_modules():
        if not name:
            continue
        low = name.lower()
        if "self_attn" in low or ".attn" in low or "attention" in low:
            specs.append(
                AttnModuleSpec(
                    module_name=name,
                    module_type=module.__class__.__name__,
                    layer_idx=maybe_extract_layer_idx(name),
                )
            )
            if len(specs) >= max_modules:
                break
    return specs


def encode_inputs(tokenizer: Any, prompt: str, device: str) -> dict[str, Any]:
    encoded = tokenizer(prompt, return_tensors="pt")
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in encoded.items()}


def extract_attentions_from_forward_output(outputs: Any) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    attentions = getattr(outputs, "attentions", None)
    if attentions is None:
        return hits

    if isinstance(attentions, (tuple, list)):
        for idx, item in enumerate(attentions):
            if isinstance(item, torch.Tensor):
                score = evaluate_weight_like_tensor(item)
                if score["is_weight_like"]:
                    hits.append(
                        {
                            "layer_idx": idx,
                            "source": "forward_return.attentions",
                            "shape": score["shape"],
                            "dtype": score["dtype"],
                            "evidence": score["reasons"],
                        }
                    )
    elif isinstance(attentions, torch.Tensor):
        score = evaluate_weight_like_tensor(attentions)
        if score["is_weight_like"]:
            hits.append(
                {
                    "layer_idx": 0,
                    "source": "forward_return.attentions",
                    "shape": score["shape"],
                    "dtype": score["dtype"],
                    "evidence": score["reasons"],
                }
            )
    return hits


def method_forward_with_output_attentions(model: Any, encoded: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "method": "forward_output_attentions",
        "attempted": True,
        "success": False,
        "error": None,
        "attentions_attr_present": False,
        "attentions_count": 0,
        "weight_hits": [],
    }
    try:
        with torch.no_grad():
            outputs = model(**encoded, output_attentions=True, return_dict=True)
        attentions = getattr(outputs, "attentions", None)
        result["attentions_attr_present"] = attentions is not None
        if isinstance(attentions, (tuple, list)):
            result["attentions_count"] = len(attentions)
        elif attentions is None:
            result["attentions_count"] = 0
        else:
            result["attentions_count"] = 1
        hits = extract_attentions_from_forward_output(outputs)
        result["weight_hits"] = hits
        result["success"] = len(hits) > 0
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def set_attn_impl_eager_if_possible(model: Any) -> dict[str, Any]:
    changes: list[str] = []
    errors: list[str] = []

    try:
        if hasattr(model, "config") and hasattr(model.config, "attn_implementation"):
            before = getattr(model.config, "attn_implementation")
            setattr(model.config, "attn_implementation", "eager")
            changes.append(f"model.config.attn_implementation: {before} -> eager")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"model.config.attn_implementation_failed:{type(exc).__name__}:{exc}")

    try:
        if hasattr(model, "config") and hasattr(model.config, "_attn_implementation"):
            before = getattr(model.config, "_attn_implementation")
            setattr(model.config, "_attn_implementation", "eager")
            changes.append(f"model.config._attn_implementation: {before} -> eager")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"model.config._attn_implementation_failed:{type(exc).__name__}:{exc}")

    return {"changes": changes, "errors": errors}


def method_forward_with_eager_config(model: Any, encoded: dict[str, Any]) -> dict[str, Any]:
    set_result = set_attn_impl_eager_if_possible(model)
    base = method_forward_with_output_attentions(model, encoded)
    return {
        "method": "forward_output_attentions_with_eager_config",
        "attempted": True,
        "success": base["success"],
        "error": base["error"],
        "attentions_attr_present": base["attentions_attr_present"],
        "attentions_count": base["attentions_count"],
        "weight_hits": base["weight_hits"],
        "eager_config_changes": set_result["changes"],
        "eager_config_errors": set_result["errors"],
    }


def method_monkey_patch_attention_modules(
    model: Any,
    encoded: dict[str, Any],
    candidates: list[AttnModuleSpec],
) -> dict[str, Any]:
    captures: list[dict[str, Any]] = []
    patched: list[tuple[Any, Any, AttnModuleSpec]] = []

    def build_wrapper(spec: AttnModuleSpec, original_forward: Any):
        def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            out = original_forward(*args, **kwargs)
            tensors = walk_tensors(out)
            for path, tensor in tensors:
                check = evaluate_weight_like_tensor(tensor)
                if check["is_weight_like"]:
                    captures.append(
                        {
                            "module_name": spec.module_name,
                            "module_type": spec.module_type,
                            "layer_idx": spec.layer_idx,
                            "source": f"monkey_patch:{spec.module_name}:{path}",
                            "shape": check["shape"],
                            "dtype": check["dtype"],
                            "evidence": check["reasons"],
                        }
                    )
            return out

        return wrapped

    for spec in candidates:
        module = model.get_submodule(spec.module_name)
        original_forward = module.forward
        wrapper = build_wrapper(spec, original_forward)
        module.forward = types.MethodType(wrapper, module)
        patched.append((module, original_forward, spec))

    error = None
    try:
        with torch.no_grad():
            _ = model(**encoded, output_attentions=True, return_dict=True)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    finally:
        for module, original_forward, _spec in patched:
            module.forward = original_forward

    unique_hits: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in captures:
        key = (
            item["module_name"],
            item["layer_idx"],
            tuple(item["shape"]),
            item["dtype"],
            item["source"],
        )
        if key in seen:
            continue
        seen.add(key)
        unique_hits.append(item)

    return {
        "method": "monkey_patch_attention_forward_wrapper",
        "attempted": True,
        "success": len(unique_hits) > 0,
        "error": error,
        "patched_module_count": len(candidates),
        "weight_hits": unique_hits,
    }


def build_conclusion(method_results: list[dict[str, Any]]) -> dict[str, Any]:
    all_hits: list[dict[str, Any]] = []
    successful_methods: list[str] = []
    errors: list[str] = []
    for item in method_results:
        hits = item.get("weight_hits", [])
        if hits:
            successful_methods.append(item["method"])
            all_hits.extend(hits)
        if item.get("error"):
            errors.append(f"{item['method']}: {item['error']}")

    attn_weights_obtained = len(all_hits) > 0
    return {
        "attn_weights_obtained": attn_weights_obtained,
        "successful_methods": successful_methods,
        "method_count": len(method_results),
        "errors": errors,
        "hit_count": len(all_hits),
        "weight_hits": all_hits,
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    run_info = report["run_info"]
    conclusion = report["conclusion"]

    lines.append("# Expose Attention Weights Experiment Report")
    lines.append("")
    lines.append("## Run Info")
    lines.append(f"- model_dir: `{run_info['model_dir']}`")
    lines.append(f"- prompt: `{run_info['prompt']}`")
    lines.append(f"- device: `{run_info['device']}`")
    lines.append(f"- trust_remote_code: `{run_info['trust_remote_code']}`")
    lines.append("")
    lines.append("## Conclusion")
    lines.append(f"- attn_weights_obtained: `{conclusion['attn_weights_obtained']}`")
    lines.append(f"- successful_methods: `{conclusion['successful_methods']}`")
    lines.append(f"- hit_count: `{conclusion['hit_count']}`")
    if conclusion["errors"]:
        lines.append("- errors:")
        for err in conclusion["errors"]:
            lines.append(f"  - `{err}`")
    lines.append("")

    lines.append("## Methods")
    for item in report["method_results"]:
        lines.append(f"- method: `{item['method']}`")
        lines.append(f"  - attempted: `{item['attempted']}`")
        lines.append(f"  - success: `{item['success']}`")
        lines.append(f"  - error: `{item['error']}`")
        if "attentions_count" in item:
            lines.append(f"  - attentions_count: `{item['attentions_count']}`")
        if "patched_module_count" in item:
            lines.append(f"  - patched_module_count: `{item['patched_module_count']}`")
        lines.append(f"  - weight_hit_count: `{len(item.get('weight_hits', []))}`")
    lines.append("")

    lines.append("## Weight Hits")
    if conclusion["weight_hits"]:
        for hit in conclusion["weight_hits"]:
            lines.append(
                "- source: `{}` | layer_idx: `{}` | shape: `{}` | dtype: `{}`".format(
                    hit.get("source"),
                    hit.get("layer_idx"),
                    hit.get("shape"),
                    hit.get("dtype"),
                )
            )
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("## Notes")
    lines.append("- This script does not modify transformers source code.")
    lines.append("- If no hit is found, attn_weights remain unavailable under current non-invasive attempts.")
    lines.append("")
    return "\n".join(lines)


def save_json(data: dict[str, Any], path: Path) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_markdown(text: str, path: Path) -> None:
    ensure_parent(path)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    setup_logging()
    args = parse_args()

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
    attn_modules = find_attention_modules(model, max_modules=args.max_modules)
    logging.info("Found attention-like modules for monkey patch: %d", len(attn_modules))

    method_results: list[dict[str, Any]] = []

    logging.info("Method 1: forward(..., output_attentions=True)")
    m1 = method_forward_with_output_attentions(model, encoded)
    method_results.append(m1)
    logging.info("Method 1 success=%s, hits=%d", m1["success"], len(m1["weight_hits"]))

    logging.info("Method 2: eager config + forward(..., output_attentions=True)")
    m2 = method_forward_with_eager_config(model, encoded)
    method_results.append(m2)
    logging.info("Method 2 success=%s, hits=%d", m2["success"], len(m2["weight_hits"]))

    logging.info("Method 3: monkey patch attention forward wrappers")
    m3 = method_monkey_patch_attention_modules(model, encoded, attn_modules)
    method_results.append(m3)
    logging.info("Method 3 success=%s, hits=%d", m3["success"], len(m3["weight_hits"]))

    conclusion = build_conclusion(method_results)
    logging.info(
        "Overall: attn_weights_obtained=%s, total_hits=%d",
        conclusion["attn_weights_obtained"],
        conclusion["hit_count"],
    )

    report = {
        "run_info": {
            "model_dir": str(args.model_dir),
            "prompt": args.prompt,
            "device": args.device,
            "trust_remote_code": args.trust_remote_code,
        },
        "model_overview": {
            "model_class_name": model.__class__.__name__,
            "tokenizer_class_name": tokenizer.__class__.__name__,
        },
        "method_results": method_results,
        "conclusion": conclusion,
    }

    save_json(report, OUTPUT_JSON)
    save_markdown(to_markdown(report), OUTPUT_MD)
    logging.info("Saved JSON report: %s", OUTPUT_JSON)
    logging.info("Saved Markdown report: %s", OUTPUT_MD)


if __name__ == "__main__":
    main()
