from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


OUTPUT_JSON = Path("outputs/reports/attn_weights_probe.json")
OUTPUT_MD = Path("outputs/reports/attn_weights_probe.md")


@dataclass
class ModuleSpec:
    name: str
    module_type: str
    layer_idx: int | None


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe whether attention weights can be obtained without modifying transformers source."
    )
    parser.add_argument("--model_dir", type=Path, required=True, help="Local model directory.")
    parser.add_argument("--prompt", type=str, required=True, help="Single prompt for one forward pass.")
    parser.add_argument("--device", type=str, default="cpu", help="Device, e.g. cpu / cuda / cuda:0.")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to model/tokenizer loading.",
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
        "kind": "tensor",
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def summarize_obj(obj: Any, depth: int = 0, max_depth: int = 2, max_items: int = 8) -> tuple[dict[str, Any], bool]:
    if isinstance(obj, torch.Tensor):
        return summarize_tensor(obj), False

    if depth >= max_depth:
        return {"kind": "max_depth_reached", "type": type(obj).__name__}, True

    if isinstance(obj, (tuple, list)):
        items: list[dict[str, Any]] = []
        complex_flag = False
        for idx, item in enumerate(obj[:max_items]):
            summary, is_complex = summarize_obj(item, depth + 1, max_depth, max_items)
            items.append({"index": idx, "summary": summary})
            complex_flag = complex_flag or is_complex
        if len(obj) > max_items:
            complex_flag = True
        return {
            "kind": type(obj).__name__,
            "len": len(obj),
            "items": items,
            "truncated": len(obj) > max_items,
        }, complex_flag

    if isinstance(obj, dict):
        keys = list(obj.keys())
        items: list[dict[str, Any]] = []
        complex_flag = False
        for key in keys[:max_items]:
            summary, is_complex = summarize_obj(obj[key], depth + 1, max_depth, max_items)
            items.append({"key": str(key), "summary": summary})
            complex_flag = complex_flag or is_complex
        if len(keys) > max_items:
            complex_flag = True
        return {
            "kind": "dict",
            "len": len(keys),
            "items": items,
            "truncated": len(keys) > max_items,
        }, complex_flag

    if obj is None:
        return {"kind": "none"}, False

    return {"kind": "other", "type": type(obj).__name__}, True


def evaluate_weight_like_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    record: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "is_weight_like": False,
        "reasons": [],
    }
    if tensor.ndim != 4:
        record["reasons"].append("ndim_is_not_4")
        return record

    q_len = int(tensor.shape[-2])
    k_len = int(tensor.shape[-1])
    if q_len < 2 or k_len < 2:
        record["reasons"].append("last_two_dims_too_small")
        return record

    record["reasons"].append("shape_looks_like_[batch,heads,q_len,k_len]")

    # Additional soft check: probability-like range/sum (best-effort).
    try:
        t = tensor.detach()
        if t.is_cuda:
            t = t.float().cpu()
        else:
            t = t.float()
        sample = t[:1, :1, : min(q_len, 4), : min(k_len, 16)]
        min_v = float(sample.min().item())
        max_v = float(sample.max().item())
        sum_last = sample.sum(dim=-1)
        max_dev = float((sum_last - 1.0).abs().max().item())
        prob_like_range = min_v >= -1e-3 and max_v <= 1.0 + 1e-3
        prob_like_sum = max_dev < 5e-2
        record["value_check"] = {
            "min": min_v,
            "max": max_v,
            "max_abs_dev_sum_lastdim_from_1": max_dev,
            "prob_like_range": prob_like_range,
            "prob_like_sum_lastdim": prob_like_sum,
        }
        if prob_like_range and prob_like_sum:
            record["is_weight_like"] = True
            record["reasons"].append("value_pattern_looks_like_attention_probabilities")
        else:
            record["reasons"].append("value_pattern_not_clearly_probability_like")
    except Exception as exc:  # noqa: BLE001
        record["reasons"].append(f"value_check_failed:{type(exc).__name__}")

    return record


def walk_tensors(obj: Any, path: str = "output", max_items: int = 8) -> list[tuple[str, torch.Tensor]]:
    found: list[tuple[str, torch.Tensor]] = []
    if isinstance(obj, torch.Tensor):
        found.append((path, obj))
        return found
    if isinstance(obj, (tuple, list)):
        for idx, item in enumerate(obj[:max_items]):
            found.extend(walk_tensors(item, path=f"{path}[{idx}]", max_items=max_items))
        return found
    if isinstance(obj, dict):
        keys = list(obj.keys())
        for key in keys[:max_items]:
            found.extend(walk_tensors(obj[key], path=f"{path}.{key}", max_items=max_items))
    return found


def find_attn_modules(model: Any) -> list[ModuleSpec]:
    modules: list[ModuleSpec] = []
    for name, module in model.named_modules():
        if not name:
            continue
        low = name.lower()
        if "self_attn" in low or ".attn" in low or "attention" in low:
            modules.append(
                ModuleSpec(
                    name=name,
                    module_type=module.__class__.__name__,
                    layer_idx=maybe_extract_layer_idx(name),
                )
            )
    return modules


def encode_prompt(tokenizer: Any, prompt: str, device: str) -> dict[str, Any]:
    encoded = tokenizer(prompt, return_tensors="pt")
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in encoded.items()}


def probe_forward_output_attentions(model: Any, encoded: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "attempted": True,
        "success": False,
        "error": None,
        "attentions_found": False,
        "attentions_type": None,
        "attentions_count": 0,
        "attentions_summaries": [],
        "weight_like_layer_candidates": [],
    }
    try:
        with torch.no_grad():
            outputs = model(**encoded, output_attentions=True, return_dict=True)
        attentions = getattr(outputs, "attentions", None)
        if attentions is None:
            result["success"] = True
            result["attentions_found"] = False
            return result

        result["success"] = True
        result["attentions_found"] = True
        result["attentions_type"] = type(attentions).__name__

        if isinstance(attentions, (tuple, list)):
            result["attentions_count"] = len(attentions)
            for idx, item in enumerate(attentions):
                item_summary, _ = summarize_obj(item)
                result["attentions_summaries"].append({"index": idx, "summary": item_summary})
                if isinstance(item, torch.Tensor):
                    score = evaluate_weight_like_tensor(item)
                    if score["is_weight_like"]:
                        result["weight_like_layer_candidates"].append(
                            {
                                "layer_idx": idx,
                                "shape": score["shape"],
                                "dtype": score["dtype"],
                                "evidence": score["reasons"],
                            }
                        )
        else:
            summary, _ = summarize_obj(attentions)
            result["attentions_summaries"].append({"index": 0, "summary": summary})
            if isinstance(attentions, torch.Tensor):
                score = evaluate_weight_like_tensor(attentions)
                if score["is_weight_like"]:
                    result["weight_like_layer_candidates"].append(
                        {
                            "layer_idx": 0,
                            "shape": score["shape"],
                            "dtype": score["dtype"],
                            "evidence": score["reasons"],
                        }
                    )
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def probe_hook_path(model: Any, encoded: dict[str, Any], candidates: list[ModuleSpec]) -> dict[str, Any]:
    hook_records: dict[str, dict[str, Any]] = {}
    handles = []

    def build_hook(spec: ModuleSpec):
        def _hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            summary, complex_flag = summarize_obj(output)
            rec = hook_records.setdefault(
                spec.name,
                {
                    "module_name": spec.name,
                    "module_type": spec.module_type,
                    "layer_idx": spec.layer_idx,
                    "call_count": 0,
                    "output_summary": summary,
                    "complex_output": complex_flag,
                    "weight_like_tensors": [],
                },
            )
            rec["call_count"] += 1
            tensors = walk_tensors(output)
            for tensor_path, tensor in tensors:
                score = evaluate_weight_like_tensor(tensor)
                if score["is_weight_like"]:
                    rec["weight_like_tensors"].append(
                        {
                            "path": tensor_path,
                            "shape": score["shape"],
                            "dtype": score["dtype"],
                            "evidence": score["reasons"],
                            "value_check": score.get("value_check"),
                        }
                    )

        return _hook

    for spec in candidates:
        module = model.get_submodule(spec.name)
        handles.append(module.register_forward_hook(build_hook(spec)))

    error = None
    try:
        with torch.no_grad():
            _ = model(**encoded)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    finally:
        for h in handles:
            h.remove()

    triggered_names = sorted(hook_records.keys())
    candidate_names = [s.name for s in candidates]
    untriggered_names = sorted(set(candidate_names) - set(triggered_names))
    weight_like_hits: list[dict[str, Any]] = []
    for name in triggered_names:
        rec = hook_records[name]
        for item in rec["weight_like_tensors"]:
            weight_like_hits.append(
                {
                    "module_name": rec["module_name"],
                    "module_type": rec["module_type"],
                    "layer_idx": rec["layer_idx"],
                    "path": item["path"],
                    "shape": item["shape"],
                    "dtype": item["dtype"],
                    "evidence": item["evidence"],
                    "value_check": item.get("value_check"),
                }
            )

    return {
        "attempted": True,
        "error": error,
        "registered_count": len(candidates),
        "triggered_count": len(triggered_names),
        "untriggered_count": len(untriggered_names),
        "triggered_module_names": triggered_names,
        "untriggered_module_names": untriggered_names,
        "hook_records": [hook_records[name] for name in triggered_names],
        "weight_like_hits": weight_like_hits,
    }


def build_conclusion(forward_probe: dict[str, Any], hook_probe: dict[str, Any]) -> dict[str, Any]:
    forward_hits = forward_probe.get("weight_like_layer_candidates", [])
    hook_hits = hook_probe.get("weight_like_hits", [])

    success = bool(forward_hits or hook_hits)
    source = "none"
    if forward_hits and hook_hits:
        source = "both_forward_and_hook"
    elif forward_hits:
        source = "forward_output_attentions"
    elif hook_hits:
        source = "hook"

    reasons: list[str] = []
    if not success:
        if forward_probe.get("error"):
            reasons.append(f"forward_output_attentions_error: {forward_probe['error']}")
        elif not forward_probe.get("attentions_found", False):
            reasons.append("forward_output_attentions_did_not_return_attentions")
        if hook_probe.get("error"):
            reasons.append(f"hook_forward_error: {hook_probe['error']}")
        if not hook_hits:
            reasons.append("hook_outputs_did_not_show_clear_weight_like_tensors")

    next_steps = []
    if not success:
        next_steps.append("Probe more granular hook points around attention internals with model-specific module names.")
        next_steps.append("If still unavailable, use non-invasive monkey patch on attention forward to expose weights.")

    return {
        "attn_weights_obtained": success,
        "source": source,
        "forward_weight_like_count": len(forward_hits),
        "hook_weight_like_count": len(hook_hits),
        "failure_reasons": reasons,
        "next_step_suggestions": next_steps,
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    run_info = report["run_info"]
    conclusion = report["conclusion"]
    forward_probe = report["forward_output_attentions_probe"]
    hook_probe = report["hook_probe"]

    lines.append("# Attention Weights Probe Report")
    lines.append("")
    lines.append("## Run Info")
    lines.append(f"- model_dir: `{run_info['model_dir']}`")
    lines.append(f"- device: `{run_info['device']}`")
    lines.append(f"- trust_remote_code: `{run_info['trust_remote_code']}`")
    lines.append(f"- prompt: `{run_info['prompt']}`")
    lines.append("")

    lines.append("## Conclusion")
    lines.append(f"- attn_weights_obtained: `{conclusion['attn_weights_obtained']}`")
    lines.append(f"- source: `{conclusion['source']}`")
    lines.append(f"- forward_weight_like_count: `{conclusion['forward_weight_like_count']}`")
    lines.append(f"- hook_weight_like_count: `{conclusion['hook_weight_like_count']}`")
    if conclusion["failure_reasons"]:
        lines.append("- failure_reasons:")
        for reason in conclusion["failure_reasons"]:
            lines.append(f"  - `{reason}`")
    if conclusion["next_step_suggestions"]:
        lines.append("- next_step_suggestions:")
        for item in conclusion["next_step_suggestions"]:
            lines.append(f"  - {item}")
    lines.append("")

    lines.append("## Probe A: forward(..., output_attentions=True)")
    lines.append(f"- attempted: `{forward_probe['attempted']}`")
    lines.append(f"- success: `{forward_probe['success']}`")
    lines.append(f"- error: `{forward_probe['error']}`")
    lines.append(f"- attentions_found: `{forward_probe['attentions_found']}`")
    lines.append(f"- attentions_type: `{forward_probe['attentions_type']}`")
    lines.append(f"- attentions_count: `{forward_probe['attentions_count']}`")
    if forward_probe["weight_like_layer_candidates"]:
        lines.append("- weight_like_layer_candidates:")
        for item in forward_probe["weight_like_layer_candidates"]:
            lines.append(
                f"  - layer_idx: `{item['layer_idx']}`, shape: `{item['shape']}`, dtype: `{item['dtype']}`"
            )
    else:
        lines.append("- weight_like_layer_candidates: (none)")
    lines.append("")

    lines.append("## Probe B: Hook Candidate Attention Modules")
    lines.append(f"- attempted: `{hook_probe['attempted']}`")
    lines.append(f"- error: `{hook_probe['error']}`")
    lines.append(f"- registered_count: `{hook_probe['registered_count']}`")
    lines.append(f"- triggered_count: `{hook_probe['triggered_count']}`")
    lines.append(f"- untriggered_count: `{hook_probe['untriggered_count']}`")
    lines.append(f"- weight_like_hit_count: `{len(hook_probe['weight_like_hits'])}`")
    if hook_probe["weight_like_hits"]:
        lines.append("- weight_like_hits:")
        for item in hook_probe["weight_like_hits"]:
            lines.append(
                "  - module: `{}` | layer_idx: `{}` | path: `{}` | shape: `{}` | dtype: `{}`".format(
                    item["module_name"],
                    item["layer_idx"],
                    item["path"],
                    item["shape"],
                    item["dtype"],
                )
            )
    else:
        lines.append("- weight_like_hits: (none)")
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

    encoded = encode_prompt(tokenizer, args.prompt, args.device)

    logging.info("Probe A: forward(..., output_attentions=True)")
    forward_probe = probe_forward_output_attentions(model, encoded)
    if forward_probe.get("error"):
        logging.warning("Probe A failed: %s", forward_probe["error"])
    else:
        logging.info(
            "Probe A completed: attentions_found=%s, weight_like_candidates=%d",
            forward_probe.get("attentions_found"),
            len(forward_probe.get("weight_like_layer_candidates", [])),
        )

    logging.info("Probe B: hook candidate attention modules")
    candidates = find_attn_modules(model)
    hook_probe = probe_hook_path(model, encoded, candidates)
    if hook_probe.get("error"):
        logging.warning("Probe B model forward error: %s", hook_probe["error"])
    logging.info(
        "Probe B completed: registered=%d, triggered=%d, weight_like_hits=%d",
        hook_probe["registered_count"],
        hook_probe["triggered_count"],
        len(hook_probe["weight_like_hits"]),
    )

    conclusion = build_conclusion(forward_probe, hook_probe)
    logging.info(
        "Conclusion: attn_weights_obtained=%s, source=%s",
        conclusion["attn_weights_obtained"],
        conclusion["source"],
    )

    report = {
        "run_info": {
            "model_dir": str(args.model_dir),
            "device": args.device,
            "trust_remote_code": args.trust_remote_code,
            "prompt": args.prompt,
        },
        "model_overview": {
            "model_class_name": model.__class__.__name__,
            "tokenizer_class_name": tokenizer.__class__.__name__,
        },
        "forward_output_attentions_probe": forward_probe,
        "hook_probe": hook_probe,
        "conclusion": conclusion,
    }

    save_json(report, OUTPUT_JSON)
    save_markdown(to_markdown(report), OUTPUT_MD)
    logging.info("Saved JSON report: %s", OUTPUT_JSON)
    logging.info("Saved Markdown report: %s", OUTPUT_MD)


if __name__ == "__main__":
    main()
