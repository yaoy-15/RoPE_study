from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_OUTPUT_JSON = Path("outputs/reports/rope_qk_path_probe.json")
DEFAULT_OUTPUT_MD = Path("outputs/reports/rope_qk_path_probe.md")
LAYER_MODULE_PATTERN = re.compile(r"^model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|q_norm|k_norm)$")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Joint probe for RoPE and Q/K path modules in one forward pass."
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


def summarize_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "kind": "tensor",
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }


def summarize_output(output: Any, max_items: int = 8) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "output_type": type(output).__name__,
    }

    if isinstance(output, torch.Tensor):
        summary["structure"] = summarize_tensor(output)
        return summary

    if isinstance(output, (tuple, list)):
        items: list[dict[str, Any]] = []
        for idx, item in enumerate(output[:max_items]):
            item_summary: dict[str, Any] = {
                "index": idx,
                "item_type": type(item).__name__,
            }
            if isinstance(item, torch.Tensor):
                item_summary["tensor"] = summarize_tensor(item)
            items.append(item_summary)
        summary["structure"] = {
            "kind": type(output).__name__,
            "len": len(output),
            "items": items,
            "truncated": len(output) > max_items,
        }
        return summary

    summary["structure"] = {
        "kind": "other",
        "type": type(output).__name__,
    }
    return summary


def collect_target_module_names(model: Any) -> list[str]:
    names: list[str] = []
    rotary_name = "model.rotary_emb"
    try:
        _ = model.get_submodule(rotary_name)
        names.append(rotary_name)
    except Exception:  # noqa: BLE001
        pass

    for module_name, _ in model.named_modules():
        if LAYER_MODULE_PATTERN.match(module_name):
            names.append(module_name)

    return sorted(set(names))


def build_hook(
    module_name: str,
    module_type: str,
    call_counter: dict[str, int],
    records: dict[str, dict[str, Any]],
) -> Any:
    def _hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        call_counter["value"] += 1
        current = call_counter["value"]
        output_summary = summarize_output(output)

        record = records.setdefault(
            module_name,
            {
                "module_name": module_name,
                "module_type": module_type,
                "call_count": 0,
                "first_call_order": None,
                "last_call_order": None,
                "output_type": output_summary["output_type"],
                "output_samples": [],
            },
        )
        record["call_count"] += 1
        if record["first_call_order"] is None:
            record["first_call_order"] = current
        record["last_call_order"] = current
        record["output_type"] = output_summary["output_type"]
        if len(record["output_samples"]) < 2:
            record["output_samples"].append(output_summary["structure"])

    return _hook


def run_single_forward(model: Any, tokenizer: Any, prompt: str, device: str) -> dict[str, Any]:
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in encoded.items()}
    with torch.no_grad():
        _ = model(**encoded)
    return {
        "input_keys": sorted(encoded.keys()),
        "input_shape_map": {
            k: list(v.shape) for k, v in encoded.items() if isinstance(v, torch.Tensor)
        },
    }


def parse_layer_slot(module_name: str) -> tuple[int, str] | None:
    match = LAYER_MODULE_PATTERN.match(module_name)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def build_layer_call_map(records: dict[str, dict[str, Any]]) -> dict[int, dict[str, int | None]]:
    layer_map: dict[int, dict[str, int | None]] = {}
    for name, rec in records.items():
        parsed = parse_layer_slot(name)
        if parsed is None:
            continue
        layer_idx, slot = parsed
        layer_info = layer_map.setdefault(
            layer_idx,
            {"q_proj": None, "k_proj": None, "q_norm": None, "k_norm": None},
        )
        value = rec.get("first_call_order")
        layer_info[slot] = value if isinstance(value, int) else None
    return layer_map


def classify_post_rope_candidates(
    rotary_first: int | None,
    layer_map: dict[int, dict[str, int | None]],
) -> dict[str, Any]:
    if rotary_first is None:
        return {
            "classification": "暂无可靠候选",
            "reason": "model.rotary_emb 未触发，无法在本次探测中定位 RoPE 与 Q/K 路径的相对顺序。",
            "candidate_modules": [],
        }

    evidence: list[dict[str, Any]] = []
    precise_count = 0
    approx_count = 0

    for layer_idx in sorted(layer_map):
        item = layer_map[layer_idx]
        q_proj = item.get("q_proj")
        k_proj = item.get("k_proj")
        q_norm = item.get("q_norm")
        k_norm = item.get("k_norm")
        layer_evidence = {
            "layer_idx": layer_idx,
            "q_proj": q_proj,
            "k_proj": k_proj,
            "q_norm": q_norm,
            "k_norm": k_norm,
            "q_norm_after_rotary": isinstance(q_norm, int) and q_norm > rotary_first,
            "k_norm_after_rotary": isinstance(k_norm, int) and k_norm > rotary_first,
            "qk_before_rotary": (
                isinstance(q_proj, int)
                and isinstance(k_proj, int)
                and q_proj < rotary_first
                and k_proj < rotary_first
            ),
            "norm_after_qk": (
                isinstance(q_proj, int)
                and isinstance(k_proj, int)
                and isinstance(q_norm, int)
                and isinstance(k_norm, int)
                and q_norm > q_proj
                and k_norm > k_proj
            ),
        }
        if (
            layer_evidence["qk_before_rotary"]
            and layer_evidence["q_norm_after_rotary"]
            and layer_evidence["k_norm_after_rotary"]
        ):
            precise_count += 1
        elif layer_evidence["norm_after_qk"]:
            approx_count += 1

        evidence.append(layer_evidence)

    if precise_count > 0:
        return {
            "classification": "精确候选",
            "reason": (
                "至少一个层内满足 q_proj/k_proj 在 rotary_emb 前，且 q_norm/k_norm 在 rotary_emb 后，"
                "q_norm/k_norm 可作为更强的 post-rope 候选。"
            ),
            "candidate_modules": ["model.layers.{i}.self_attn.q_norm", "model.layers.{i}.self_attn.k_norm"],
            "evidence_counts": {
                "precise_layer_count": precise_count,
                "approx_layer_count": approx_count,
            },
        }

    if approx_count > 0:
        return {
            "classification": "近似候选",
            "reason": (
                "未观察到稳定的 rotary_emb 夹在 q_proj/k_proj 与 q_norm/k_norm 之间；"
                "但 q_norm/k_norm 在调用顺序上普遍晚于 q_proj/k_proj，可作为近似 post-rope 候选。"
            ),
            "candidate_modules": ["model.layers.{i}.self_attn.q_norm", "model.layers.{i}.self_attn.k_norm"],
            "evidence_counts": {
                "precise_layer_count": precise_count,
                "approx_layer_count": approx_count,
            },
        }

    return {
        "classification": "暂无可靠候选",
        "reason": "本次调用顺序证据不足，无法稳定定位可无侵入 hook 的 post-rope 候选点。",
        "candidate_modules": [],
        "evidence_counts": {
            "precise_layer_count": precise_count,
            "approx_layer_count": approx_count,
        },
    }


def analyze_report(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rotary_record = records.get("model.rotary_emb")
    rotary_first = None
    rotary_last = None
    if rotary_record is not None:
        first = rotary_record.get("first_call_order")
        last = rotary_record.get("last_call_order")
        rotary_first = first if isinstance(first, int) else None
        rotary_last = last if isinstance(last, int) else None

    q_firsts: list[int] = []
    k_firsts: list[int] = []
    qn_firsts: list[int] = []
    kn_firsts: list[int] = []
    for name, rec in records.items():
        parsed = parse_layer_slot(name)
        if parsed is None:
            continue
        _, slot = parsed
        first = rec.get("first_call_order")
        if not isinstance(first, int):
            continue
        if slot == "q_proj":
            q_firsts.append(first)
        elif slot == "k_proj":
            k_firsts.append(first)
        elif slot == "q_norm":
            qn_firsts.append(first)
        elif slot == "k_norm":
            kn_firsts.append(first)

    layer_map = build_layer_call_map(records)
    candidate_classification = classify_post_rope_candidates(rotary_first=rotary_first, layer_map=layer_map)

    relative_summary = {
        "rotary_first_call_order": rotary_first,
        "rotary_last_call_order": rotary_last,
        "q_proj_first_call_min": min(q_firsts) if q_firsts else None,
        "q_proj_first_call_max": max(q_firsts) if q_firsts else None,
        "k_proj_first_call_min": min(k_firsts) if k_firsts else None,
        "k_proj_first_call_max": max(k_firsts) if k_firsts else None,
        "q_norm_first_call_min": min(qn_firsts) if qn_firsts else None,
        "q_norm_first_call_max": max(qn_firsts) if qn_firsts else None,
        "k_norm_first_call_min": min(kn_firsts) if kn_firsts else None,
        "k_norm_first_call_max": max(kn_firsts) if kn_firsts else None,
    }

    return {
        "relative_position_summary": relative_summary,
        "per_layer_call_orders": [
            {
                "layer_idx": i,
                "q_proj": layer_map[i]["q_proj"],
                "k_proj": layer_map[i]["k_proj"],
                "q_norm": layer_map[i]["q_norm"],
                "k_norm": layer_map[i]["k_norm"],
            }
            for i in sorted(layer_map)
        ],
        "post_rope_candidate_assessment": candidate_classification,
    }


def build_report(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    target_modules: list[dict[str, str]],
    records: dict[str, dict[str, Any]],
    forward_input: dict[str, Any],
) -> dict[str, Any]:
    analysis = analyze_report(records)
    return {
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
        "probe_scope": {
            "target_rule": [
                "model.rotary_emb",
                "model.layers.{i}.self_attn.q_proj",
                "model.layers.{i}.self_attn.k_proj",
                "model.layers.{i}.self_attn.q_norm",
                "model.layers.{i}.self_attn.k_norm",
            ],
            "target_module_count": len(target_modules),
            "target_modules": target_modules,
        },
        "hook_records": {
            "triggered_module_count": len(records),
            "records": [records[k] for k in sorted(records.keys())],
            "forward_input": forward_input,
        },
        "analysis": analysis,
        "limitations": [
            "This probe does not implement q_after_rope / k_after_rope capture.",
            "Forward-hook call order is runtime evidence, not a formal computational graph proof.",
            "If kernels are fused or paths are inlined, exact tensor boundaries may remain ambiguous.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    run_info = report["run_info"]
    overview = report["model_overview"]
    scope = report["probe_scope"]
    hook = report["hook_records"]
    analysis = report["analysis"]
    rel = analysis["relative_position_summary"]
    candidate = analysis["post_rope_candidate_assessment"]

    lines: list[str] = []
    lines.append("# RoPE-QK Path Probe Report")
    lines.append("")
    lines.append("## Run Info")
    lines.append(f"- model_dir: `{run_info['model_dir']}`")
    lines.append(f"- device: `{run_info['device']}`")
    lines.append(f"- trust_remote_code: `{run_info['trust_remote_code']}`")
    lines.append(f"- prompt: `{run_info['prompt']}`")
    lines.append("")
    lines.append("## Model Overview")
    lines.append(f"- model class: `{overview['model_class_name']}`")
    lines.append(f"- tokenizer class: `{overview['tokenizer_class_name']}`")
    lines.append("")
    lines.append("## Probe Scope")
    lines.append(f"- target module count: `{scope['target_module_count']}`")
    lines.append(f"- triggered module count: `{hook['triggered_module_count']}`")
    lines.append("")
    lines.append("## Relative Order Summary")
    lines.append(f"- rotary first call: `{rel['rotary_first_call_order']}`")
    lines.append(f"- q_proj first call range: `[{rel['q_proj_first_call_min']}, {rel['q_proj_first_call_max']}]`")
    lines.append(f"- k_proj first call range: `[{rel['k_proj_first_call_min']}, {rel['k_proj_first_call_max']}]`")
    lines.append(f"- q_norm first call range: `[{rel['q_norm_first_call_min']}, {rel['q_norm_first_call_max']}]`")
    lines.append(f"- k_norm first call range: `[{rel['k_norm_first_call_min']}, {rel['k_norm_first_call_max']}]`")
    lines.append("")
    lines.append("## Candidate Assessment")
    lines.append(f"- classification: **{candidate['classification']}**")
    lines.append(f"- reason: {candidate['reason']}")
    lines.append(f"- candidate_modules: `{candidate.get('candidate_modules', [])}`")
    lines.append("")
    lines.append("## Per-Layer Orders (first 12)")
    for item in analysis["per_layer_call_orders"][:12]:
        lines.append(
            f"- layer {item['layer_idx']}: q_proj={item['q_proj']}, "
            f"k_proj={item['k_proj']}, q_norm={item['q_norm']}, k_norm={item['k_norm']}"
        )
    lines.append("")
    lines.append("## Limitations")
    for text in report["limitations"]:
        lines.append(f"- {text}")

    return "\n".join(lines) + "\n"


def save_json(path: Path, data: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_text(path: Path, text: str) -> None:
    ensure_parent(path)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    setup_logging()
    args = parse_args()

    logging.info("Loading tokenizer from %s", args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_dir),
        trust_remote_code=args.trust_remote_code,
    )

    logging.info("Loading model from %s", args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir),
        trust_remote_code=args.trust_remote_code,
    )
    model.to(args.device)
    model.eval()

    logging.info("Collecting target modules...")
    target_names = collect_target_module_names(model)
    target_modules: list[dict[str, str]] = []
    for name in target_names:
        module = model.get_submodule(name)
        target_modules.append(
            {
                "module_name": name,
                "module_type": module.__class__.__name__,
            }
        )
    logging.info("Target module count: %d", len(target_modules))

    call_counter: dict[str, int] = {"value": 0}
    records: dict[str, dict[str, Any]] = {}
    handles: list[Any] = []
    try:
        for item in target_modules:
            name = item["module_name"]
            module = model.get_submodule(name)
            handle = module.register_forward_hook(
                build_hook(
                    module_name=name,
                    module_type=item["module_type"],
                    call_counter=call_counter,
                    records=records,
                )
            )
            handles.append(handle)

        logging.info("Running one forward pass...")
        forward_input = run_single_forward(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            device=args.device,
        )
    finally:
        for handle in handles:
            handle.remove()

    report = build_report(
        args=args,
        model=model,
        tokenizer=tokenizer,
        target_modules=target_modules,
        records=records,
        forward_input=forward_input,
    )
    markdown = render_markdown(report)

    save_json(DEFAULT_OUTPUT_JSON, report)
    save_text(DEFAULT_OUTPUT_MD, markdown)

    logging.info("Saved JSON report to %s", DEFAULT_OUTPUT_JSON)
    logging.info("Saved Markdown report to %s", DEFAULT_OUTPUT_MD)


if __name__ == "__main__":
    main()
