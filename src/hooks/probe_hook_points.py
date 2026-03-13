from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_OUTPUT_JSON = Path("outputs/reports/hook_probe_report.json")
DEFAULT_OUTPUT_MD = Path("outputs/reports/hook_probe_report.md")


@dataclass
class HookSpec:
    module_name: str
    module_type: str


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe forward-hook output structures for candidate attention modules."
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
        "--module_name_patterns",
        type=str,
        nargs="+",
        required=True,
        help="Keyword patterns used to filter module names (case-insensitive substring match).",
    )
    parser.add_argument(
        "--max_hooks",
        type=int,
        default=64,
        help="Maximum number of hooks to register after filtering.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def matches_any_pattern(name: str, patterns: list[str]) -> bool:
    low_name = name.lower()
    return any(p.lower() in low_name for p in patterns)


def pick_candidate_modules(model: Any, patterns: list[str], max_hooks: int) -> list[HookSpec]:
    matches: list[HookSpec] = []
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        if matches_any_pattern(module_name, patterns):
            matches.append(HookSpec(module_name=module_name, module_type=module.__class__.__name__))
            if len(matches) >= max_hooks:
                break
    return matches


def summarize_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "kind": "tensor",
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def summarize_structure(obj: Any, depth: int = 0, max_depth: int = 2, max_items: int = 8) -> tuple[dict[str, Any], bool]:
    if isinstance(obj, torch.Tensor):
        return summarize_tensor(obj), False

    if depth >= max_depth:
        return {"kind": "max_depth_reached", "type": type(obj).__name__}, True

    if isinstance(obj, (tuple, list)):
        items: list[dict[str, Any]] = []
        is_complex = False
        limited = obj[:max_items]
        for idx, item in enumerate(limited):
            item_summary, item_complex = summarize_structure(item, depth=depth + 1, max_depth=max_depth, max_items=max_items)
            items.append({"index": idx, "summary": item_summary})
            is_complex = is_complex or item_complex
        if len(obj) > max_items:
            is_complex = True
        return {
            "kind": type(obj).__name__,
            "len": len(obj),
            "items": items,
            "truncated": len(obj) > max_items,
        }, is_complex

    if isinstance(obj, dict):
        keys = list(obj.keys())
        limited_keys = keys[:max_items]
        values: list[dict[str, Any]] = []
        is_complex = False
        for key in limited_keys:
            value_summary, value_complex = summarize_structure(
                obj[key], depth=depth + 1, max_depth=max_depth, max_items=max_items
            )
            values.append({"key": str(key), "summary": value_summary})
            is_complex = is_complex or value_complex
        if len(keys) > max_items:
            is_complex = True
        return {
            "kind": "dict",
            "len": len(keys),
            "items": values,
            "truncated": len(keys) > max_items,
        }, is_complex

    if obj is None:
        return {"kind": "none"}, False

    return {"kind": "other", "type": type(obj).__name__}, True


def build_hook(
    module_name: str,
    module_type: str,
    trigger_store: dict[str, dict[str, Any]],
) -> Any:
    def _hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        summary, is_complex = summarize_structure(output)
        record = trigger_store.setdefault(
            module_name,
            {
                "module_name": module_name,
                "module_type": module_type,
                "call_count": 0,
                "output_type": type(output).__name__,
                "output_summaries": [],
                "complex_output": False,
            },
        )
        record["call_count"] += 1
        if len(record["output_summaries"]) < 2:
            record["output_summaries"].append(summary)
        record["complex_output"] = bool(record["complex_output"] or is_complex)

    return _hook


def run_single_forward(model: Any, tokenizer: Any, prompt: str, device: str) -> None:
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in encoded.items()}
    with torch.no_grad():
        _ = model(**encoded)


def build_report(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    candidates: list[HookSpec],
    trigger_store: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    registered_names = [c.module_name for c in candidates]
    triggered_names = sorted(trigger_store.keys())
    untriggered_names = sorted(set(registered_names) - set(triggered_names))
    complex_names = sorted(
        name for name, rec in trigger_store.items() if bool(rec.get("complex_output", False))
    )

    return {
        "run_info": {
            "model_dir": str(args.model_dir),
            "device": args.device,
            "trust_remote_code": args.trust_remote_code,
            "prompt": args.prompt,
            "module_name_patterns": args.module_name_patterns,
            "max_hooks": args.max_hooks,
        },
        "model_overview": {
            "model_class_name": model.__class__.__name__,
            "tokenizer_class_name": tokenizer.__class__.__name__,
        },
        "hook_registration": {
            "registered_count": len(candidates),
            "registered_modules": [
                {"module_name": c.module_name, "module_type": c.module_type} for c in candidates
            ],
        },
        "hook_trigger_summary": {
            "triggered_count": len(triggered_names),
            "untriggered_count": len(untriggered_names),
            "complex_output_count": len(complex_names),
            "triggered_module_names": triggered_names,
            "untriggered_module_names": untriggered_names,
            "complex_output_module_names": complex_names,
        },
        "hook_records": [trigger_store[name] for name in triggered_names],
        "limitations": [
            "This is a probe tool for output structure only; it does not save formal q/k/v activations.",
            "q_after_rope / k_after_rope are NOT guaranteed here and may be unavailable without deeper model-specific instrumentation.",
        ],
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    run_info = report["run_info"]
    trigger_summary = report["hook_trigger_summary"]

    lines.append("# Hook Probe Report")
    lines.append("")
    lines.append("## Run Info")
    lines.append(f"- model_dir: `{run_info['model_dir']}`")
    lines.append(f"- device: `{run_info['device']}`")
    lines.append(f"- trust_remote_code: `{run_info['trust_remote_code']}`")
    lines.append(f"- prompt: `{run_info['prompt']}`")
    lines.append(f"- module_name_patterns: `{run_info['module_name_patterns']}`")
    lines.append(f"- max_hooks: `{run_info['max_hooks']}`")
    lines.append("")
    lines.append("## Trigger Summary")
    lines.append(f"- registered_count: `{report['hook_registration']['registered_count']}`")
    lines.append(f"- triggered_count: `{trigger_summary['triggered_count']}`")
    lines.append(f"- untriggered_count: `{trigger_summary['untriggered_count']}`")
    lines.append(f"- complex_output_count: `{trigger_summary['complex_output_count']}`")
    lines.append("")

    lines.append("## Triggered Modules")
    if trigger_summary["triggered_module_names"]:
        for name in trigger_summary["triggered_module_names"]:
            lines.append(f"- `{name}`")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("## Untriggered Modules")
    if trigger_summary["untriggered_module_names"]:
        for name in trigger_summary["untriggered_module_names"]:
            lines.append(f"- `{name}`")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("## Complex Output Modules")
    if trigger_summary["complex_output_module_names"]:
        for name in trigger_summary["complex_output_module_names"]:
            lines.append(f"- `{name}`")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("## Hook Records")
    for rec in report["hook_records"]:
        lines.append(
            "- module: `{}` | type: `{}` | call_count: `{}` | output_type: `{}` | complex_output: `{}`".format(
                rec["module_name"],
                rec["module_type"],
                rec["call_count"],
                rec["output_type"],
                rec["complex_output"],
            )
        )
        lines.append(f"  - output_summaries: `{json.dumps(rec['output_summaries'], ensure_ascii=False)}`")
    if not report["hook_records"]:
        lines.append("- (no hook record)")
    lines.append("")

    lines.append("## Limitations")
    for item in report["limitations"]:
        lines.append(f"- {item}")
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

    logging.info("Selecting modules by patterns: %s", args.module_name_patterns)
    candidates = pick_candidate_modules(model, args.module_name_patterns, args.max_hooks)
    logging.info("Registered module candidates: %d", len(candidates))
    if not candidates:
        logging.warning("No modules matched the given patterns.")

    trigger_store: dict[str, dict[str, Any]] = {}
    handles = []
    for cand in candidates:
        module = model.get_submodule(cand.module_name)
        handle = module.register_forward_hook(
            build_hook(cand.module_name, cand.module_type, trigger_store)
        )
        handles.append(handle)

    try:
        logging.info("Running one forward pass...")
        run_single_forward(model, tokenizer, args.prompt, args.device)
    finally:
        for h in handles:
            h.remove()

    registered_names = [c.module_name for c in candidates]
    triggered_names = sorted(trigger_store.keys())
    untriggered_names = sorted(set(registered_names) - set(triggered_names))
    complex_names = sorted(
        name for name, rec in trigger_store.items() if bool(rec.get("complex_output", False))
    )

    logging.info("Triggered hooks: %d", len(triggered_names))
    for name in triggered_names:
        logging.info("TRIGGERED: %s", name)

    logging.info("Registered but not triggered hooks: %d", len(untriggered_names))
    for name in untriggered_names:
        logging.warning("UNTRIGGERED: %s", name)

    logging.info("Complex/unstructured outputs: %d", len(complex_names))
    for name in complex_names:
        logging.warning("COMPLEX_OUTPUT: %s", name)

    report = build_report(args, model, tokenizer, candidates, trigger_store)
    save_json(report, DEFAULT_OUTPUT_JSON)
    save_markdown(to_markdown(report), DEFAULT_OUTPUT_MD)
    logging.info("Saved JSON: %s", DEFAULT_OUTPUT_JSON)
    logging.info("Saved Markdown: %s", DEFAULT_OUTPUT_MD)


if __name__ == "__main__":
    main()
