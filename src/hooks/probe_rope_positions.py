from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_OUTPUT_JSON = Path("outputs/reports/rope_position_probe.json")
DEFAULT_OUTPUT_MD = Path("outputs/reports/rope_position_probe.md")
RELATED_KEYWORDS = ("rope", "rotary", "rotary_emb", "apply_rotary",)


@dataclass
class ModuleMeta:
    name: str
    module_type: str
    is_rope_like: bool
    is_q_proj: bool
    is_k_proj: bool
    attn_scope: str | None


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe probable RoPE position in Qwen-style models via module discovery and "
            "single-forward hook structure inspection."
        )
    )
    parser.add_argument("--model_dir", type=Path, required=True, help="Local model directory.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt used for one forward pass.")
    parser.add_argument("--device", type=str, default="cpu", help="Device, e.g. cpu / cuda / cuda:0.")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True when loading tokenizer/model.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_module_name(name: str) -> str:
    return name.lower()


def infer_attn_scope(name: str) -> str | None:
    if ".self_attn." in name:
        return name.split(".self_attn.", 1)[0] + ".self_attn"
    if ".attn." in name:
        return name.split(".attn.", 1)[0] + ".attn"
    if name.endswith(".self_attn"):
        return name
    if name.endswith(".attn"):
        return name
    return None


def is_related_name(name: str) -> bool:
    low = normalize_module_name(name)
    return any(k in low for k in RELATED_KEYWORDS)


def collect_related_modules(model: Any) -> list[ModuleMeta]:
    items: list[ModuleMeta] = []
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        if not is_related_name(module_name):
            continue
        low = normalize_module_name(module_name)
        items.append(
            ModuleMeta(
                name=module_name,
                module_type=module.__class__.__name__,
                is_rope_like=("rope" in low or "rotary" in low),
                is_q_proj=low.endswith(".q_proj"),
                is_k_proj=low.endswith(".k_proj"),
                attn_scope=infer_attn_scope(module_name),
            )
        )
    return items


def summarize_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "kind": "tensor",
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def summarize_structure(
    obj: Any,
    depth: int = 0,
    max_depth: int = 2,
    max_items: int = 8,
) -> dict[str, Any]:
    if isinstance(obj, torch.Tensor):
        return summarize_tensor(obj)
    if depth >= max_depth:
        return {"kind": "max_depth_reached", "type": type(obj).__name__}
    if isinstance(obj, (tuple, list)):
        items: list[dict[str, Any]] = []
        limited = obj[:max_items]
        for idx, item in enumerate(limited):
            items.append(
                {
                    "index": idx,
                    "summary": summarize_structure(
                        item,
                        depth=depth + 1,
                        max_depth=max_depth,
                        max_items=max_items,
                    ),
                }
            )
        return {
            "kind": type(obj).__name__,
            "len": len(obj),
            "items": items,
            "truncated": len(obj) > max_items,
        }
    if isinstance(obj, dict):
        keys = list(obj.keys())
        limited_keys = keys[:max_items]
        items: list[dict[str, Any]] = []
        for key in limited_keys:
            items.append(
                {
                    "key": str(key),
                    "summary": summarize_structure(
                        obj[key],
                        depth=depth + 1,
                        max_depth=max_depth,
                        max_items=max_items,
                    ),
                }
            )
        return {
            "kind": "dict",
            "len": len(keys),
            "items": items,
            "truncated": len(keys) > max_items,
        }
    if obj is None:
        return {"kind": "none"}
    return {"kind": "other", "type": type(obj).__name__}


def extract_tensor_shapes(summary: dict[str, Any]) -> list[list[int]]:
    shapes: list[list[int]] = []
    if summary.get("kind") == "tensor":
        shape = summary.get("shape")
        if isinstance(shape, list):
            shapes.append([int(x) for x in shape])
        return shapes

    if summary.get("kind") in {"tuple", "list"}:
        for item in summary.get("items", []):
            child = item.get("summary", {})
            if isinstance(child, dict):
                shapes.extend(extract_tensor_shapes(child))
        return shapes

    if summary.get("kind") == "dict":
        for item in summary.get("items", []):
            child = item.get("summary", {})
            if isinstance(child, dict):
                shapes.extend(extract_tensor_shapes(child))
        return shapes

    return shapes


def build_hook(
    module_name: str,
    module_type: str,
    call_counter: dict[str, int],
    hook_records: dict[str, dict[str, Any]],
) -> Any:
    def _hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        call_counter["value"] += 1
        call_idx = call_counter["value"]
        summary = summarize_structure(output)
        shapes = extract_tensor_shapes(summary)
        record = hook_records.setdefault(
            module_name,
            {
                "module_name": module_name,
                "module_type": module_type,
                "call_count": 0,
                "first_call_order": None,
                "last_call_order": None,
                "output_type": type(output).__name__,
                "output_summaries": [],
                "tensor_shapes_observed": [],
            },
        )
        record["call_count"] += 1
        if record["first_call_order"] is None:
            record["first_call_order"] = call_idx
        record["last_call_order"] = call_idx
        if len(record["output_summaries"]) < 2:
            record["output_summaries"].append(summary)
        if shapes:
            known = {json.dumps(s) for s in record["tensor_shapes_observed"]}
            for shp in shapes:
                dumped = json.dumps(shp)
                if dumped not in known:
                    record["tensor_shapes_observed"].append(shp)
                    known.add(dumped)

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


def estimate_confidence(
    rope_module_count: int,
    rope_triggered_count: int,
    near_qk_candidate_count: int,
) -> str:
    if rope_triggered_count > 0 and near_qk_candidate_count > 0:
        return "high confidence"
    if rope_module_count > 0 and rope_triggered_count > 0:
        return "medium confidence"
    return "low confidence"


def analyze_qk_and_rope_relationship(
    related_modules: list[ModuleMeta],
    hook_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rope_meta = [m for m in related_modules if m.is_rope_like]
    q_meta = [m for m in related_modules if m.is_q_proj]
    k_meta = [m for m in related_modules if m.is_k_proj]
    rope_names = {m.name for m in rope_meta}
    q_names = {m.name for m in q_meta}
    k_names = {m.name for m in k_meta}
    triggered_names = set(hook_records.keys())

    qk_scopes = sorted(
        {
            m.attn_scope
            for m in (q_meta + k_meta)
            if m.attn_scope is not None
        }
    )

    per_scope: list[dict[str, Any]] = []
    near_qk_candidates: list[dict[str, Any]] = []
    for scope in qk_scopes:
        q_name = scope + ".q_proj"
        k_name = scope + ".k_proj"
        q_rec = hook_records.get(q_name)
        k_rec = hook_records.get(k_name)
        q_first = q_rec.get("first_call_order") if q_rec else None
        k_first = k_rec.get("first_call_order") if k_rec else None

        after_threshold: int | None = None
        if isinstance(q_first, int) and isinstance(k_first, int):
            after_threshold = max(q_first, k_first)
        elif isinstance(q_first, int):
            after_threshold = q_first
        elif isinstance(k_first, int):
            after_threshold = k_first

        scope_rope = sorted(m.name for m in rope_meta if m.attn_scope == scope)
        scope_rope_triggered = [
            n for n in scope_rope if n in hook_records and hook_records[n].get("call_count", 0) > 0
        ]

        after_qk_modules: list[dict[str, Any]] = []
        if after_threshold is not None:
            for module_name, rec in hook_records.items():
                if not module_name.startswith(scope + "."):
                    continue
                first_call = rec.get("first_call_order")
                if isinstance(first_call, int) and first_call > after_threshold:
                    after_qk_modules.append(
                        {
                            "module_name": module_name,
                            "module_type": rec.get("module_type"),
                            "first_call_order": first_call,
                            "tensor_shapes_observed": rec.get("tensor_shapes_observed", []),
                            "is_rope_like_name": module_name in rope_names,
                        }
                    )

        after_qk_modules = sorted(after_qk_modules, key=lambda x: x["first_call_order"])
        for item in after_qk_modules:
            if item["is_rope_like_name"]:
                near_qk_candidates.append(item)

        per_scope.append(
            {
                "attn_scope": scope,
                "q_proj_name": q_name,
                "k_proj_name": k_name,
                "q_proj_triggered": q_name in triggered_names,
                "k_proj_triggered": k_name in triggered_names,
                "q_proj_first_call_order": q_first,
                "k_proj_first_call_order": k_first,
                "rope_modules_in_scope": scope_rope,
                "rope_modules_triggered_in_scope": scope_rope_triggered,
                "modules_after_qk_in_scope": after_qk_modules[:12],
            }
        )

    confidence = estimate_confidence(
        rope_module_count=len(rope_meta),
        rope_triggered_count=len([n for n in rope_names if n in triggered_names]),
        near_qk_candidate_count=len(near_qk_candidates),
    )

    if len(rope_meta) == 0:
        message = (
            "No explicit rope/rotary-like module name was found. "
            "RoPE may be inlined in attention forward and is not directly hookable by module name."
        )
    elif len(near_qk_candidates) > 0:
        message = (
            "At least one rope/rotary-like module is observed after q_proj/k_proj in call order "
            "within attention scope; these are plausible non-intrusive hook candidates."
        )
    elif len([n for n in rope_names if n in triggered_names]) > 0:
        message = (
            "Rope/rotary-like module names exist and were triggered, but not clearly positioned "
            "after q_proj/k_proj for this probe input."
        )
    else:
        message = (
            "Rope/rotary-like module names exist but were not triggered in this run. "
            "Position cannot be confirmed from current probe."
        )

    return {
        "q_proj_module_count": len(q_names),
        "k_proj_module_count": len(k_names),
        "rope_like_module_count": len(rope_names),
        "rope_like_triggered_count": len([n for n in rope_names if n in triggered_names]),
        "per_attention_scope": per_scope,
        "near_qk_rope_candidates": near_qk_candidates[:24],
        "conclusion_confidence": confidence,
        "conclusion": message,
    }


def build_report(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    related_modules: list[ModuleMeta],
    hook_records: dict[str, dict[str, Any]],
    forward_info: dict[str, Any],
) -> dict[str, Any]:
    rope_related = analyze_qk_and_rope_relationship(related_modules, hook_records)
    rope_names = sorted(m.name for m in related_modules if m.is_rope_like)
    triggered_names = sorted(hook_records.keys())
    triggered_rope_names = sorted(name for name in rope_names if name in hook_records)

    return {
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
        "module_discovery": {
            "related_keyword_list": list(RELATED_KEYWORDS),
            "related_module_count": len(related_modules),
            "related_modules": [
                {
                    "name": m.name,
                    "module_type": m.module_type,
                    "is_rope_like": m.is_rope_like,
                    "is_q_proj": m.is_q_proj,
                    "is_k_proj": m.is_k_proj,
                    "attn_scope": m.attn_scope,
                }
                for m in related_modules
            ],
            "rope_like_module_names": rope_names,
            "q_proj_module_names": sorted(m.name for m in related_modules if m.is_q_proj),
            "k_proj_module_names": sorted(m.name for m in related_modules if m.is_k_proj),
        },
        "hook_run": {
            "registered_hook_count": len(related_modules),
            "triggered_hook_count": len(triggered_names),
            "triggered_module_names": triggered_names,
            "triggered_rope_like_module_names": triggered_rope_names,
            "forward_input": forward_info,
            "hook_records": [hook_records[n] for n in triggered_names],
        },
        "rope_position_analysis": rope_related,
        "conclusions": [
            {
                "question": "Is there a nominal rotary_emb/rope module?",
                "answer": (
                    f"Found {len(rope_names)} rope/rotary-like modules by name."
                    if rope_names
                    else "No explicit rotary_emb/rope-like module name found."
                ),
                "confidence": "high confidence" if rope_names else "low confidence",
            },
            {
                "question": (
                    "After q_proj/k_proj outputs, can we still observe rope-related module "
                    "or intermediate tensor candidate?"
                ),
                "answer": rope_related["conclusion"],
                "confidence": rope_related["conclusion_confidence"],
            },
            {
                "question": (
                    "Do we have non-intrusive hook positions that may approximate "
                    "q_after_rope / k_after_rope?"
                ),
                "answer": (
                    "Potential candidates exist in near_qk_rope_candidates."
                    if rope_related["near_qk_rope_candidates"]
                    else (
                        "Cannot confirm reliable non-intrusive hook position from this run. "
                        "No stable q_after_rope / k_after_rope candidate is proven."
                    )
                ),
                "confidence": (
                    "medium confidence"
                    if rope_related["near_qk_rope_candidates"]
                    else "low confidence"
                ),
            },
        ],
        "limitations": [
            "This script is position probing only; it does NOT directly capture q_after_rope / k_after_rope.",
            "Forward-hook call order provides heuristic evidence, not formal execution graph proof.",
            "If RoPE is fused/inlined inside attention forward, named module probing may miss exact tensor boundaries.",
            "If rope-related modules are not triggered for the given prompt length/settings, position cannot be confirmed.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    run_info = report["run_info"]
    overview = report["model_overview"]
    discovery = report["module_discovery"]
    hook_run = report["hook_run"]
    analysis = report["rope_position_analysis"]
    conclusions = report["conclusions"]

    lines: list[str] = []
    lines.append("# RoPE Position Probe Report")
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
    lines.append("## Module Discovery")
    lines.append(f"- related module count: `{discovery['related_module_count']}`")
    lines.append(f"- rope/rotary-like module count: `{len(discovery['rope_like_module_names'])}`")
    lines.append(f"- q_proj module count: `{len(discovery['q_proj_module_names'])}`")
    lines.append(f"- k_proj module count: `{len(discovery['k_proj_module_names'])}`")
    lines.append("")
    lines.append("## Hook Summary")
    lines.append(f"- registered hook count: `{hook_run['registered_hook_count']}`")
    lines.append(f"- triggered hook count: `{hook_run['triggered_hook_count']}`")
    lines.append(
        f"- triggered rope-like module count: `{len(hook_run['triggered_rope_like_module_names'])}`"
    )
    lines.append("")
    lines.append("## Key Conclusions")
    for item in conclusions:
        lines.append(f"- {item['question']}")
        lines.append(f"  - answer: {item['answer']}")
        lines.append(f"  - confidence: **{item['confidence']}**")
    lines.append("")
    lines.append("## Scope-Level Evidence (first 10 scopes)")
    for scope_item in analysis["per_attention_scope"][:10]:
        lines.append(f"- scope: `{scope_item['attn_scope']}`")
        lines.append(
            f"  - q_proj first call: `{scope_item['q_proj_first_call_order']}` | "
            f"k_proj first call: `{scope_item['k_proj_first_call_order']}`"
        )
        lines.append(
            f"  - rope modules in scope: `{len(scope_item['rope_modules_in_scope'])}` | "
            f"triggered: `{len(scope_item['rope_modules_triggered_in_scope'])}`"
        )
        lines.append(
            f"  - modules_after_qk_in_scope (sample): "
            f"`{[x['module_name'] for x in scope_item['modules_after_qk_in_scope'][:4]]}`"
        )
    lines.append("")
    lines.append("## Non-Intrusive Candidate Note")
    lines.append(
        "- `near_qk_rope_candidates` in JSON contains modules observed after q_proj/k_proj in "
        "call order and named like rope/rotary."
    )
    lines.append(
        "- These are heuristic candidates only and do NOT prove reliable q_after_rope / k_after_rope capture."
    )
    lines.append("")
    lines.append("## Limitations")
    for text in report["limitations"]:
        lines.append(f"- {text}")

    return "\n".join(lines) + "\n"


def save_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


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

    logging.info("Collecting related modules by keywords: %s", RELATED_KEYWORDS)
    related_modules = collect_related_modules(model)
    logging.info("Related modules discovered: %d", len(related_modules))

    call_counter: dict[str, int] = {"value": 0}
    hook_records: dict[str, dict[str, Any]] = {}
    handles: list[Any] = []
    try:
        for meta in related_modules:
            module = model.get_submodule(meta.name)
            handle = module.register_forward_hook(
                build_hook(
                    module_name=meta.name,
                    module_type=meta.module_type,
                    call_counter=call_counter,
                    hook_records=hook_records,
                )
            )
            handles.append(handle)

        logging.info("Running one forward pass for hook probing...")
        forward_info = run_single_forward(
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
        related_modules=related_modules,
        hook_records=hook_records,
        forward_info=forward_info,
    )
    md_text = render_markdown(report)

    save_json(DEFAULT_OUTPUT_JSON, report)
    save_text(DEFAULT_OUTPUT_MD, md_text)

    logging.info("Saved JSON report to %s", DEFAULT_OUTPUT_JSON)
    logging.info("Saved Markdown report to %s", DEFAULT_OUTPUT_MD)


if __name__ == "__main__":
    main()
