from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer


KEYWORDS = (
    "attn",
    "attention",
    "self_attn",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "rotary",
    "rope",
)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect model modules and collect attention/rope candidate hook points."
    )
    parser.add_argument("--model_dir", type=Path, required=True, help="Local model directory.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device string for model loading, e.g. cpu / cuda / cuda:0.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to AutoModel/AutoTokenizer loading.",
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=Path("outputs/reports/inspect_attention_modules.md"),
        help="Markdown report path.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("outputs/reports/inspect_attention_modules.json"),
        help="JSON report path.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def maybe_extract_parent_layer_index(module_name: str) -> int | None:
    """
    Conservatively extract layer index from common textual patterns.
    If no reliable pattern is found, return None.
    """
    patterns = (
        r"(?:^|\.)layers?\.(\d+)(?:\.|$)",
        r"(?:^|\.)blocks?\.(\d+)(?:\.|$)",
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


def module_hit(module_name: str, module_type: str) -> bool:
    low_name = module_name.lower()
    low_type = module_type.lower()
    return any(k in low_name or k in low_type for k in KEYWORDS)


def collect_candidate_modules(model: Any) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not name:
            continue
        module_type = module.__class__.__name__
        if module_hit(name, module_type):
            hits.append(
                {
                    "module_name": name,
                    "module_type": module_type,
                    "parent_layer_index": maybe_extract_parent_layer_index(name),
                }
            )
    return hits


def collect_backbone_clues(model: Any) -> dict[str, Any]:
    top_level_names = [name for name, _ in model.named_children()]
    model_attr = getattr(model, "model", None)
    clues = {
        "top_level_child_modules": top_level_names,
        "has_attr_layers": hasattr(model, "layers"),
        "has_attr_model": model_attr is not None,
        "has_attr_model_layers": hasattr(model_attr, "layers") if model_attr is not None else False,
        "top_level_layer_like_names": [
            name
            for name in top_level_names
            if ("layer" in name.lower() or "block" in name.lower() or name.lower() == "h")
        ],
    }
    clues["has_layers_or_model_layers_clue"] = bool(
        clues["has_attr_layers"] or clues["has_attr_model_layers"] or clues["top_level_layer_like_names"]
    )
    return clues


def build_candidate_summary(hits: list[dict[str, Any]]) -> dict[str, list[str]]:
    attn_like: list[str] = []
    proj_like: list[str] = []
    rotary_like: list[str] = []

    for item in hits:
        name = item["module_name"]
        low = name.lower()
        if "attn" in low or "attention" in low:
            attn_like.append(name)
        if any(k in low for k in ("q_proj", "k_proj", "v_proj", "o_proj")):
            proj_like.append(name)
        if "rotary" in low or "rope" in low:
            rotary_like.append(name)

    def unique_sorted(values: list[str]) -> list[str]:
        return sorted(set(values))

    return {
        "attention_main_candidates": unique_sorted(attn_like),
        "qkv_projection_candidates": unique_sorted(proj_like),
        "rotary_position_candidates": unique_sorted(rotary_like),
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Attention Module Inspection Report")
    lines.append("")
    lines.append("## Run Info")
    lines.append(f"- model_dir: `{report['run_info']['model_dir']}`")
    lines.append(f"- device: `{report['run_info']['device']}`")
    lines.append(f"- trust_remote_code: `{report['run_info']['trust_remote_code']}`")
    lines.append("")
    lines.append("## Model Overview")
    lines.append(f"- model_class_name: `{report['model_overview']['model_class_name']}`")
    lines.append(f"- tokenizer_class_name: `{report['model_overview']['tokenizer_class_name']}`")
    lines.append(f"- total_named_modules: `{report['model_overview']['total_named_modules']}`")
    lines.append("")
    lines.append("## Backbone Clues")
    backbone = report["backbone_clues"]
    lines.append(f"- has_attr_layers: `{backbone['has_attr_layers']}`")
    lines.append(f"- has_attr_model: `{backbone['has_attr_model']}`")
    lines.append(f"- has_attr_model_layers: `{backbone['has_attr_model_layers']}`")
    lines.append(f"- has_layers_or_model_layers_clue: `{backbone['has_layers_or_model_layers_clue']}`")
    lines.append("- top_level_child_modules:")
    for name in backbone["top_level_child_modules"]:
        lines.append(f"  - `{name}`")
    lines.append("")
    lines.append("## Keyword Hits")
    lines.append(f"- hit_count: `{len(report['keyword_hits'])}`")
    lines.append("")
    for item in report["keyword_hits"]:
        lines.append(
            "- module_name: `{}` | module_type: `{}` | parent_layer_index: `{}`".format(
                item["module_name"], item["module_type"], item["parent_layer_index"]
            )
        )
    lines.append("")
    lines.append("## 小结")
    lines.append("以下均为候选判断，不是确定结论。")
    summary = report["candidate_summary"]
    lines.append("- 候选 attention 主模块:")
    if summary["attention_main_candidates"]:
        for name in summary["attention_main_candidates"]:
            lines.append(f"  - `{name}`")
    else:
        lines.append("  - （未发现明显候选）")
    lines.append("- 候选 q/k/v 投影模块:")
    if summary["qkv_projection_candidates"]:
        for name in summary["qkv_projection_candidates"]:
            lines.append(f"  - `{name}`")
    else:
        lines.append("  - （未发现明显候选）")
    lines.append("- 候选 rotary 相关位置:")
    if summary["rotary_position_candidates"]:
        for name in summary["rotary_position_candidates"]:
            lines.append(f"  - `{name}`")
    else:
        lines.append("  - （未发现明显候选）")
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

    logging.info("Enumerating modules...")
    total_named_modules = sum(1 for _ in model.named_modules())
    keyword_hits = collect_candidate_modules(model)
    backbone_clues = collect_backbone_clues(model)
    candidate_summary = build_candidate_summary(keyword_hits)

    report: dict[str, Any] = {
        "run_info": {
            "model_dir": str(args.model_dir),
            "device": args.device,
            "trust_remote_code": args.trust_remote_code,
        },
        "model_overview": {
            "model_class_name": model.__class__.__name__,
            "tokenizer_class_name": tokenizer.__class__.__name__,
            "total_named_modules": total_named_modules,
        },
        "backbone_clues": backbone_clues,
        "keyword_hits": keyword_hits,
        "candidate_summary": candidate_summary,
    }

    save_json(report, args.output_json)
    save_markdown(to_markdown(report), args.output_md)
    logging.info("Saved JSON: %s", args.output_json)
    logging.info("Saved Markdown: %s", args.output_md)


if __name__ == "__main__":
    main()
