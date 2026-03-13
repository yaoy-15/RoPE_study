from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SUPPORTED_TENSORS = {"q", "k", "v"}
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
        description="Capture q/k/v projection outputs by forward hook for Qwen attention modules."
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
        help="Tensor to capture: q / k / v.",
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
    suffix = TENSOR_TO_MODULE_SUFFIX[tensor_name]
    return f"model.layers.{layer_idx}.self_attn.{suffix}"


def run_single_forward(model: Any, tokenizer: Any, prompt: str, device: str) -> None:
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in encoded.items()}
    with torch.no_grad():
        _ = model(**encoded)


def save_metadata(path: Path, metadata: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    setup_logging()
    args = parse_args()

    tensor_name = args.tensor_name
    layer_idx = args.layer_idx
    source_module = build_source_module_name(layer_idx=layer_idx, tensor_name=tensor_name)

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

    logging.info("Resolving source module: %s", source_module)
    module = model.get_submodule(source_module)

    capture_store: dict[str, torch.Tensor] = {}

    def _hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                f"Expected Tensor output from {source_module}, but got {type(output).__name__}."
            )
        # IMPORTANT: We capture the linear projection output of q_proj/k_proj/v_proj.
        # This is saved as rope_stage='uncertain' because whether it strictly equals
        # RoPE-before tensors is not fully confirmed in a non-invasive way.
        capture_store["tensor"] = output.detach().cpu().clone()

    handle = module.register_forward_hook(_hook)
    try:
        logging.info("Running one forward pass for capture...")
        run_single_forward(model, tokenizer, args.prompt, args.device)
    finally:
        handle.remove()

    if "tensor" not in capture_store:
        raise RuntimeError(f"Hook did not capture any tensor from module: {source_module}")

    captured = capture_store["tensor"]
    model_name = args.model_dir.name
    safe_prompt_name = sanitize_prompt_name(args.prompt_name)
    out_dir = Path("outputs") / "tensors" / model_name / safe_prompt_name
    tensor_path = out_dir / f"layer_{layer_idx}_{tensor_name}.pt"
    metadata_path = out_dir / "metadata.json"

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
        "rope_stage": "uncertain",
        "source_module": source_module,
        "capture_method": "forward_hook",
        "tensor_file": str(tensor_path),
    }
    save_metadata(metadata_path, metadata)

    logging.info("Captured tensor shape: %s", list(captured.shape))
    logging.info("Captured tensor dtype: %s", captured.dtype)
    logging.info("Saved tensor: %s", tensor_path)
    logging.info("Saved metadata: %s", metadata_path)


if __name__ == "__main__":
    main()
