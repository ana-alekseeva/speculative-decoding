#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

# `oneshot` import path changed across llmcompressor versions; support both.
try:
    from llmcompressor import oneshot
except ImportError:  # pragma: no cover - older layout
    from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
        help="Source model id or path (default: Qwen/Qwen3-8B).",
    )
    parser.add_argument(
        "--save-dir",
        default="Qwen3-8B-FP8-Dynamic",
        help="Output directory for the quantized model "
        "(default: ./Qwen3-8B-FP8-Dynamic).",
    )
    parser.add_argument(
        "--ignore",
        nargs="+",
        default=["lm_head"],
        help="Modules to leave unquantized (default: lm_head).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty --save-dir (default: refuse).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    save_dir = Path(args.save_dir)
    # Guard: never clobber an existing (non-empty) directory unless asked.
    if save_dir.exists() and any(save_dir.iterdir()) and not args.overwrite:
        sys.exit(
            f"ERROR: '{save_dir}' already exists and is not empty. "
            "Refusing to overwrite. Pass --overwrite to override, or pick "
            "another --save-dir."
        )
    # Guard: refuse to write into the source model directory itself.
    src = Path(args.model)
    if src.exists() and save_dir.resolve() == src.resolve():
        sys.exit("ERROR: --save-dir must differ from the source --model path.")

    print(f"==> Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # FP8 dynamic: weights FP8, activations dynamic FP8, Linear layers only,
    # lm_head left untouched. Dynamic activation quant needs no calibration set.
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=args.ignore,
    )

    print("==> Applying FP8_DYNAMIC quantization to Linear layers "
          f"(ignored: {args.ignore})")
    oneshot(model=model, recipe=recipe)

    print(f"==> Saving compressed model to: {save_dir}")
    model.save_pretrained(str(save_dir), save_compressed=True)
    tokenizer.save_pretrained(str(save_dir))

    # Verify the saved config actually carries a quantization section.
    config_path = save_dir / "config.json"
    with open(config_path) as fh:
        cfg = json.load(fh)
    qcfg = cfg.get("quantization_config")
    if not qcfg:
        sys.exit(
            f"ERROR: no 'quantization_config' found in {config_path}. "
            "Quantization did not persist as expected."
        )

    print("\n==> Saved. Quantization config summary:")
    print(f"    quant_method : {qcfg.get('quant_method')}")
    print(f"    format       : {qcfg.get('format')}")
    groups = qcfg.get("config_groups", {})
    for name, group in groups.items():
        w = group.get("weights", {})
        a = group.get("input_activations", {})
        print(
            f"    {name}: targets={group.get('targets')} "
            f"weights={w.get('type')}{w.get('num_bits')} "
            f"activations={a.get('type')}{a.get('num_bits')} "
            f"dynamic={a.get('dynamic')}"
        )
    print(f"    ignore       : {qcfg.get('ignore')}")
    print(f"\nDone. Quantized model directory: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
