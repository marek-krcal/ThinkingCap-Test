"""A starting point: loads Qwen3-0.6B and the prepared data, generates a reply and scores it.

This only shows how to use our helpers (utils/train/prepare_data.py, utils/train/rewards.py); you don't have
to stick to them. Build your own method from here. Good luck!
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.train.rewards import correctness

TRAIN_DATA = Path("data/train.jsonl")


def resolve_runtime(device="auto", dtype="auto"):
    """Choose a portable device/dtype pair for Apple Silicon and cloud GPUs."""
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if dtype == "auto":
        if device == "cuda":
            # T4/P100 cannot execute bf16; Ampere and newer can.
            major, _ = torch.cuda.get_device_capability()
            dtype = "bfloat16" if major >= 8 else "float16"
        else:
            # FP32 is the most dependable correctness smoke test on an M1.
            dtype = "float32"
    return device, getattr(torch, dtype)


def load(model_id, device="auto", dtype="auto"):
    """Return a model/tokenizer using a runtime supported by the current host."""
    device, dtype = resolve_runtime(device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    print(f"Loaded {model_id} on {device} with {dtype}", flush=True)
    return model, tokenizer


def rollout(model, tokenizer, row, max_new_tokens=2048):
    """Generate one reply to a prepared row; return (reply text, number of generated tokens)."""
    inputs = tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True).to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                            temperature=0.6, top_p=0.95, top_k=20)
    generated = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True), len(generated)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--data", default=str(TRAIN_DATA))
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"),
                        default="auto")
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    train_data = Path(arguments.data)
    if not train_data.exists():
        raise SystemExit(f"{train_data} does not exist yet. Build the suggested datasets with:\n"
                         "    python utils/train/prepare_data.py")
    model, tokenizer = load(arguments.model, arguments.device, arguments.dtype)
    data = load_dataset("json", data_files=str(train_data), split="train")

    row = data[arguments.row]
    reply, length = rollout(model, tokenizer, row, arguments.max_new_tokens)
    reward = correctness([reply], [row["answer"]], [row["task"]])[0]
    print(f"{row['source']}: reward {reward}, {length} tokens")


if __name__ == "__main__":
    main()
