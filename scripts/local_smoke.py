"""Run a small real-model smoke test on CPU, Apple MPS, or CUDA."""

import argparse
import json
from pathlib import Path
import tempfile

from train import load, rollout
from utils.train.rewards import correctness


ROW = {
    "prompt": [{"role": "user", "content":
                "What is 7 + 5? Put the final answer on the last line, like: Final answer: 12"}],
    "answer": "12",
    "task": "math",
    "source": "local_smoke",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"),
                        default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    # Exercise the JSONL format before spending time loading model weights.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "smoke.jsonl"
        path.write_text(json.dumps(ROW) + "\n")
        assert json.loads(path.read_text().splitlines()[0]) == ROW

    model, tokenizer = load(args.model, args.device, args.dtype)
    reply, tokens = rollout(model, tokenizer, ROW, args.max_new_tokens)
    reward = correctness([reply], [ROW["answer"]], [ROW["task"]])[0]
    print("\n--- reply ---")
    print(reply)
    print("--- smoke summary ---")
    print(json.dumps({"generated_tokens": tokens, "reward": reward}, indent=2))
    if tokens == 0:
        raise SystemExit("Smoke test generated no tokens")


if __name__ == "__main__":
    main()
