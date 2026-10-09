# Local, Kaggle, and Colab workflow

The Git repository is the source of truth. Develop with Codex locally, make a
small M1 smoke run, commit the exact code, and let Kaggle or Colab clone that
commit. Do not put Kaggle, GitHub, or Hugging Face tokens in this repository.

## Local Apple-Silicon smoke test

```bash
python3 -m venv .venv-local
source .venv-local/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-local.txt
python -m unittest discover -s tests -t .
python -m scripts.local_smoke --device mps --dtype float32 --max-new-tokens 64
```

The first real-model run downloads Qwen3-0.6B. It checks loading, the chat
template, MPS generation, token counting, and reward parsing. It is not a
performance or accuracy measurement.

For a tiny, safe local evaluation sample using the actual benchmark graders:

```bash
python evaluate.py \
  --out runs/m1-tiny-safe-fp16-512 \
  --smoke --samples 1 --limit-per-dataset 1 --max-tokens 512 \
  --backend transformers --device mps --dtype half \
  --benchmarks bfcl,svamp,ifeval
```

The Transformers backend is intended for local development and timing, not the
official final score: its sampler is not bit-identical to vLLM. MBPP is omitted
because its grader executes model-written Python; run that benchmark in Kaggle's
Linux environment instead of directly on the development Mac.

## Kaggle

1. Push this repository to your own GitHub fork or branch.
2. Edit `cloud/kaggle/run_kaggle.py` defaults to that public repository and
   branch, or provide `THINKINGCAP_REPO` and `THINKINGCAP_REF` in the runtime.
3. Replace `YOUR_KAGGLE_USERNAME` in `cloud/kaggle/kernel-metadata.json`.
4. Authenticate the Kaggle CLI locally, then submit:

```bash
kaggle kernels push -p cloud/kaggle
```

The default job is a one-sample smoke evaluation over the frozen 30-question
subset. After it succeeds, change `THINKINGCAP_MODE` to `full` for the proper
eight-sample baseline. Outputs are written under `/kaggle/working/runs/` so
Kaggle captures them as notebook output.

Kaggle T4 and P100 GPUs use fp16 automatically. Ampere and newer GPUs use bf16.
Always evaluate the base and candidate with the same dtype.

The dedicated `cloud/kaggle-vllm` runner evaluates 100 questions per benchmark
(400 generations, one each) using one vLLM engine with tensor parallel size 2
on Kaggle's two T4s. It queues all 400 jobs together and records both GPU
utilization and memory every 15 seconds. Tensor parallelism can be slower than
one GPU for a small model; compare generation-only time and output-token counts.
Parallel size and queue batch size are recorded with scores because outputs can
change with these settings. Disposable environments, source and model downloads
live under `/kaggle/temp`, not the archived `/kaggle/working` output tree.
The memory-safe scheduler activates at most 32 sequences and 2048 prefill tokens,
with 70% GPU memory allocation. Queuing 400 requests does not mean running 400
simultaneously. After the 400-question run succeeds, this time-bounded campaign
runs the full 2759-question cohort with four generations each (11036 answers).
No new evaluation starts after 2026-10-10 02:00 Prague time; the absolute stop
is 04:00 Prague, and monitoring decides whether a run merits the extra window.

## Colab

Open `cloud/colab/ThinkingCap_Test.ipynb`, change `REPO_URL` and `REPO_REF` to
your fork and exact branch/commit, select a GPU runtime, and run top to bottom.
The notebook defaults to a smoke evaluation. Set `FULL_EVALUATION = True` only
after the smoke run succeeds. Download `runs/` or copy it to Drive before the
runtime is released.

## Experiment discipline

- Never train on SVAMP, BFCL, MBPP+, or IFEval.
- Preserve one baseline run for every precision/evaluation configuration.
- Save the exact Git commit and resolved training configuration with checkpoints.
- For T4/P100 training use fp16, LoRA, micro-batch 1, gradient accumulation,
  gradient checkpointing, and chunked/fused cross-entropy.
- The supplied `train.py` is a starting example, not a finished training method.
# Offline two-GPU data parallelism

For a model that fits on one GPU, the shared evaluator can use one controller
with two independent vLLM worker engines:

```sh
python cloud/run_eval.py --out runs/dp-400 --backend vllm --dtype half \
  --limit-per-benchmark 100 --samples 1 --data-parallel-size 2 --batch-size 128
```

This is an offline multiprocessing controller, not the vLLM API-server DP mode.
It launches the same worker function on each GPU, using fresh spawned processes
and GPU visibility isolation. Jobs retain their original IDs, sample indices,
and seeds; shard records are merged into one `records.jsonl` and graded once.
Each worker keeps batch-level records under `shards/` for failure diagnostics.
Tensor parallelism must remain 1 in this mode. The full run uses `--full
--samples 4` instead of `--limit-per-benchmark 100 --samples 1`.
