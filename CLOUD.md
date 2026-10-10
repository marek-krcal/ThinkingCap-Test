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
2. Use the tested CUDA vLLM runner (`cloud/kaggle-vllm`). The inefficient
   one-response-at-a-time Transformers Kaggle runner has been removed;
   Transformers remains available for local CPU/MPS smoke tests.
3. Edit the chosen runner's repository/ref defaults to your public fork and
   exact tested commit, or set `THINKINGCAP_REPO` and `THINKINGCAP_REF` in its runtime.
   While reviewing this PR, select `feature/kaggle-evaluation`, not the old `main`.
4. Replace `YOUR_KAGGLE_USERNAME` in the chosen `kernel-metadata.json`.
   Keep notebook privacy enabled; enable Internet and select GPUs in Kaggle.
5. Authenticate the Kaggle CLI locally, then submit:

```bash
kaggle kernels push -p cloud/kaggle-vllm
kaggle kernels status YOUR_KAGGLE_USERNAME/thinkingcap-vllm-evaluation
```

The default vLLM job is a one-sample smoke evaluation over the frozen 30-question
subset: it tests the graders but deliberately reports no benchmark scores.
Edit these defaults in `run.py` before uploading (CLI environment variables on
your laptop do **not** automatically become Kaggle environment variables):

| Setting | Default | Purpose |
| --- | --- | --- |
| `THINKINGCAP_MODE` | `smoke` | `full` selects all 2,759 questions |
| `THINKINGCAP_SAMPLES` | `1` | Allowed: `1`, `2`, `4`, `8` answers per question |
| `THINKINGCAP_LIMIT_PER_BENCHMARK` | `0` | Allowed: `100` (400 questions), `200` (800 questions), or `0` (mode selects smoke/full) |
| `THINKINGCAP_TIMEOUT_SECONDS` | `7200` | Total wall-clock allowance, including setup |

Do not combine full mode with a question limit. Hardware is deliberately fixed:
**FP16, TP1, DP2, continuous queue, 64 active answers per GPU**, 2048 prefill
tokens and 70% GPU memory target. Old parallelism/batch environment variables
are no longer read. Outputs go under
`/kaggle/working/runs/evaluation` for the vLLM runner.

Credentials stay outside Git: the Kaggle CLI supports the user's access token
file in `~/.kaggle/access_token`. Never print it, commit it, or put it in notebook
source. Submitting to Kaggle from a different machine/cloud environment requires
independently configuring its credentials.

The Kaggle entry points explicitly select FP16 (`half`). The shared evaluator's
`--dtype auto` selects FP16 for pre-Ampere CUDA GPUs and BF16 on newer CUDA GPUs.
Always evaluate the base and candidate with the same dtype and backend. Only T4
has been validated end-to-end here; these settings are not a blanket P100 guarantee.

The vLLM runner records both GPU utilization and memory every 15 seconds.
The fixed two-GPU recipe was measured on 400 questions: generation 331.94s,
1072.23 output tokens/s, versus 416.72 tokens/s for a single-GPU chunked queue.
This is one trial, not a guaranteed speedup; seeds do not guarantee identical
answers across scheduling configurations. Settings are recorded with scores.
Disposable environments, source and model downloads
live under `/kaggle/temp`, not the archived `/kaggle/working` output tree.
Concurrency means active sequences per GPU, not submitted queue size. The
controller queues all selected answers across its two workers without batch
barriers. Eager mode remains fixed; CUDA graphs have not been measured here.

### Environment, caching and outputs

The tested runner creates Python 3.12 in a disposable environment and installs
`vllm==0.11.2`, CUDA 12.8 PyTorch wheels, `nltk==3.9.2` and local requirements.
It does not upgrade Kaggle's host NVIDIA driver. A real-driver `libcuda.so`
link and compiler check resolve a missing linker name; CUDA stubs are rejected.
`VLLM_USE_FLASHINFER_SAMPLER=0` avoids a JIT sampler dependency on a full toolkit.
The exact checkout commit and installed packages are saved in notebook output.
Top-level packages are pinned, but this is not a complete dependency lock file.

Attach a previous successful notebook's output to reuse `eval-cache/questions`
and NLTK resources; the evaluator verifies the frozen cohort hash. This cache
includes grader checkouts required to grade the answers. Model weights and the
environment are **not** cached between Kaggle sessions by this setup. They stay
in `/kaggle/temp`, outside the archived output directory. Previously archiving
the environment/source caused unnecessary multi-GB output and delayed finalization.

Download only needed results, not the environment. Inspect paginated output
listings when files are missing:

```bash
kaggle kernels output YOUR_KAGGLE_USERNAME/thinkingcap-vllm-evaluation \
  -p runs/kaggle-download --page-size 200 \
  --file-pattern '(setup-timing\.json|run-config\.json|installed-packages\.txt|gpu-utilization\.jsonl|results\.json|records\.jsonl|manifest\.json)$'
```

Run artifacts include `setup-timing.json` (setup and evaluation wall times),
`gpu-utilization.jsonl` (busy-time and device memory, not tensor-core occupancy),
`run-config.json`, `installed-packages.txt`, graded answers and summary scores.
Engine startup and generation timing also appear in `results.json` and logs.

### Verified full-dataset example

The historical v7 run evaluated Qwen3-0.6B, pinned revision
`c1899de289a04d12100db370d81485cdf75e47ca`, on 2,759 questions with one answer
each. Configuration: FP16, TP=2, DP=1, queue=512, max active sequences=32,
prefill token budget=2048, memory fraction=.70, eager mode. It produced 2,067,746
tokens, with 66 truncated answers and no grader failures.

| Phase | Measured duration |
| --- | --- |
| Dependency installation | 75.53 s |
| Engine startup, cold model loading and health check | 157.38 s |
| Generation, excluding startup | 4,359.28 s (72m39s) |
| Evaluation, including startup and grading | 4,614.51 s (76m55s) |
| Total measured setup plus evaluation | 4,722.11 s (78m42s) |

Generation-only throughput was about 474 output tokens/s across both GPUs.
Corrected accuracies: MBPP 195/378 (51.59%), BFCL 677/840 (80.60%), SVAMP
871/1000 (87.10%), IFEval 300/541 (55.45%). The upstream README has no absolute
baseline scores; its comparison example is illustrative. On the matching
400-question development slice, accuracy differed by -1, 0, 0, +3 percentage
points, with mean answer lengths changing by less than 7%. This is a diagnostic
comparison, not a paired significance test: the earlier per-answer artifacts
were unavailable. Full x1 is not an official full x8 submission.

The IFEval grader now rejects truncated answers as documented upstream. Four
answers previously counted correct were corrected locally, lowering IFEval from
56.19% to 55.45%, without generating again. Preserve raw results, then correct:

```bash
python -m cloud.correct_truncation --source runs/raw --out runs/corrected
python cloud/verify_eval.py --out runs/corrected
```

The verifier targets this pinned full base-model cohort (defaults: 2,759 x1), not
arbitrary checkpoints. It checks answer completeness, seeds, token caps, cohort
hash, grades and summary consistency. Synthetic tests do not prove GPU performance.

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
python cloud/run_eval.py --out runs/dp-400 --limit-per-benchmark 100 --samples 1
python cloud/run_eval.py --out runs/dp-800 --limit-per-benchmark 200 --samples 2
python cloud/run_eval.py --out runs/dp-full --full --samples 4
```

This is an offline multiprocessing controller, not the vLLM API-server DP mode.
It launches the same worker function on each GPU, using fresh spawned processes
and GPU visibility isolation. Jobs retain their original IDs, sample indices,
and seeds; shard records are merged into one `records.jsonl` and graded once.
Each worker saves answers as they finish under `shards/` for failure diagnostics.
Full x1 uses `--full --samples 1`; official full x8 uses `--full --samples 8`.
Omitting all selection flags also means full coverage, with one answer by default;
use `--smoke` explicitly for the 30-question plumbing set. Model, adapter,
benchmark selection and output path remain configurable. Hardware flags are not
accepted by this convenience runner; `evaluate.py` retains low-level research
controls and local CPU/MPS support. Colab calls it directly for its single GPU.
DP2 continuous queue is validated on the 400-question cohort; the historical
full evaluation used TP2, so full DP2 coverage is not yet measured.

## Safety and limits

Never launch overlapping retries. Use notebook status and logs to confirm a
failed session has stopped before submitting again. Timeouts terminate the
subprocess group; a separate monitor is still required to cancel remote sessions
and respond to failures. No recurring monitor is installed by this repository.
Kaggle VM isolation is not a substitute for a code sandbox: MBPP executes
model-written Python. The measured run used resource limits only because bwrap
was unavailable. Prefer `--sandbox bwrap` where supported and avoid exposing
credentials or sensitive files to model-generated code.

Colab remains a convenience path, not an end-to-end validated GPU configuration.
The measured continuous-queue DP2 implementation is included in this PR;
the experimental campaign harness and CUDA-graph experiments remain separate.
