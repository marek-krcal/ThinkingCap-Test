# Evaluation throughput experiment

Measured 2026-10-10 on Kaggle kernel
`krcalkrcal/thinkingcap-evaluation-optimization`, version 1, runner commit
`0361dc6c9683a53e018b403692abb45ceb8d6c6c`. Changes are isolated on
`experiment/eval-throughput`; the basic setup PR is unchanged.

## Fixed evaluation recipe

- Qwen/Qwen3-0.6B, revision `c1899de289a04d12100db370d81485cdf75e47ca`.
- FP16, vLLM 0.11.2, PyTorch 2.9.0+cu128, two T4 GPUs allocated.
- Same frozen 400 questions: 100 each MBPP, BFCL, SVAMP, IFEval, one answer each.
- Temperature 0.6, top-p 0.95, top-k 20, unchanged per-question seeds.
- Output caps 8192 for MBPP and 4096 otherwise; capped answers score wrong.
- Eager execution, prefix caching disabled, prefill budget 2048 tokens,
  GPU memory target 0.70. No CUDA graph or quantization changes.

All five profiles passed an independent local `inspect_run` audit: 400 distinct
answers, identical cohort/model/revision/recipe, valid seeds and caps, completed
grading, no failed graders. The cohort SHA256 is
`16df64e67336b8a1cfa49298cd27e0e1f265927d12ad97f064f3968414d55a86`.

## Measurements

Generation wall-time spans the first worker starting generation to the last
finishing, excluding engine startup and grading. Tokens/s uses actual output
tokens, not a nominal token budget. DP2 startup is the maximum worker startup.

| Profile | Generation s | Output tokens | Tokens/s | Startup s | Evaluation wall s |
| --- | ---: | ---: | ---: | ---: | ---: |
| TP1, chunks, 32 | 871.099 | 363001 | 416.72 | 134.70 | 1046.29 |
| DP2, chunks, 32/GPU | 429.035 | 365448 | 851.79 | 13.32 | 477.47 |
| DP2, continuous, 32/GPU | 355.562 | 364445 | 1024.98 | 13.25 | 405.75 |
| DP2, continuous, 64/GPU | 331.939 | 355914 | 1072.23 | 12.89 | 382.96 |
| DP2, continuous, 128/GPU | 356.131 | 357114 | 1002.76 | 13.16 | 411.57 |

The first engine paid cold download/compilation costs; later engines reused the
session caches. Do not attribute the startup difference to parallelism.
Setup plus all profiles took 2830.79 seconds (47m11s); dependency installation
took 74.30 seconds and import/device checking took 13.51 seconds.

DP2 doubled generation throughput (2.04x); eliminating chunk barriers added
20.3%; increasing concurrency from 32 to 64 added 4.6%. The combined fastest
profile was 2.57x the single-GPU baseline by output tokens/s. Concurrency 128 was
slower than 64, despite higher instantaneous utilization on one GPU.

These are single trials, not statistically established speedups. Output lengths
vary with batching and floating-point execution despite fixed seeds; the 64
profile produced about 2.3% fewer tokens than continuous-32. Confirm the modest
64-versus-32 improvement with repeats before treating it as a universal default.

## Scores and GPU usage

| Profile | MBPP % | BFCL % | SVAMP % | IFEval % | Capped answers |
| --- | ---: | ---: | ---: | ---: | ---: |
| TP1 chunks 32 | 46 | 75 | 76 | 56 | 13 |
| DP2 chunks 32 | 48 | 76 | 78 | 55 | 11 |
| DP2 continuous 32 | 49 | 77 | 77 | 54 | 12 |
| DP2 continuous 64 | 51 | 76 | 79 | 55 | 9 |
| DP2 continuous 128 | 47 | 75 | 78 | 53 | 13 |

Scores show no large observed regression; 100 questions per benchmark cannot
establish quality equivalence. For the winner versus TP1, paired better/worse
answer counts were MBPP 7/2, BFCL 1/0, SVAMP 5/2, IFEval 5/6.

Telemetry sampled `nvidia-smi` approximately once per second and was filtered to
the generation wall-time windows (including imbalance between DP workers).

| Profile | Mean GPU0 active % | Mean GPU1 active % | Peak GPU0 MiB | Peak GPU1 MiB |
| --- | ---: | ---: | ---: | ---: |
| TP1 chunks 32 | 75.8 | 0.0 | 11163 | 3 |
| DP2 chunks 32 | 72.9 | 72.2 | 11177 | 11125 |
| DP2 continuous 32 | 89.0 | 77.9 | 11125 | 11125 |
| DP2 continuous 64 | 86.0 | 87.9 | 11235 | 11235 |
| DP2 continuous 128 | 91.4 | 70.9 | 11897 | 11685 |

GPU-active utilization is not achieved FLOPS, tensor-core occupancy or bandwidth
utilization. Sampling overhead and roughly one-second resolution also limit
precision. Worker-tail imbalance remains: GPU1 was below 10% active for 11.2%
of the 128-profile generation window, versus 1.9% for the 64 profile.

## Implementation and reuse

`evaluate.py --continuous-queue` queues all requests in one vLLM engine and
steps it until finished, saving answers as they complete. DP2 uses one controller
with two spawned isolated engines, one per GPU, and merges records for grading.
`--max-num-seqs` limits active sequences **per engine**, not queue size.
The chunk baseline used `--batch-size 128`; continuous mode ignores chunk
barriers and queues 200 answers on each GPU.

The best measured configuration is `--tensor-parallel-size 1
--data-parallel-size 2 --continuous-queue --max-num-seqs 64 --dtype half`.
The continuous-32 profile remains the clean baseline before concurrency tuning.

`cloud/optimize_eval.py` runs profiles sequentially with bounded timeouts,
checks GPU processes are gone before advancing, validates artifacts, and rejects
large score/token drifts from automatic winner selection. The heuristic is not
a statistical quality-equivalence test.

Local artifacts live in `runs/kaggle-optimization-v1/`: summary at
`runs/optimization/optimization-summary.json`, all profile results/records and
worker events, setup timing, utilization samples, and the Kaggle log. Only
result artifacts were downloaded, not environments or model weights.
All 114 local unit tests passed before submission; continuous-engine behavior
was additionally exercised successfully on the real GPUs.
