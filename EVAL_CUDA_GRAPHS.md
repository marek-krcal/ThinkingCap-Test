# Decode-only CUDA graph trial — 400 questions

Kaggle `krcalkrcal/thinkingcap-cuda-graphs`, version 2, completed October 10,
2026. Runner `6e726eaf9e0f5105355d06e07d2e41d066eceba4`; wrapper `65ce619`.
Version 1 failed before installation because GitHub rejected a short fetch SHA;
its terminal ERROR status was verified before the corrected submission.

## Controlled recipe and audit

Same frozen 100 questions each MBPP, BFCL, SVAMP, IFEval, one answer each;
Qwen/Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`;
FP16, two independent single-T4 engines, continuous queue, 64 active sequences
per GPU, prefill budget 2048, memory fraction 0.70. All 400 question IDs,
per-answer seeds, model, sampling, grader revisions and token caps match the
previous `dp2-continuous-64` profile. All 400 unique answers were graded, with
zero grader errors and no truncated answer counted correct.

Only inference execution change: `enforce_eager=False`, compilation mode 0
(no torch.compile), `FULL_DECODE_ONLY`, capture sizes
`[1,2,4,8,16,24,32,40,48,56,64]`. Both engines' logs confirm successful graph
capture, about one second and 0.19 GiB each. Mixed/prefill batches remain eager.

## Measured comparison

| Metric | Eager baseline | Decode graphs |
| --- | ---: | ---: |
| Generation wall time | 331.939 s | 359.622 s |
| Generated tokens | 355,914 | 355,342 |
| Generation throughput | 1,072.23 tokens/s | 988.10 tokens/s |
| Startup including health check | 12.893 s, warm | 133.817 s, cold |
| Evaluation including startup/grading | 382.961 s | 538.064 s |
| Setup plus evaluation | shared campaign setup | 625.113 s |
| Peak memory per GPU | 11,235 MiB | 11,425 MiB |
| MBPP correct / 100 | 51 | 50 |
| BFCL correct / 100 | 76 | 75 |
| SVAMP correct / 100 | 79 | 78 |
| IFEval correct / 100 | 55 | 55 |
| Truncated answers | 9 | 11 |

Generation wall time is `max(worker generation_finished) -
min(worker generation_started)`, excluding startup and process cleanup.
The parent `generation_only_seconds=368.797` includes orchestration overhead;
its reported `tokens_per_second=706.988` also includes startup and is **not**
the generation-only throughput used above.

Graph generation took 8.34% longer, with 7.85% lower tokens/s. Output length
changed only -0.16%, so aggregate length does not explain the slowdown.
Scores differ by 0–1 point per benchmark, with 13 paired correctness changes;
this is not evidence of a meaningful quality change on a single 100-question
slice. Fixed seeds do not guarantee bit-identical sampling after execution
changes, and differing long-answer distribution affects the tail.

One-second telemetry aligned to the global generation window:

| GPU | Mean busy | Median busy | Below 10% busy | Peak memory |
| --- | ---: | ---: | ---: | ---: |
| 0 | 77.10% | 95% | 18.31% | 11,425 MiB |
| 1 | 95.32% | 96% | 0% | 11,425 MiB |

GPU 0 finished its 200 jobs in 293.020 seconds; GPU 1 needed 359.610 seconds,
leaving an approximately 67-second imbalance tail. High busy percentage is
not equivalent to useful tokens/s: padding/captured work and workload shape
can affect efficiency. We did not profile individual kernels or graph replay
coverage, so the precise slowdown cause is not established.

Cold graph startup included fresh model loading and FlashInfer warmup; the
baseline was a later warm profile in the earlier campaign. Do not attribute
the whole startup difference to graph capture itself. Dependency installation
took 61.794 seconds, imports 17.363 seconds; questions were restored from cache.

## Decision

Keep the measured eager DP2 continuous-64 baseline in PR #1. This single
decode-graph trial did not improve performance. It does not rule out other
graph modes, compiler fusion, training graphs, or a repeated interleaved
comparison; none of those additional GPU runs was launched.

Artifacts are local under `runs/kaggle-cudagraph-v2/`: `setup-timing.json`,
`gpu-utilization.jsonl`, `run-config.json`, `thinkingcap-cuda-graphs.log`, and
`runs/evaluation/{results.json,records.jsonl,events.jsonl,shards/}`. Only results,
logs and telemetry were downloaded, not environments or weights.
