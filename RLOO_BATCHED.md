# Batched binary overfitting repeat

This branch retains the original tiny probe and PR#1 evaluation baseline.
It implements fresh vLLM rounds for binary test-set overfitting, not the
brainstorm plan's later brevity/KL objective or clean generalization setup.

## Fixed recipe

- Same pinned Qwen3-0.6B, FP16 base, FP32 rank16/alpha32 LoRA q/k/v/o,
  dropout0, AdamW lr1e-4, gradient clip1; no warmup change in this comparison.
- Same 16 TEST question IDs. A RLOO round repeats those IDs twice as 32
  independently seeded prompt instances, with eight completions each (256).
- Temperature1, top-p1, top-k disabled, min-p0, repetition1, no reward/KL
  penalty. Incorrect, malformed or truncated outputs receive0, correct1.
- Same training stop token151645 as the original HF probe; vLLM's model EOS
  defaults are explicitly disabled in favor of this training stop policy.
- 1024 generated-token cap, including EOS. Stop reasons and cap hits are
  recorded separately. No per-group120-second stop: a bounded whole-trainer
  alarm aborts incomplete rounds, which are never partially updated.
- RLOO leave-one-out advantages within each group8; no standard-deviation
  normalization. One accumulated update with denominator256. Exact zero
  advantage terms skip forward/backward, but not the denominator.
- GPU0 is learner; GPU1 is a persistent eager FP16 vLLM engine, continuous
  queue64, prefill2048, memory0.70. No overlapping policy updates/generation.
- Once per round save a scratch LoRA snapshot, verify hash in the worker,
  remove the old adapter, register a distinct new adapter ID, check every
  input request's actual adapter route. Prefix cache disabled.
- Check sampled vs learner token log-probabilities on two trajectories each
  round; fail before updates if mean abs difference>.05 or max>.5 (FP16
  backend numerical tolerances, not a claim of exact arithmetic equivalence).
- Transformer conditions on the full prompt; only completion predictors are
  projected through the frozen LM head. Cross-entropy is chunked128 positions
  with activation recomputation, avoiding full sequence-vocabulary logits.
  CPU Qwen/LoRA tests check loss and gradient equivalence against the unfused
  implementation. This is a PyTorch chunked path, not a Liger dependency.

## SimPO implementation, not launched in this RLOO campaign

64 prompt instances x8 =512 trajectories per round, refreshed current policy.
At most one complete correct/incorrect pair per prompt instance; no brevity
pairs yet. Exclude truncated responses from pairs. One pass over up to eight
balanced disjoint pair minibatches, no pair replay. Existing beta2/gamma1
preserved. Chunked completion scoring is shared. This deliberately differs
from the brainstorm's future compression margins/rewards.

## Budget and comparison

19-minute training alarm (including loading, vLLM startup, rollouts, grading,
updates), outer20-minute hard cap. Only a finite checkpoint with completed
updates receives the unchanged eager DP2 continuous64 400x1 evaluation, with
original caps MBPP8192/others4096 and evaluation sampling .6/.95/20.
Whole Kaggle session cap3600 seconds includes installation/training/evaluation.

Log round phase timings, tokens/s, rewards, informative/all-correct/all-wrong
groups, cap numerator/denominator and cumulative hit rate, actual stop reasons,
probability differences, gradient norm, updates/skips. One-second GPU telemetry.
Compare to prior RLOO:100 rollouts,25 groups,5 updates,1110.196s training,
11 ceiling hits and4 earlier time stops; evaluation60/78/84/57 versus base
51/76/79/55. Batch size, sample multiplicity and update cadence change, so
report learning and throughput separately; not a pure one-factor speed test.

Artifacts: campaign/training summaries, training events/IDs, rollouts with
token IDs/sampler probabilities, small adapter, 400 evaluation records/results,
events, setup timing and GPU telemetry. Snapshots/environments/base weights
remain in `/kaggle/temp`, never downloaded. No longer training runs without
human approval. No automatic extra hyperparameter sweeps are authorized.

## Completed GPU repeat: October 13, 2026

[Kaggle version 1](https://www.kaggle.com/code/krcalkrcal/thinkingcap-batched-rloo-overfitting)
is COMPLETE. Runner `26d1c2f1bf700eeda2d2454a420052089c846728`, wrapper
`74957d7`; metadata subsequently corrected to Kaggle's actual generated slug.
No corrective replacement, overlapping session, longer training or additional
hyperparameter run was submitted. All operations used CLI/API; no browser.

**Throughput improved substantially, but this checkpoint did not beat the
previous RLOO probe's evaluation score.** These are intentionally contaminated
test-overfit debugging results, not a valid upstream submission or clean
generalization evidence. Iterative SimPO is implemented but not GPU-tested
or trained in this campaign.

### Training audit and efficiency

| Measurement | Previous Transformers RLOO | Batched vLLM RLOO |
| --- | ---: | ---: |
| Training including loading and save | 1,110.196 s | 1,131.510 s (18m52s) |
| Generated trajectories | 100 | 1,024 |
| Generated tokens | 46,422 | 455,091 |
| Prompt instances / groups | 25 × 4 completions | 128 × 8 completions |
| Completed optimizer updates | 5 | 4 |
| Informative groups | 5 | 47 |
| All-correct / all-incorrect groups | 13 / 7 | 59 / 22 |
| Nonfinite skips | 0 | 0 |
| Mean binary reward over all rollouts | 0.6400 | 0.6885 |
| Ceiling hits | 11/100 (11%) | 119/1,024 (11.62%) |
| Other early time-stop truncations | 4 | 0 |

The new run processed **10.05× trajectories per training-wall-second** and
**9.62× generated tokens per training-wall-second**. These are observed
end-to-end processing-rate ratios, not an isolated vLLM or kernel speedup:
the new run uses the second GPU for generation, larger queues, K=8 rather
than 4, one update per 256 rather than per 4 informative trajectories, and a
different scoring path. The old run has no precise generation-only timer.
Four optimizer steps versus five also means these results do not establish
faster learning or faster overfitting merely from greater sample throughput.

All four rounds completed, with 256 trajectories each, unique round/group/
sample identities and seeds, the same 16 unique training IDs, and distinct
verified adapter hashes/IDs at each policy refresh. The unchanged 16 IDs each
received 64 trajectories total. Each update used denominator 256; there were
376 nonzero-advantage trajectories across 47 informative groups. All 256 terms
remain in the denominator even when zero-gradient terms skip computation.

| Round | Reward mean | Ceiling hits /256 | Informative groups /32 | Generation | Update |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.6367 | 36 (14.06%) | 14 | 202.932 s | 45.074 s |
| 1 | 0.6758 | 32 (12.50%) | 12 | 195.172 s | 40.024 s |
| 2 | 0.7148 | 27 (10.55%) | 11 | 188.232 s | 38.243 s |
| 3 | 0.7266 | 24 (9.38%) | 10 | 162.533 s | 31.320 s |

Totals: generation 748.868 s, update 154.661 s, grading 12.668 s,
snapshot/save+engine registration 1.080 s. Aggregate generation throughput
**607.705 tokens/s** on the rollout GPU. Initial engine readiness occurred
at 161.262 s including learner/module loading; vLLM initialization 126.015 s.
The small two-trajectory 16-token preflight added approximately 50.1 s and is
excluded from the 1,024 training trajectories. No incomplete round was used.

Sampler/learner checks passed: per-round mean absolute token log-probability
differences 0.00237–0.00257, maximum 0.03099–0.04984. This checks selected
trajectories, not every token of all 1,024 outputs. All completed trajectories
retain their actual selected-token sampling probabilities for further audit.
905 terminated naturally at the explicit training EOS 151645; 119 stopped by
length, all exactly 1,024 tokens. None stopped below the cap for time reasons.
Unclipped gradient norms 10.00–12.89 were finite, then clipped to 1.

The saved 18,380,008-byte adapter has 224 finite tensors and nonzero LoRA B
weights. Its SHA256 matches both campaign and training summaries:
`b05a15e4131d84e8687d7c0bdcf271d7b54d757334a89b964fa8537ee95ef1a2`.

### Matched 400×1 evaluation

All 400 answers are unique, graded with zero infrastructure errors, and match
the baseline's IDs, seeds, model/cohort/grader revisions, sampling, caps and
hardware recipe. Every record has verified engine input adapter routing
and explicit vLLM completion metadata. No truncated or malformed-thinking
answer was counted correct; no CPU correction/regrading was needed.

| Correct /100 | Unadapted base | Previous RLOO | Batched RLOO |
| --- | ---: | ---: | ---: |
| MBPP+ | 51 | 60 | 49 |
| BFCL | 76 | 78 | 77 |
| SVAMP | 79 | 84 | 79 |
| IFEval | 55 | 57 | 63 |
| Total /400 | 261 | 279 | 268 |

Paired against the base: 33 gains, 26 losses, net +7 (+1.75 percentage points).
Against previous RLOO: 21 gains, 32 losses, net −11 (−2.75 points). A diagnostic
paired-question bootstrap (10,000 resamples, seed 42) gives 95% intervals
−2.0 to +5.5 points versus base and −6.25 to +0.75 versus prior RLOO. This one
400-question, one-training-seed trial does not statistically separate the
overall scores; the coding decrease nevertheless merits inspection.

On actual training-seen IDs: base 11/16, previous RLOO 14/16, new 12/16.
On the 10 IDs producing gradient signal in this new run: base 7/10,
previous RLOO 10/10, new 8/10. The previous run's gradient-signal set was
smaller, so this is not a matched gradient-exposure comparison.
Other 384 IDs: base 250, previous 265, new 256 correct. The two seen-question
losses versus previous RLOO were `Mbpp/451` and `Mbpp/79`: both completed
and parsed, but failed their program tests, not a grader infrastructure issue.
No claim of successful full memorization or clean generalization is made.

Evaluation total output tokens 314,405 versus base 355,914 (11.66% fewer),
but 1.85% more than prior RLOO 308,687. Thinking tokens 285,189 versus
base 323,722 (11.90% fewer) and prior 283,265. These length changes occurred
without a length reward and are not by themselves a compression success.
New mean/median/p90 thinking length 712.97/321/1,447.4 tokens; total response
length 786.01/377.5/1,552.2. Twelve evaluation truncations (MBPP 6, SVAMP 3,
IFEval 3) versus baseline 9 and prior RLOO 3; all scored wrong.

Global generation window (earliest worker start to latest worker finish)
**692.210 s**, 454.204 tokens/s; prior RLOO 681.631 s, 452.865 tokens/s. Parent
generation-only timing 699.832 s also includes shutdown overhead. Max worker
startup 25.031 s. This shows essentially similar adapter evaluation throughput,
not a speedup to evaluation from the new training pipeline.

### GPU telemetry, quota and artifacts

Actual median telemetry cadence 1.041 s, maximum gap 5.99 s; 1,807 samples.
Numbers below are sample averages and sampled peak allocation, not kernel-
level utilization or guaranteed instantaneous memory peaks.

| Phase | GPU0 mean utilization / max sampled MiB | GPU1 mean utilization / max sampled MiB |
| --- | ---: | ---: |
| Training rollout generation | 0% /1,779 | 51.35% /11,219 |
| Training updates | 71.53% /1,781 | 0% /11,219 |
| Evaluation generation | 46.34% /11,329 | 50.11% /11,327 |

Generation and updates are deliberately sequential; the persistent vLLM KV
cache retains memory on GPU1 during learning but does not perform generation.
This arrangement still leaves one GPU idle in each training phase. Improving
that duty cycle is a separate optimization; no extra campaign was launched.

Dependency installation 71.751 s, import check 11.950 s. Training+evaluation
campaign 1,888.130 s (31m28s). Charged quota delta 1,993.348 s (33m13s).
Post-run used 9,933.764/21,600 s (2h45m34s, 45.99%), reserved 0; remaining
11,666.236 s (~3h14m26s), reset October 17 00:00 UTC /02:00 Prague.

Filtered artifacts are under `runs/kaggle-rloo-batched-v1/`, specifically
`runs/rloo-batched/training/` and `runs/rloo-batched/evaluation/`, plus setup
timing, one-second telemetry and logs. No environment, base weights or scratch
snapshots were downloaded. Session COMPLETE verified; monitor paused after
the terminal audit. Next training-speed or learning-cadence experiments
require a new scoped request; longer training still requires explicit approval.
