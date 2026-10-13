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
