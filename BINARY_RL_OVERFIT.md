# Binary-reward SimPO / RLOO plumbing experiment

**Intentionally contaminated test-set experiment. Not a valid leaderboard or
generalization result.** The upstream README prohibits evaluation-set training
for submissions; this separate experiment is only an explicitly requested
overfitting/debug probe. Do not mix these adapters into clean future experiments.

`train_rl.py` shares the pinned Qwen3-0.6B checkpoint, LoRA setup, benchmark
graders, logging, time budget and checkpoint saving between the two algorithms.
The existing `train.py` rollout demonstration is unchanged.

- SimPO: fixed base-policy preference pool, strict correct-versus-incorrect
  pairs, `-logsigmoid(2*(mean_logp_chosen-mean_logp_rejected)-1)`.
  Ties are skipped, never broken by length. The standard mean-log-probability
  normalization remains, but no token-count reward is used.
- RLOO: fresh groups of four from the current policy, leave-one-out reward
  baseline, one REINFORCE update per group. Sequence **sum** log probability,
  no advantage standardization, replay, critic, reference policy or KL penalty.
- Reward exactly 1 for a correctly graded completed answer, 0 for wrong or
  truncated trajectories. Infrastructure/grader failures raise errors rather
  than being silently interpreted as wrong answers.
- Sampling for training is temperature 1, unrestricted top-p/top-k, with all
  generated tokens including EOS scored. Prompt/pad tokens are excluded. This
  intentionally differs from the evaluation sampling recipe so vanilla
  on-policy RLOO's scored and generating distributions match.

## Short probe configuration

Single-T4 FP16 base weights, FP32 LoRA/optimizer state, rank 16, alpha 32,
q/k/v/o projections, dropout 0, AdamW learning rate 1e-4, no weight decay,
gradient clipping 1, FP16 gradient scaling, checkpointed forwards and
microbatched trajectory backward passes.

At most 16 training questions (four per benchmark, BFCL balanced across its
three subsets), interleaved across tasks. These are inside the 400-question
evaluation slice. Actual sampled IDs and reward groups are logged. Model
loading, generation, grading and updates are all inside the 19-minute budget;
the launcher imposes a 20-minute training cap including final adapter saving.
At most 100 attempted updates. SimPO spends at most roughly seven minutes
collecting its fixed preference pool, then replays strict pairs. RLOO always
samples fresh trajectories after the latest optimizer update.

Training completions are capped at 1,024 tokens for a cheap first probe, with
a 120-second safety stop per rollout group. This can zero out otherwise
potentially correct longer trajectories; report truncation and useful groups.
It is a compute constraint, not a length reward, and the evaluation keeps the
original caps (MBPP 8,192, others 4,096). No claim of having solved overfitting
is warranted from a run with only a few informative updates.

Each adapter is evaluated with `cloud/run_eval.py --limit-per-benchmark 100
--samples 1`: FP16 eager vLLM, TP1/DP2, continuous queue, concurrency 64/GPU,
same pinned cohort/sampling/grading as the measured evaluation baseline.
Scores must be split into seen training questions and other questions in the
400 slice. Report paired baseline correctness changes, reward trajectories,
optimizer updates, nonfinite skipped steps, truncation, elapsed time and
grader errors. A short sanity run is not a statistical ranking of algorithms.

## Local verification / cloud execution

120 CPU tests passed, including exact objective derivatives, causal completion
mask boundaries and a tiny Qwen3 LoRA forward/backward with sliced logits.
Local numerical smoke tests require no full model download or GPU.

`cloud/run_rl_campaign.py` runs SimPO then RLOO and evaluates each; failure of
one algorithm is recorded without silently treating it as trained. The Kaggle
wrapper enables this campaign using the same isolated CUDA environment/cache
setup as evaluation. Disposable environments/weights stay in `/kaggle/temp`;
only small LoRA adapters, rollout logs, training summaries and evaluations are
written into `/kaggle/working`. No longer training runs are authorized without
asking the user first.

Sources: [SimPO paper](https://arxiv.org/abs/2405.14734),
[RLOO formulation](https://huggingface.co/docs/trl/v0.28.0/rloo_trainer).
