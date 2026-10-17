# Reusable offline trajectories for SimPO / SFT

## Scope update: task-aligned training subset (October 16)

The user replaced exhaustive training-source collection with subsampling toward
the test task types. `cloud/select_offline_training.py` prepares **2,000** training
prompts: GSM8K 1,400; Skywork 400; CommonsenseQA 100; ARC Easy 100. The size/mix is
an initial implementation assumption, not a user-specified exact number.
Selection uses declared arithmetic/story features and penalties for abstract
maths, with stable hash tie-breaking. It does not read test content or outcomes.
It is a heuristic, not measured similarity or a guarantee of improved scores.
GSM8K is closest to SVAMP. Skywork still includes harder arithmetic/combinatorial
stories; the small QA samples retain breadth. None covers code synthesis, tool
calling or rich instruction constraints directly.

Prepared files: `data/offline-subset/{train,validation,test}.jsonl` and
`selection.json`. Original IDs, seeds, messages, revisions and held-out splits
are retained. Train SHA: `e8d86ce8fb6c753864d88b752ac8fbe0f0f54f0b19cb417e06a69c81ed031543`.
Test and validation bytes are unchanged. At eight samples, the revised intended
target is **44,472** answers (16,000 train + 22,072 test + 6,400 validation),
superseding the old 402,968 target below.

**Selected campaign transition completed:** both old bounded a2 sessions finished
COMPLETE normally. Their immutable outputs were preserved; no more old full-training
shards may be launched. Dispatch
now blocks train jobs unless the data hash matches this selected subset. Before
new selected-training launches, migrate the frozen plan/private input/ledger to
the subset, reconcile reusable selected-ID answers from completed old outputs,
and skip their exact keys with seeds preserved. Do not overwrite the old campaign
or raw outputs, and do not silently regenerate completed selected trajectories.
Existing test collection can continue unchanged. No training is authorized.

The selected training plan has five shards / 16,000 answers and SHA-256
`860f6c66a5b75d1dc702f47720075830984520994ad88b1e72ffa7ebe385e8e2`.
The separate private input `krcalkrcal/thinkingcap-offline-subset-input` was
created with the selected train file, unchanged held-out files and provenance;
READY verified. Old test/validation keep their old input/ledger.
`cloud/reconcile_offline_subset.py` is stdlib-only: it validates original raw
hashes, model/sampling/caps/dtype, question messages/IDs/revision/seeds, rejects
duplicates, and recomputes selected binary correctness before importing. Raw
selected answers are copied only within Kaggle, never downloaded locally.

After **all old training attempts COMPLETE and audited** by legacy refresh:

```sh
# CPU-only, no GPU. Full SHA must already be published. Never blindly retry push.
.venv-kaggle/bin/python cloud/dispatch_subset_reconcile.py --commit FULL_SHA
# Once this exact CPU notebook is COMPLETE, download small indexes/labels only:
.venv-kaggle/bin/python cloud/dispatch_subset_reconcile.py --audit
.venv-kaggle/bin/python cloud/refresh_offline.py --data data/offline-subset \
  --state runs/offline-subset-campaign
# With a free GPU slot and fresh quota/draft checks:
.venv-kaggle/bin/python cloud/dispatch_offline.py --job train-000 \
  --data data/offline-subset --state runs/offline-subset-campaign \
  --dataset krcalkrcal/thinkingcap-offline-subset-input \
  --prefix 'ThinkingCap offline subset' --commit FULL_SHA --submit
```

The audited CPU bootstrap becomes attempt 1 for each new shard (possibly zero
reused answers); its raw records remain in the CPU notebook output. The first GPU
attempt is therefore `thinkingcap-offline-subset-train-000-a2`, not an old-name
replacement. Its attached bootstrap has one index per shard, selected by job ID.
New GPU attempts generate only missing keys. New dispatch is blocked until the
bootstrap audit succeeds and requires distinct state/input/notebook namespaces.
The two ledgers share a dispatch lock and account-wide capacity checks, including
locally pending/unlisted submissions. The existing CPU grading slot also excludes
concurrent reconciliation. Do not submit reconciliation before old a2 completes:
it must import all completed old selected answers, including a2's new output.

### Transition audit, October 17 shortly after midnight Prague

- Old test-000 a2 saved 714 new answers / 1,647,672 tokens; cumulative 1,588.
- Old train-000 a2 saved 795 new answers / 2,163,327 tokens, 174 correct and
  411 capped; cumulative legacy-source answers 1,705. Most are now outside the
  selected scope and must not be counted toward its completion.
- CPU-only `thinkingcap-offline-subset-reconcile` v1 COMPLETE: **48** selected
  existing answers imported, 18 correct, 10 capped, 94,864 tokens. Hashes,
  identities, seeds and original grades passed; zero regenerated answers and
  no model runtime/GPU. Audited in `runs/offline-subset-campaign/ledger.json`.
- New GPU collectors: `thinkingcap-offline-test-001-a1` v1 and
  `thinkingcap-offline-subset-train-000-a2` v1, pinned
  `f33b1980370911d6154e7cacef336c21c20a5d91`. The selected shard's bootstrap
  skips 8 existing answers; the other imported answers belong to later shards.
- CPU grader `thinkingcap-offline-grade-test-000-a2` v1 grades the new 714 saved
  test answers independently. Keep at most these two GPU collectors plus one
  campaign CPU job; do not dispatch another while both GPU slots are occupied.
- Quota before selected launch: **22,144.62 / 108,000 seconds** used, refresh
  October 17 00:00 UTC / 02:00 Prague. No quota reset has yet occurred.

Run `cloud/report_offline_scope.py` after both ledger refreshes for authoritative
**revised-scope** counts: it includes original test/validation plus selected
training, excludes legacy out-of-scope training, validates seeds and rejects
duplicated graded keys. It saves `runs/offline-subset-campaign/scope-progress.json`.
The legacy refresh still reports its old historical target; that is not the
authorized current target. Revised target remains **44,472**, not 402,968.
161 local regression tests pass. No actual training has been run.

### Selected collector startup stall, October 17 around 00:27 Prague

`thinkingcap-offline-subset-train-000-a2` v1 never exposed a launcher/setup log
after more than 20 minutes RUNNING. Both log endpoints and the exact-session SSE
were empty; its notebook page showed **Output 0 B**. The cause is not established
(provider startup versus an unlogged initial fetch). Session **359151837** was
cancelled through the API and **CANCEL_ACKNOWLEDGED verified**. No new committed
raw output exists; the CPU bootstrap's 48 selected answers remain intact.

The launcher now prints immediately and fetches pinned setup code in a bounded
90-second subprocess, so a DNS/fetch stall cannot bypass its timeout once Python
starts. A replacement can use the audited bootstrap only after the failed
session remains stopped and fresh quota/concurrency are checked. It gets a new
attempt identity; no overlapping retry or silent reuse of the cancelled output.
162 tests pass, including rejecting retries without zero-output/stopped evidence.
Independent `thinkingcap-offline-test-000-a3` is running; test-001's CPU grader
has completed and its 3,200 labels need refresh/audit.

At **00:37 Prague**, the one bounded replacement
`thinkingcap-offline-subset-train-000-a3` v1 (runner
`c9568846991a0ad5ed0a844d2475d122b1946b45`) passed startup and is **actually
generating on both T4s**, with 96–97% utilization. Its logged command uses the
selected input/plan and `reconciled/bootstrap/train-000/collection-index.json`;
eight saved keys are skipped (3,192 missing answers / 1,596 per GPU). The prior
zero-output stall's cause remains unknown; successful retry does not establish
that the bounded-fetch change was the cause of recovery. Test-000 a3 is also
generating, with its expected 806 missing requests per GPU.

Test-001 CPU grades are now audited: all 3,200 answers, 2,643 correct, zero capped,
632,212 tokens. Revised-scope saved/graded coverage is **4,836/44,472**, comprising
4,788 test answers plus 48 reused selected training answers. Current running
answers are not included until sealed/audited. The CPU grading slot is free.
Quota before the successful retry: 23,692.33 / 108,000 seconds used, reserved 0;
refresh remains 02:00 Prague. No new training runs or recipe changes occurred.

At **01:16 Prague**, test-000 a3 is COMPLETE and its CPU grades are audited:
529 new answers, 244 correct, 49 capped, 1,306,842 tokens. Test-000 cumulative
coverage is 2,117/3,200 (1,070 correct, 178 capped); it remains partial, with
1,083 keys missing. The revised scope now has **5,365 saved and graded** answers
(5,317 test + 48 selected training), no pending CPU grades at this checkpoint.
`thinkingcap-offline-test-002-a1` v1 was submitted in the freed GPU slot with
runner `5252b906ee0e0513e3eb2a130ad8b429f935de28`; selected train-000 a3 continues
independently. Quota before test-002 launch: 27,749.42/108,000 seconds used,
reserved 0, refresh 02:00 Prague. Fresh draft OFF and account-wide capacity checks
passed. No old full-training collection was resumed.

At **01:32 Prague**, revised coverage is **9,658 saved AND graded / 44,472**:
8,517 test answers and 1,141 selected training answers, zero awaiting CPU grading.
The scope audit found no duplicate keys or invalid seed/truncation labels.
Selected train-000 a3 finished its normal deadline with 1,093 new answers
(385 correct, 343 capped, 2,559,127 tokens), bringing that shard to 1,101/3,200
including its eight reused answers. Its generation-only window was 2,563.07s;
cold engine startup 130.74s, total generation stage 2,693.82s. Partial-deadline
rates are completion-biased and must not be treated as final source scores.

| Selected train-000 a3 source | New answers | Correct | Capped | Tokens |
| --- | ---: | ---: | ---: | ---: |
| GSM8K | 753 | 283 | 205 | 1,759,320 |
| Skywork | 224 | 24 | 138 | 741,566 |
| CommonsenseQA | 68 | 43 | 0 | 31,279 |
| ARC Easy | 48 | 35 | 0 | 26,962 |

Test-002 a1 finished all 3,200 answers and its separate CPU grades are audited:
2,550 correct, zero capped, 699,079 tokens. Generation-only 427.38s, cold startup
137.17s, generation stage 564.55s. This mostly BFCL shard is much shorter than
the selected arithmetic or MBPP streams; it cannot support a single corpus ETA.

New GPU submissions are `thinkingcap-offline-test-003-a1` v1 and
`thinkingcap-offline-subset-train-001-a2` v1, both pinned
`2ea857071b915bd8cf048bdc76860e968ea4b68f`. The selected shard attaches its
audited bootstrap and must skip 16 exact saved keys (3,184 missing). Their
actual queues passed fresh startup verification at 01:35 Prague: test queues
360/1,600 and 300/1,600 completed, both T4s 92–95% busy; selected train queue
40/1,592 completed with both T4s 96–97% busy. The expected 1,592 requests/GPU
confirm that its 16 reused answers were excluded. CPU slot is free.
Before the second launch quota was 29,085.42/108,000 seconds used, reserved 0;
refresh still 02:00 Prague. Fresh draft OFF checks preceded both submissions.
The full revised corpus remains incomplete; no training is being run.

At **02:04 Prague**, test-003 a1 and its CPU grader are COMPLETE and audited:
3,200 unique answers, 2,786 correct, 49 capped, 2,137,744 tokens. Controller
generation-only time 1,249.47s, engine startup 137.83s, total generation stage
1,387.30s. Revised-scope coverage is now **12,858 saved AND graded / 44,472**,
with zero awaiting grading (11,717 test + 1,141 selected training answers).
No duplicate or invalid truncated-correct labels were found.

The free GPU slot was filled by `thinkingcap-offline-test-004-a1` v1, pinned
`26c38141396bd2ebcd16f4487bc8a979e9fc82c3`, submitted October 17 00:01:26 UTC.
It is the next disjoint 400-prompt SVAMP shard; actual startup needs verification.
Selected train-001 a2 continues independently, with both T4s busy in fresh logs.
CPU slot is free. Kaggle's fresh pre-launch quota confirmed the legitimate
weekly reset: used 0, reserved 0, allowed 108,000 seconds, next refresh
October 24 00:00 UTC. This is a provider-reported reset, not a quota workaround.
Fresh unrelated-draft OFF verification preceded dispatch. No training authorized.

At **02:17 Prague**, selected train-001 a2 is COMPLETE and audited, sealing
a normal deadline partial: 1,014 new answers, 380 correct, 342 capped,
2,470,855 tokens. Cumulative shard coverage is 1,030/3,200 including its 16
reused keys. Generation-only 2,560.68s, cold startup 133.88s, stage 2,694.56s.
New answers by source (answers / correct / capped / tokens): GSM8K
722 / 277 / 226 / 1,797,257; Skywork 185 / 22 / 116 / 610,712;
CommonsenseQA 44 / 28 / 0 / 22,340; ARC Easy 63 / 53 / 0 / 40,546.
These deadline partials remain completion-biased, not final source metrics.

Revised coverage is **13,872 saved AND graded / 44,472**, zero awaiting grading:
11,717 test + 2,155 selected training answers. Test-004 has actual generation
on both T4s (93–94% utilization, queues 240 and 220/1,600 at 02:13).
The freed slot now holds `thinkingcap-offline-subset-train-002-a2` v1,
runner `9a22fd37d33c032bf1259fc0b1e6c0de6a63909e`, submitted 00:16:57 UTC.
It attaches its audited bootstrap: skip 24 exact saved keys, generate only
3,176 remaining / 1,588 per GPU. Actual startup still needs verification.
Fresh draft OFF, capacity and quota checks passed: used 621.23 / 108,000s,
reserved 0, refresh October 24 00:00 UTC. CPU slot is free. No actual training.

At **02:42 Prague**, test-004 a1 and its CPU-only grader are COMPLETE/audited:
3,200 answers, 2,882 correct, 35 capped, 2,135,856 tokens. Generation-only
1,199.80s, cold startup 133.94s, stage 1,333.74s. Revised coverage is
**17,072 saved AND graded / 44,472**, none awaiting grading (14,917 test +
2,155 selected training). Full coverage remains false; four test shards are
complete, test-000 remains partial, and validation is still pending.

`thinkingcap-offline-test-005-a1` v1 was submitted at 00:39:52 UTC, pinned
`52239e68ced992f3f96abe1766453ef6599ea073`, replacing the completed test GPU
session. Actual startup needs verification. Selected train-002 a2 is generating
on both T4s (96–97% utilization); its command uses the selected input/plan and
train-002 bootstrap, with 1,588 requests/GPU confirming 24 keys excluded.
CPU slot is free. Fresh pre-launch quota: 2,828.51/108,000s used, reserved 0,
next refresh October 24 00:00 UTC. Fresh unrelated draft OFF/capacity checks
passed. No training, cap changes or duplicate regeneration were introduced.

At **03:10 Prague**, selected train-002 a2 is COMPLETE/audited with 1,137
new answers, 400 correct, 315 capped, 2,660,604 tokens; cumulative 1,161/3,200
including 24 reused answers. Generation-only 2,561.96s, startup 132.52s,
stage 2,694.47s. New per-source answers / correct / capped / tokens:
GSM8K 837 / 334 / 194 / 1,911,791; Skywork 212 / 21 / 121 / 704,695;
CommonsenseQA 48 / 13 / 0 / 26,901; ARC Easy 40 / 32 / 0 / 17,217.
Partial-deadline selection remains biased and is not a final source score.

Test-005 a1 sealed all 3,200 answers / 2,170,432 tokens, generation-only
1,162.92s, startup 134.28s, stage 1,297.20s. Its CPU-only grader
`thinkingcap-offline-grade-test-005-a1` v1 is RUNNING, so these answers are
saved but not yet counted as graded. Revised scope: **21,409 saved / 18,209
graded / 44,472 target**, 3,200 awaiting grading (3,292 selected train graded).

Both free GPU slots were filled with new disjoint jobs, pinned
`4b9cfaca145b0daa4bfe9a841274614b7dc31229`: `thinkingcap-offline-test-006-a1`
v1 (359 IFEval prompts, 2,872 answers, submitted 01:06:27 UTC) and
`thinkingcap-offline-subset-train-003-a2` v1 (400 selected prompts, 3,200 answers,
zero bootstrap keys, submitted 01:08:11 UTC). Verify actual startup/queues next.
Before the second launch quota was 5,786.43/108,000s used, reserved 0, next
refresh October 24 00:00 UTC. Fresh draft OFF checks preceded each launch.
Keep at most these two GPU jobs and one CPU grader; full coverage is not done.

### Silent CPU grader recovery, October 17 around 03:28 Prague

Test-005 CPU grader v1 exposed no logs or output after more than 21 minutes
RUNNING (browser reported 1,298.8s and Output 0 B; bounded CLI logs were empty).
Its exact session **359186126** was cancelled and **CANCEL_ACKNOWLEDGED verified**.
The cause remains unknown: provider startup and an unlogged initial fetch are
not distinguishable from this evidence. Both independent GPU collectors remain
healthy and untouched. All 3,200 raw answers are safe in test-005 a1 output.

The CPU dispatcher now logs immediately and fetches its pinned launcher in a
90-second subprocess, bounding even DNS stalls once Python starts. Retry is
still allowed only after an explicit terminal failed/cancelled status, preserves
the previous grade attempt in ledger history, and keeps GPU disabled. Two new
launcher regression checks bring the suite to **164 passing tests**. The repair
improves observability/bounds; it does not establish the prior stall's cause.
A single repaired CPU-only retry will grade the same saved answers; no model
runtime or regeneration is involved. Do not blindly repeat if it also stalls.

At **03:39 Prague**, the single repaired test-005 CPU grader **v2 COMPLETE**
and all 3,200 labels are audited: 2,316 correct, 77 capped, 2,170,432 tokens;
GPU used false and model runtime imported false. No raw answer was regenerated.
Successful retry does not establish why v1 never started. The launcher repair
and 164 tests are published at `5889cdbefb5eddb1403c96314c3694d72b29b730`.

Test-006 a1 also COMPLETE, sealing all 2,872 answers / 2,001,306 tokens;
generation-only 1,304.45s, startup 130.13s, stage 1,434.58s. Its CPU grader
`thinkingcap-offline-grade-test-006-a1` v1 is RUNNING. Revised coverage:
**24,281 saved / 21,409 graded / 44,472**, 2,872 awaiting grading. The latest
graded partitions are test 18,117 answers and selected train 3,292 answers.

The freed GPU slot now holds `thinkingcap-offline-test-000-a4` v1, submitted
01:36:15 UTC, pinned `5889cdbefb5eddb1403c96314c3694d72b29b730`. It attaches
test-000 a3 and must skip exactly 2,117 saved keys, generating only the remaining
1,083 (not a new seed cohort). Actual startup/resume skip still needs checking.
Selected train-003 a2 continues with both T4s busy. Fresh draft OFF and capacity
checks passed; quota before launch 8,923.51/108,000s used, reserved 0, refresh
October 24 00:00 UTC. Test-000, selected-train tails and both validation shards
remain unfinished. No training, recipe changes or duplicate collection.

At **03:41 Prague**, test-006 CPU grading is COMPLETE/audited: 2,872 labels,
1,692 correct, 107 capped, 2,001,306 tokens. Revised coverage is now **24,281
saved AND graded / 44,472**, zero awaiting CPU grading. Test accounts for
20,989 answers and selected training for 3,292; the remaining test deficit is
the 1,083 keys assigned to test-000 a4. Its startup command attaches test-000
a3 and routes exactly 542/541 remaining requests across the two GPUs, confirming
the 2,117-key skip. Actual engine generation still needs a fresh check.
Selected train-003 remains healthy on both T4s; CPU slot is free.

At **04:00 Prague**, selected train-003 a2 is COMPLETE/audited, a normal
deadline partial: **1,261** new answers, 469 correct, 353 capped, 2,925,627
tokens. Generation-only 2,564.42s, startup 129.86s, stage 2,694.29s.
Per-source answers / correct / capped / tokens: GSM8K 915 / 367 / 243 /
2,189,694; Skywork 216 / 24 / 110 / 666,983; CommonsenseQA 48 / 17 / 0 /
20,116; ARC Easy 82 / 61 / 0 / 48,834. These remain biased partial metrics.

Revised coverage is **25,542 saved AND graded / 44,472**, none awaiting
grading: 20,989 test + 4,553 selected training answers. The final fresh selected
shard, `thinkingcap-offline-subset-train-004-a2` v1, was submitted at 01:59:47
UTC, pinned `f240bddb5fbb88e953ff98caefd73eb9ad89cd93`, attaching its zero-key
audited bootstrap. Verify actual startup/3,200-request queue next. Test-000 a4
is still generating on both GPUs, with its exact 542/541 missing-key routing.
Fresh quota before launch: 11,456.45/108,000s used, reserved 0; refresh October
24 00:00 UTC. Fresh draft OFF/capacity checks passed. CPU slot is free.
All five selected shards still have missing answers; validation is unstarted.
The complete corpus remains pending and no training has been run.

At **04:30 Prague**, test-000 a4 is COMPLETE, sealing a normal deadline
partial of **991 new answers / 1,730,022 tokens**. Cumulative test-000 coverage
is 3,108/3,200: **92 test trajectories remain**, not a completed test corpus.
Generation-only 2,561.26s, startup 131.49s, stage 2,692.75s. Its separate
CPU grader `thinkingcap-offline-grade-test-000-a4` v1 is submitted; raw answers
are saved and immutable while grades await audit. Revised coverage is
**26,533 saved / 25,542 graded / 44,472**, 991 awaiting CPU grading.

The freed GPU slot now collects held-out validation:
`thinkingcap-offline-validation-000-a1` v1, submitted 02:29:27 UTC, runner
`98628265c0510fe4b2c3276811be94c047d83ca7`. This keeps validation separate from
the selected training subset. Verify actual startup/3,200 requests next.
Selected train-004 a2 remains healthy and generating on both T4s with its
expected 1,600-request queues. Fresh draft OFF and capacity checks passed;
quota 14,651.92/108,000s used, reserved 0, refresh October 24 00:00 UTC.
After CPU audit, test-000 a5 may attach a4 and skip 3,108 exact keys, but do not
launch it concurrently with either current GPU job. Full coverage is pending.

At **04:34 Prague**, test-000 a4 CPU grading is COMPLETE/audited: all 991 new
answers, 588 correct, 55 capped, 1,730,022 tokens. Revised coverage is **26,533
saved AND graded / 44,472**, no pending CPU labels. Test totals 21,980 answers;
selected training totals 4,553. The 92 missing test keys may now safely resume
from audited test-000 a4 when capacity frees. Both current GPU slots remain
reserved for validation-000 a1 and selected train-004 a2. Full coverage is false.

At **04:54 Prague**, selected train-004 a2 is COMPLETE/audited: 1,241 new
answers, 499 correct, 331 capped, 2,759,376 tokens. Generation-only 2,554.55s,
startup 139.17s, stage 2,693.71s. New per-source answers / correct / capped /
tokens: GSM8K 905 / 338 / 235 / 2,137,246; Skywork 164 / 20 / 96 / 541,946;
CommonsenseQA 76 / 46 / 0 / 39,227; ARC Easy 96 / 95 / 0 / 40,957.
All five selected shards now have audited deadline partials, totaling 5,794 /
16,000 answers, not full training-data coverage. Partial metrics remain biased.

Revised scope is **27,774 saved AND graded / 44,472**, none awaiting grading.
The freed slot resumes the final 92 missing test keys:
`thinkingcap-offline-test-000-a5` v1, submitted 02:53:50 UTC, pinned
`ffaf0fa53d86d52140560c40dd6ecf8f067e99f1`, attaches CPU-audited test-000 a4
and must skip exactly 3,108 keys (46 requests/GPU). Verify actual startup next.
Validation-000 a1 continues on both T4s; validation-001 remains unstarted.
Fresh quota before launch: 17,318.99/108,000s used, reserved 0; refresh October
24 00:00 UTC. Fresh draft OFF and capacity checks passed. CPU slot is free.
After the test-tail GPU finishes, prioritize validation-001 or selected resumes;
after test-tail CPU audit, verify the full 22,072-key test coverage separately.

At **05:10 Prague (03:10 UTC, October 17)**, test-000 a5 and its CPU-only
grader are COMPLETE/audited: 92 new answers, 70 correct, 1 capped, 117,972
tokens. Generation-only 260.55s, startup 135.28s, stage 395.83s; CPU grading
5.63s, no GPU/model runtime and no regeneration. Exact audit verified all
**22,072 unique test keys = 2,759 prompts x 8**, matching saved indexes and
original seeds, without duplicates or correct capped answers. Test collection
and grading are complete, not the full campaign.

Final test answers / correct / capped / tokens:

| Source | Answers | Correct | Capped | Tokens |
| --- | ---: | ---: | ---: | ---: |
| MBPP | 3,024 | 1,582 | 234 | 6,445,190 |
| BFCL simple Python | 3,200 | 2,649 | 0 | 626,634 |
| BFCL multiple | 1,600 | 1,218 | 0 | 345,072 |
| BFCL irrelevance | 1,920 | 1,596 | 0 | 423,255 |
| SVAMP | 8,000 | 7,080 | 115 | 5,405,769 |
| IFEval | 4,328 | 2,472 | 153 | 3,009,283 |
| Total | 22,072 | 16,597 | 502 | 16,255,203 |

Revised scope: **27,866 saved AND graded / 44,472**, zero awaiting grading,
5,794 selected-training answers; full campaign coverage remains false.
Validation-000 continues actual generation on both T4s. The freed GPU slot now
runs `thinkingcap-offline-validation-001-a1` v1, submitted 03:07:34 UTC, pinned
`6a3cd8812f7c0499c4287c614121b1bb2b08f0d1`, 400 separate held-out prompts /
3,200 answers. Startup dependency logs verified; actual queues/both GPUs must
still be checked. Fresh quota before launch: 18,645.51/108,000s used, reserved
0, next refresh October 24 00:00 UTC. Fresh draft OFF/capacity checks passed.
CPU slot is free. Next freed GPU can resume the audited selected-training
partials, never old full-training shards. No optimization/training was run.

At **05:19 Prague (03:19 UTC)**, validation-000 a1 is COMPLETE/audited with a
normal deadline partial: 1,355 / 3,200 answers, all GSM8K, 585 correct, 324
capped, 3,034,191 tokens. Generation-only 2,563.76s, startup 129.58s, stage
2,693.34s. Its 1,845 missing keys remain eligible for exact-key resume from
this audited parent; partial correctness is completion-biased. Revised scope
is **29,221 saved AND graded / 44,472**, zero awaiting grading, no duplicate or
seed errors. Test stays fully verified; validation and selected training are
still incomplete.

Validation-001 a1 has verified actual 1,600-request queues on each GPU,
approximately 1,040 + 1,080 completed at 03:17 UTC, both T4s 90-92% busy.
The freed slot now runs `thinkingcap-offline-subset-train-000-a4` v1, submitted
03:18:58 UTC, pinned `f45398d11641c6c17c3d806fcf2a732cd7e86572`, attaching
latest audited selected train-000 a3. It must skip its exact 1,101 saved keys
and collect only the remaining 2,099; actual startup/queues need checking.
Fresh quota: 19,857.23/108,000s used, reserved 0, refresh October 24 00:00 UTC;
fresh draft OFF and global capacity checks passed. CPU slot is free. This is
offline answer generation only, not training or tuning.

This branch separates **generation** from **grading**. It does not train an
adapter. PR #1's eager evaluation baseline and other experiment branches remain
unchanged. Collection uses the unadapted, pinned Qwen3-0.6B, not an overfit adapter.

## Frozen recipe and coverage

- Model: `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`.
- Thinking chat template; temperature **0.6**, top-p **0.95**, top-k **20**.
- Completion ceilings: **8,192 MBPP**, **4,096 all other datasets**. These are
  output-token limits, not prompt-plus-output limits. No 1,024-token RL cap or
  temperature-1 online-RL recipe is used here.
- Eager FP16, one model replica per T4, two replicas per session, continuous queue,
  64 active sequences/GPU, 2,048 prefill tokens/step, memory fraction 0.70.
- Eight independently seeded trajectories per prompt. Original global prompt
  indices and sample indices survive sharding and resumption; no re-seeding a
  remaining subset. Changing queue composition can still change sampled outputs;
  fixed seeds are not a promise of bit-identical generation across schedules.

| Partition | Prompts | Answers at 8/prompt |
|---|---:|---:|
| Benchmark test | 2,759 | 22,072 |
| Training | 46,812 | 374,496 |
| Validation (200/source) | 800 | 6,400 |
| Total | 50,371 | 402,968 |

The README's original preparation logic is retained, including Skywork's integer
answer/easy-band filter and validation split seed 0. Before validation splitting:
GSM8K 7,473; Skywork 28,147; CommonsenseQA 9,741; ARC Easy 2,251. Each Hugging Face
source revision and each prepared file's SHA-256 are recorded in the input plan.
The full benchmark cohort hash is
`16df64e67336b8a1cfa49298cd27e0e1f265927d12ad97f064f3968414d55a86`.

Train, validation and test stay in separate files/shards. Test outputs are for
evaluation/debugging, never automatically mixed into offline training. Exact
prepared-message overlap is audited; this is not a guarantee of semantic
decontamination across upstream datasets. Source datasets retain their licenses;
the prepared input and generated output are private on Kaggle.

## Generate once, grade later

From the repository root, on a suitable two-GPU environment:

```sh
python evaluate.py --out runs/raw400 --generate-only --samples 1 \
  --limit-per-benchmark 100 --backend vllm --dtype half \
  --data-parallel-size 2 --continuous-queue --max-num-seqs 64 \
  --max-num-batched-tokens 2048 --gpu-memory-utilization 0.70
python cloud/grade_saved.py --source runs/raw400 --out runs/graded400
```

`cloud/verify_offline.py` runs that 400-answer check, then grades the same saved
answers twice. It requires identical verdicts and unchanged raw-file hashes.
Generation calls the same queue and token-to-record conversion as evaluation;
grading does not load a language model or generate replacements.

Each raw directory contains:

- `prompts.jsonl`: full prompt/gold/grader/source metadata and exact prompt token IDs.
- `records.jsonl`: question/sample ID, stable seed, text, parsed final response,
  exact completion token IDs/counts, reasoning count, finish/stop reason,
  truncation and malformed-thinking flags. It is ungraded and immutable.
- `generation-manifest.json`: model, revision, sampling, caps, source/cohort
  provenance, expected/completed counts, artifact hashes, timings and explicit
  `completed`/`partial` status.
- `completed-keys.jsonl`: small coverage index for the new answers only.

Grading writes to another fresh directory, never overwrites raw trajectories,
and records its source hashes and `regenerated_answers: 0`. Test answers use the
original benchmark graders. Training/validation answers use the repository's
task-specific correctness checks. Truncated or malformed answers cannot earn
correctness. Grader infrastructure errors fail the grading audit rather than
silently becoming wrong answers. No token-length reward or pair selection is
introduced: later SimPO/SFT experiments can make that choice from these records.

## Full collection and resumption

Local preparation:

```sh
python cloud/prepare_offline_data.py --out data/offline-input
python cloud/plan_offline_collection.py --data data/offline-input
```

`data/offline-input/collection-plan.json` freezes 127 disjoint shards of up to
400 prompts each. The current frozen plan SHA-256 is
`eb589b2ff243e10377c7e7525ae28844e0d8b488166c6b01d36dd5ff84c69633`.
Input files are uploaded privately as `krcalkrcal/thinkingcap-offline-input`.
The source checkout, environments, weights, compiler cache and intermediate
per-GPU record copies stay in `/kaggle/temp`, not Kaggle's archived output.

After the 400-question verification passes, the explicit dispatcher builds a
pinned launcher and submits one shard:

```sh
.venv-kaggle/bin/python cloud/dispatch_offline.py --job test-000 \
  --commit FULL_PUBLISHED_COMMIT_SHA --submit
.venv-kaggle/bin/python cloud/refresh_offline.py
```

The dispatcher requires a passing verification artifact; it checks actual quota
and existing account sessions, allows at most two concurrent sessions (two pairs
of T4s), and records submission intent before network calls. It uses a full
commit ID, exact plan hash and unique notebook per attempt. A lock and ledger
prevent accidental duplicate submissions. An ambiguous submission is investigated,
not blindly retried. Defaults: 45-minute generation/startup budget and one-hour
outer safety timeout, leaving time for setup/grading/archival. Actual shard size
is preserved even if a session ends partially complete.

The shared queue saves each completed trajectory immediately. At the collection
deadline it aborts unfinished requests and seals an explicitly partial artifact,
then exits normally. Training/validation correctness checks take negligible CPU
time and run immediately. Benchmark grading is deferred to a **CPU-only Kaggle
job**, avoiding paid-in-quota GPU idle time during MBPP execution. The original evaluation path
has no such partial mode and still requires complete coverage.

After refresh reports a benchmark shard as `generated`, submit its saved-answer
CPU grader (at most one campaign CPU grader at a time):

```sh
.venv-kaggle/bin/python cloud/dispatch_offline_grade.py --job test-000 \
  --commit FULL_PUBLISHED_COMMIT_SHA
```

The CPU environment contains neither torch nor vLLM. It reads the attached raw
Kaggle output, verifies hashes and grades without a model download. A separate
400-answer CPU verification compares every verdict with the original GPU-session
grading. The ledger distinguishes **generated** from **graded**, so saved-but-not-
graded answers cannot masquerade as a finished dataset.

`collection/collection-index.json` carries cumulative completed question/sample
keys for that shard, plan hash, parent-index provenance and remaining coverage.
For a partial shard, the next unique attempt attaches the previous successful
notebook output and skips exactly those keys. Earlier raw answers stay in their
original immutable Kaggle output; only new answers are written on the next
attempt. Never delete those old versions. A fatal run with unsealed outputs must
be recovered explicitly; it is not automatically accepted as an attachable input.

The local ledger is `runs/offline-campaign/ledger.json`. Refresh downloads only
small manifests, summaries, provenance and per-answer `labels.jsonl`, not raw
trajectory corpora/environments/weights. Labels support local correctness, cap
rate and future preference-pool audits without downloading multi-GB text files.
Failed or unaudited attempts block automatic resubmission of that shard. Raw
generation is preserved even if subsequent grading needs repair.

The full 402,968-answer target is larger than the current free weekly quota.
Use legitimate quota resets and preserve the full target; do not silently
subsample, lower caps, buy compute, or evade account limits. Measure ETA from
actual per-source output lengths/throughput rather than assuming the 400-question
mix represents every training dataset. Monitoring is through the existing
three-minute in-chat automation; the local app/host must remain online for it.
Kaggle computation itself continues remotely.

## Validation status

Local regression tests cover the unchanged evaluation queue, raw artifact hashes,
seed/cap/token validation, separate grading without regeneration, partial queue
shutdown, resume identity, fixed private launchers and quota/concurrency guards.

The October 16 GPU verification **passed**:

- 400 unique saved answers, matching baseline question IDs/sample indices/seeds.
- Both saved-answer grading passes identical; zero regenerated answers or grader errors.
- Raw records SHA-256 `da56766b2cd659c33d3857447a73ce0ab591253c26bc435407a1b545b7d7c406`.
- Scores MBPP **50/100**, BFCL **75/100**, SVAMP **79/100**, IFEval **55/100**:
  259/400 versus the earlier 261/400 base model. Small differences are consistent
  with stochastic/runtime variation, not a demonstrated quality change. 312/400
  decoded completions exactly match the earlier run; fixed seeds do not make
  GPU sampling bit-identical across runtime conditions.
- 362,023 output tokens; 11 capped completions. All remain marked as truncated.
- Reported controller generation-only time **381.50s**; cold engine startup
  **139.63s**, generation stage **532.16s**, grading twice **51.11s**.
  This is not a speed benchmark against a warm engine.
- A separate tiny local M1 CPU smoke saved/read one 16-token completion correctly.

Initial collection sessions, runner `2c3e01acebaefe11838c9013319926b6ec175cc4`:

- [Test shard test-000, attempt 1](https://www.kaggle.com/code/krcalkrcal/thinkingcap-offline-test-000-a1)
- [Training shard train-000, attempt 1](https://www.kaggle.com/code/krcalkrcal/thinkingcap-offline-train-000-a1)
- [CPU-only 400-answer grading verification](https://www.kaggle.com/code/krcalkrcal/thinkingcap-offline-cpu-verification)

The CPU-only verification version 2 **passed**: all 400 individual verdicts
matched the original grades, 259 correct, 11 truncated, 25.89 seconds grading,
no torch/vLLM import and no GPU used. Version 1 caught a missing EvalPlus
dependency (`psutil`); that CPU environment was repaired before retrying on the
same saved answers. No trajectories were regenerated for either CPU attempt.
The current local regression suite passes **154 tests**.

Both collection jobs have passed engine health checks and started their queues,
1,600 requests per GPU (400 prompts × 8 / 2), with actual utilization on all four
T4s. First-launch generic `run-config.json` still labels its default wrapper
sample count as 1; the actual collector command, frozen plan and raw generation
manifest specify **8**. Later launchers correct that display metadata; it does
not change those running jobs' generation count or require a rerun.

### First collection checkpoint (October 16, approximately 21:03 UTC)

Both initial GPU sessions finished **COMPLETE**, sealing deadline-limited partial
outputs normally: test-000 saved **874/3,200** answers (1,676,066 tokens), and
train-000 saved **910/3,200** (2,322,581 tokens). These are **1,784/402,968** target
answers, not completed shards. Unfinished requests were aborted and not saved.
The test output is being graded by a separate CPU-only job; training labels were
graded inline with no duplicate keys detected by the ledger audit. Resume jobs
attach prior outputs and skip their exact completed keys, preserving seeds.

Training's first partial labels (not representative final dataset scores):

| Source | Saved answers | Correct | Token-cap hits | Output tokens |
| --- | ---: | ---: | ---: | ---: |
| Skywork | 538 | 22 | 396 | 1,929,430 |
| GSM8K | 137 | 54 | 38 | 287,836 |
| CommonsenseQA | 186 | 101 | 0 | 79,238 |
| ARC Easy | 49 | 43 | 0 | 26,077 |

All 434 capped training answers are marked truncated and incorrect. The high
Skywork cap rate deserves monitoring; caps and sampling remain unchanged.
Completion-at-deadline selection favors answers that finished earlier, so these
partial correctness, length and cap rates must not be treated as final rates.
Reported generation windows were 2,563.59s test and 2,538.02s training; saved
answers exclude work spent on aborted requests. Both GPUs per job were busy,
but KV-cache usage near 100% coincided with strongly fluctuating throughput.
At the initial aggregate rate, the full target would take roughly **170 paired-T4
session-hours** (about 85 wall-hours with two sessions); this is a very uncertain
extrapolation across heterogeneous sources, partial-answer bias and resume tails,
not a completion promise. The full target is retained across legitimate resets.

A draft-only unrelated notebook (`test-gpus`) caused the version-status API to
return 404. Its editor was checked and showed **Draft Session off**. Dispatch now
requires an explicit, fresh, exact-ref acknowledgement to exempt such a draft;
404 alone, a published version, or any other status error still blocks submission.
155 local regression tests pass.

The entire corpus is **not** complete. At initial collection submission,
Kaggle's live API reported **108,000s (30h)** weekly allowance, **10,624.69s** used,
zero reserved; refresh October 17 00:00 UTC. This supersedes its earlier 6h
allowance. The dispatcher always re-reads actual quota; it does not assume either
number remains fixed. No training runs are part of this collection request.
