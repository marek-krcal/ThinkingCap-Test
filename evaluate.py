"""Scores one checkpoint on four benchmarks and writes the result.

    python utils/eval/benchmarks.py                      # once, writes ./questions

There are three things you might want to score, and they differ only in which
flags you pass:

    # 1. the base model, unchanged -- score this once, it is what you compare to
    python evaluate.py --out runs/base

    # 2. a LoRA on top of that base -- the usual case
    python evaluate.py --out runs/mine --adapter ./my-lora

    # 3. your own weights, with or without an adapter of their own
    python evaluate.py --out runs/mine --model ./my-qwen3
    python evaluate.py --out runs/mine --model ./my-qwen3 --adapter ./my-lora

Add --smoke to any of them to use a 30-question subset instead, which checks
the plumbing in about a minute and reports no scores.

Case 1 pins the base model to an exact revision, so it is reproducible without
you knowing the commit. Cases 2 and 3 do not pin anything you supply: a local
path has no revision, and if you push your weights to the Hub their contents
can change under the same name. Note what you scored.

Seeds come from the cohort rather than from the run, so every checkpoint
answers the same questions with the same generations and the scores line up.

Three words appear throughout:

    selection  the frozen list of questions a run is scored on, named `full` or
               `smoke` and stored as questions.jsonl. Changing one word of one
               prompt makes a different selection, whose scores are not
               comparable with the old one -- which is why it is also hashed.
    question   one line of it: an id, which benchmark it came from, which
               grader scores it, and the messages to send the model.
    record     one generation and its verdict. Questions x --samples of them,
               stored as records.jsonl.
"""

import argparse
from collections import Counter
import json
import multiprocessing
import os
import queue
import signal
from pathlib import Path
import time

from utils.eval import sandbox
from utils.eval.benchmarks import paths, sha256
from utils.eval.graders import (grade_call, grade_numeric, load_bfcl, load_ifeval, load_sanitize,
                           mbpp_code, split_thinking)
from utils.eval.recipe import DEFAULT_MAX_TOKENS, MAX_TOKENS, SAMPLES, SAMPLING, SEED
from utils.eval.stats import confidence, summarize

BASE_MODEL = "Qwen/Qwen3-0.6B"
BASE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
BATCH = 256
RECORDS, RESULTS = "records.jsonl", "results.json"
# Room for the prompt on top of the generation budget. Far more than the 823
# tokens the longest question measures, and deliberately not trimmed to fit:
# vLLM's batching depends on this, so changing it changes what the model
# generates at a fixed seed -- 9 of 10 replies differed when I tried. It is
# part of the run's configuration, recorded in results.json, not a free dial.
PROMPT_BUDGET = 4096

# A benchmark is what you ask for, a dataset is a slice of the cohort, and a
# grader is the checker that scores it. BFCL is one benchmark over three
# datasets, so these cannot be the same name.
BENCHMARKS = {
    "mbpp": ("mbpp",),
    "bfcl": ("bfcl_simple_python", "bfcl_multiple", "bfcl_irrelevance"),
    "svamp": ("svamp",),
    "ifeval": ("ifeval",),
}


def main():
    arguments = parse_arguments()

    # root holds the questions and is read-only; out is this checkpoint's own
    # directory. Everything below this point can fail cheaply, and does so
    # before the model is loaded -- that alone takes about twenty seconds.
    root, out = Path(arguments.questions).resolve(), Path(arguments.out).resolve()
    suite = load_suite(root, arguments.smoke)
    listing = "smoke" if arguments.smoke else "questions"
    questions = select(read_jsonl(paths(root)[listing]), arguments.benchmarks)
    if arguments.limit_per_dataset:
        questions = limit_questions_per_dataset(questions, arguments.limit_per_dataset)
        suite["total"] = len(questions)
        suite["selection"] += f"-limit{arguments.limit_per_dataset}"
    if arguments.limit_per_benchmark:
        questions = limit_questions_per_benchmark(
            questions, arguments.benchmarks, arguments.limit_per_benchmark)
        suite["total"] = len(questions)
        suite["selection"] += f"-benchmark-limit{arguments.limit_per_benchmark}"
    prepare_output(out, arguments.regrade)

    # records are the answers: one generation per question per sample, graded.
    grading = grading_context(suite, arguments)
    records = reload_records(out) if arguments.regrade else generate(questions, arguments, out)
    grade_all(questions, records, grading)
    save_records(out, records)

    # results are per benchmark; failed names any whose grader could not run at
    # all, which is reported beside the others rather than thrown away.
    results, failed = collect(questions, records, arguments.samples)
    save_results(out, arguments, suite, grading, results, failed,
                 caps_in_use(questions, arguments))
    print(report(results, suite, failed))
    print(f"\nWrote {out / RESULTS}")


def parse_arguments():
    """The command line, with every default this run depends on filled in.

    Anything derived from a flag is derived here and nowhere else, so no two
    places can disagree about what was asked for.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default="questions",
                        help="Question directory built by utils/eval/benchmarks.py "
                             "(default: ./questions)")
    parser.add_argument("--smoke", action="store_true",
                        help="Use the small subset instead of the full benchmark, to check "
                             "the plumbing. Exercises everything and reports no scores.")
    parser.add_argument("--out", required=True, help="Fresh directory for this checkpoint")
    parser.add_argument("--model", default=BASE_MODEL,
                        help="Weights to score: leave it for the base model, or give a "
                             "local path or Hub id for your own full finetune")
    parser.add_argument("--adapter",
                        help="LoRA to apply on top of --model. Works with the base model "
                             "or with your own weights")
    parser.add_argument("--revision",
                        help="Pin --model to one revision. Filled in automatically for the "
                             "base model; has no effect on a local path, which has none")
    parser.add_argument("--benchmarks", default=list(BENCHMARKS), type=benchmark_list,
                        help=f"Comma-separated subset of: {', '.join(BENCHMARKS)}")
    parser.add_argument("--samples", type=int, default=SAMPLES,
                        help=f"Generations per question (default {SAMPLES})")
    parser.add_argument("--max-tokens", type=int,
                        help=f"One cap for every benchmark, overriding the per-benchmark "
                             f"defaults ({DEFAULT_MAX_TOKENS}, {MAX_TOKENS['mbpp']} for MBPP+). "
                             "Changing it changes what the model generates, so runs with "
                             "different caps are not comparable")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--backend", choices=("auto", "vllm", "transformers"), default="auto",
                        help="Generation engine. auto uses vLLM on CUDA and Transformers otherwise")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto",
                        help="Device for the Transformers backend")
    parser.add_argument("--dtype", choices=("auto", "half", "bfloat16", "float"), default="auto",
                        help="Weight dtype. auto selects a dtype supported by the device")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="GPUs used by one vLLM engine (default 1)")
    parser.add_argument("--data-parallel-size", type=int, default=1,
                        help="Independent single-GPU engines managed by this launcher")
    parser.add_argument("--batch-size", type=int, default=BATCH,
                        help="Number of jobs queued per vLLM generate call")
    parser.add_argument("--max-num-seqs", type=int, default=32,
                        help="Maximum concurrently active vLLM sequences")
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70,
                        help="vLLM memory fraction; leave headroom for sampling")
    parser.add_argument("--limit-per-dataset", type=int,
                        help="Development only: keep the first N questions from each dataset")
    parser.add_argument("--limit-per-benchmark", type=int,
                        help="Development only: keep N questions from each requested benchmark, "
                             "balanced across its component datasets")
    parser.add_argument("--sandbox", choices=sandbox.MODES, help="Default: bwrap if available")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                        help="Parallel MBPP+ grading processes")
    parser.add_argument("--regrade", action="store_true",
                        help=f"Re-grade a finished run's {RECORDS} without generating again")
    arguments = parser.parse_args()
    if arguments.tensor_parallel_size < 1 or arguments.batch_size < 1:
        parser.error("--tensor-parallel-size and --batch-size must be positive")
    if arguments.data_parallel_size < 1:
        parser.error("--data-parallel-size must be positive")
    if arguments.data_parallel_size > 1 and arguments.tensor_parallel_size != 1:
        parser.error("This offline data-parallel launcher requires tensor-parallel-size=1")
    if arguments.data_parallel_size > 1 and arguments.backend == "transformers":
        parser.error("Data parallelism currently requires the vLLM backend")
    if arguments.max_num_seqs < 1 or arguments.max_num_batched_tokens < 1:
        parser.error("vLLM scheduler limits must be positive")
    if not 0 < arguments.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be between 0 and 1")
    if arguments.limit_per_dataset is not None and arguments.limit_per_dataset < 1:
        parser.error("--limit-per-dataset must be positive")
    if arguments.limit_per_benchmark is not None and arguments.limit_per_benchmark < 1:
        parser.error("--limit-per-benchmark must be positive")
    if arguments.limit_per_dataset is not None and arguments.limit_per_benchmark is not None:
        parser.error("Choose only one of --limit-per-dataset and --limit-per-benchmark")
    # Pin the base model unless the user named their own: a default run should
    # be reproducible without anyone having to know the commit. Their weights
    # get no revision, because a local path has none.
    if arguments.revision is None and arguments.model == BASE_MODEL:
        arguments.revision = BASE_REVISION
    return arguments


def load_suite(root, smoke=False):
    """The manifest, verified against the questions, with grader paths filled in.

    Checking the hash here is the point: the manifest records what was built,
    and without comparing it nothing notices a question that has been edited
    since. One such edit went unseen for a week because every check was a
    regrade, and regrading reuses the file rather than rebuilding it.

    Paths are resolved rather than stored, so the directory can be copied.
    """
    at = paths(root)
    suite = json.loads(at["manifest"].read_text())
    listing = "smoke" if smoke else "questions"
    recorded = suite["smoke_sha256"] if smoke else suite["cohort_sha256"]
    actual = sha256(at[listing])
    if actual != recorded:
        raise SystemExit(
            f"{at[listing]} does not match the manifest beside it:\n"
            f"  manifest says {recorded[:12]}\n"
            f"  file is       {actual[:12]}\n"
            "Rebuild with utils/eval/benchmarks.py, or restore the questions it was built from.")
    suite.update({name: str(at[name]) for name in ("evalplus", "bfcl", "ifeval", "mbpp")})
    suite["selection"] = "smoke" if smoke else "full"
    suite["smoke"] = smoke
    suite["cohort_sha256"] = actual
    suite["total"] = suite["smoke_total"] if smoke else suite["total"]
    return suite


def benchmark_list(text):
    """--benchmarks is parsed once, here, so nothing downstream re-derives it."""
    names = [name.strip() for name in text.split(",") if name.strip()]
    unknown = sorted(set(names) - set(BENCHMARKS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown benchmark(s): {unknown}; choose from {sorted(BENCHMARKS)}")
    return names


def select(questions, benchmarks):
    """The questions belonging to the named benchmarks, in their stored order."""
    unknown = sorted(set(benchmarks) - set(BENCHMARKS))
    if unknown:
        raise ValueError(f"Unknown benchmark(s): {unknown}; choose from {sorted(BENCHMARKS)}")
    wanted = {dataset for name in benchmarks for dataset in BENCHMARKS[name]}
    chosen = [q for q in questions if q["dataset"] in wanted]
    if not chosen:
        raise SystemExit(f"No questions for {benchmarks} in this cohort")
    return chosen


def limit_questions_per_dataset(questions, limit):
    """Keep a deterministic prefix of every selected dataset for timing smoke tests."""
    counts, chosen = Counter(), []
    for question in questions:
        dataset = question["dataset"]
        if counts[dataset] < limit:
            chosen.append(question)
            counts[dataset] += 1
    return chosen


def limit_questions_per_benchmark(questions, benchmarks, limit):
    """Keep a deterministic, balanced prefix for each requested benchmark."""
    selected = []
    for benchmark in benchmarks:
        datasets = BENCHMARKS[benchmark]
        pools = {dataset: [q for q in questions if q["dataset"] == dataset]
                 for dataset in datasets}
        positions = Counter()
        while sum(positions.values()) < limit:
            added = False
            for dataset in datasets:
                if sum(positions.values()) >= limit:
                    break
                if positions[dataset] < len(pools[dataset]):
                    selected.append(pools[dataset][positions[dataset]])
                    positions[dataset] += 1
                    added = True
            if not added:
                break
    return selected


def grading_context(suite, arguments):
    """Everything grading needs: the suite, how to run code, and the checkers.

    Loading a checker is not free, so only the ones this run asked for are built.
    """
    if "mbpp" in arguments.benchmarks:
        mode, why = sandbox.choose_mode(arguments.sandbox)
    else:
        mode, why = "unused", "no code execution"
    return {"suite": suite, "mode": mode, "why": why, "workers": arguments.workers,
            "checkers": load_checkers(suite, arguments.benchmarks)}


def reload_records(out):
    """Read a finished run's generations back, discarding their old verdicts.

    Clearing them matters: a row that fails to grade this time would otherwise
    keep the answer from last time and look like it had been graded.
    """
    records = read_jsonl(out / RECORDS)
    for record in records:
        record.pop("correct", None)
        record.pop("grading_error", None)
    print(f"Regrading {len(records)} stored generations")
    return records


def save_records(out, records):
    """Rewrite records.jsonl now the rows carry their grades.

    generate() already streamed these, so a crash keeps its generations; this
    replaces that file with the graded version.
    """
    (out / RECORDS).write_text("".join(json.dumps(r) + "\n" for r in records))


def prepare_output(out, regrade):
    if regrade:
        if not (out / RECORDS).exists():
            raise SystemExit(f"No {RECORDS} to regrade in {out}")
        return
    # A crashed run leaves an empty directory behind, and refusing to reuse that
    # would make every retry a manual cleanup. Refuse only finished results.
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)


def load_checkers(suite, wanted):
    checkers = {}
    if "mbpp" in wanted:
        checkers["mbpp"] = load_sanitize(suite["evalplus"])
    if "bfcl" in wanted:
        checkers["bfcl"] = load_bfcl(suite["bfcl"])
    if "ifeval" in wanted:
        checkers["ifeval"] = load_ifeval(suite["ifeval"])
    return checkers


def generate(questions, arguments, out):
    """Generate every question `--samples` times with the selected backend."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(arguments.model, revision=arguments.revision)
    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    open_id, close_id = (tokenizer.convert_tokens_to_ids(tag) for tag in ("<think>", "</think>"))
    # Tokenize before loading the model: it is quick, and a question too long to
    # ask should fail now rather than after twenty seconds of model load.
    jobs = build_jobs(questions, tokenizer, arguments)
    longest = max(len(job["ids"]) for job in jobs)
    if longest > PROMPT_BUDGET:
        raise SystemExit(f"A question is {longest} tokens, over the {PROMPT_BUDGET}-token "
                         "prompt budget; it would be truncated and score as a failure")
    backend = resolve_backend(arguments)
    if backend == "transformers":
        return generate_transformers(jobs, arguments, out, tokenizer,
                                     eos_ids, open_id, close_id)
    if arguments.data_parallel_size > 1:
        return generate_vllm_data_parallel(jobs, arguments, out,
                                          max(caps_in_use(questions, arguments).values()))
    return generate_vllm(jobs, arguments, out, tokenizer, eos_ids, open_id, close_id,
                         max(caps_in_use(questions, arguments).values()))


def resolve_backend(arguments):
    if arguments.backend != "auto":
        return arguments.backend
    import torch
    return "vllm" if torch.cuda.is_available() else "transformers"


def finish_timing(arguments, started, records, backend, device):
    elapsed = time.perf_counter() - started
    tokens = sum(record["output_tokens"] for record in records)
    arguments.generation_stats = {
        "backend": backend, "device": device, "elapsed_seconds": elapsed,
        "engine_startup_seconds": getattr(arguments, "startup_seconds", None),
        "responses": len(records), "output_tokens": tokens,
        "tokens_per_second": tokens / elapsed if elapsed else None,
        "seconds_per_response": elapsed / len(records) if records else None,
    }
    print("Generation timing: " + json.dumps(arguments.generation_stats), flush=True)


def shard_jobs(jobs, count):
    """Distribute original jobs without changing their identifiers or seeds."""
    return [jobs[index::count] for index in range(count)]


def merge_shard_records(jobs, records):
    order = {(job["row"]["id"], job["sample"]): position
             for position, job in enumerate(jobs)}
    keys = [(record["id"], record["sample_index"]) for record in records]
    if len(keys) != len(jobs) or set(keys) != set(order):
        raise RuntimeError("Data-parallel merge has missing or duplicate generations")
    return sorted(records, key=lambda record: order[(record["id"], record["sample_index"])])


def data_parallel_worker(index, jobs, arguments, out, longest_generation, notices):
    """One isolated CUDA engine; spawned fresh, never forks a CUDA context."""
    os.setsid()  # lets the controller clean up this engine and its descendants
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(arguments.model, revision=arguments.revision)
        eos = tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        open_id, close_id = (tokenizer.convert_tokens_to_ids(tag)
                             for tag in ("<think>", "</think>"))
        out.mkdir(parents=True, exist_ok=True)
        print(f"Data-parallel worker {index}: {len(jobs)} jobs on GPU {index}", flush=True)
        generate_vllm(jobs, arguments, out, tokenizer, eos_ids, open_id, close_id,
                      longest_generation)
        notices.put((index, arguments.generation_stats, None))
    except BaseException:
        import traceback
        notices.put((index, None, traceback.format_exc()))
        raise


def generate_vllm_data_parallel(jobs, arguments, out, longest_generation):
    """One controller, independent GPU workers, one merged evaluation output."""
    import torch
    count = arguments.data_parallel_size
    if torch.cuda.device_count() < count:
        raise RuntimeError("Not enough CUDA GPUs for requested data parallel size")
    if count > len(jobs):
        raise ValueError("Data parallel size exceeds number of generation jobs")
    started = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    notices = context.Queue()
    workers, completed = [], {}
    shards = out / "shards"
    def terminate_controller(signum, frame):
        raise SystemExit("Data-parallel controller stopped")
    previous_handler = signal.signal(signal.SIGTERM, terminate_controller)
    try:
        for index, subset in enumerate(shard_jobs(jobs, count)):
            worker = context.Process(target=data_parallel_worker,
                args=(index, subset, arguments, shards / str(index), longest_generation, notices))
            worker.start()
            workers.append(worker)
        while len(completed) < count:
            try:
                index, stats, error = notices.get(timeout=10)
            except queue.Empty:
                for index, worker in enumerate(workers):
                    if index not in completed and worker.exitcode is not None:
                        raise RuntimeError(f"Data-parallel worker {index} exited before reporting: "
                                           f"{worker.exitcode}")
                continue
            if error:
                raise RuntimeError(f"Data-parallel worker {index} failed:\n{error}")
            completed[index] = stats
            print(f"Data-parallel worker {index} finished: {stats['responses']} responses", flush=True)
        records = [record for index in range(count)
                   for record in read_jsonl(shards / str(index) / RECORDS)]
        records = merge_shard_records(jobs, records)
        save_records(out, records)
        arguments.startup_seconds = max(stats['engine_startup_seconds'] for stats in completed.values())
        finish_timing(arguments, started, records, "vllm", "cuda")
        arguments.generation_stats['workers'] = completed
        return records
    finally:
        for worker in workers:
            worker.join(timeout=2)
            # The controller process may have exited while its vLLM engine
            # descendants are still alive: clean its group even in that case.
            try:
                os.killpg(worker.pid, signal.SIGTERM)
            except ProcessLookupError:
                if worker.is_alive():
                    worker.terminate()
            worker.join(timeout=5)
            if worker.is_alive():
                try:
                    os.killpg(worker.pid, signal.SIGKILL)
                except ProcessLookupError:
                    worker.kill()
                worker.join(timeout=5)
        signal.signal(signal.SIGTERM, previous_handler)


def generate_vllm(jobs, arguments, out, tokenizer, eos_ids, open_id, close_id,
                  longest_generation):
    """Generate through the fast CUDA-only vLLM engine."""
    from vllm import SamplingParams

    started = time.perf_counter()
    # The engine has to hold the largest budget any benchmark in this run uses.
    model, lora = load_engine(arguments, longest_generation)
    print('Engine health check: generating 16 tokens before the evaluation queue', flush=True)
    def health_timeout(signum, frame):
        raise TimeoutError('16-token engine health check exceeded 120 seconds')
    previous_handler = signal.signal(signal.SIGALRM, health_timeout)
    signal.alarm(120)
    try:
        model.generate([{"prompt_token_ids": jobs[0]["ids"]}],
                       SamplingParams(max_tokens=16, temperature=0, seed=arguments.seed),
                       lora_request=lora, use_tqdm=False)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
    print('Engine health check passed', flush=True)
    arguments.startup_seconds = time.perf_counter() - started
    print(f"vLLM engine startup seconds: {arguments.startup_seconds:.3f}", flush=True)

    records = []
    with (out / RECORDS).open("w") as stream:
        batch_size = arguments.batch_size
        for start in range(0, len(jobs), batch_size):
            batch = jobs[start:start + batch_size]
            settings = [SamplingParams(max_tokens=token_cap(job["row"]["dataset"], arguments),
                                       seed=job["seed"], stop_token_ids=list(eos_ids),
                                       **SAMPLING) for job in batch]
            outputs = model.generate([{"prompt_token_ids": job["ids"]} for job in batch],
                                     settings, lora_request=lora)
            if len(outputs) != len(batch):
                raise ValueError("vLLM returned a different number of outputs")
            for job, output in zip(batch, outputs):
                record = to_record(job, output, lora, tokenizer, eos_ids, open_id, close_id)
                records.append(record)
                stream.write(json.dumps(record) + "\n")
            stream.flush()  # a crash late in a long run keeps what it earned
            print(f"{start + len(batch)}/{len(jobs)}", flush=True)
    finish_timing(arguments, started, records, "vllm", "cuda")
    return records


def generate_transformers(jobs, arguments, out, tokenizer, eos_ids, open_id, close_id):
    """Portable one-response-at-a-time generation for CPU and Apple Silicon."""
    import torch
    from transformers import AutoModelForCausalLM

    device = arguments.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    dtype_name = arguments.dtype
    if dtype_name == "auto":
        if device == "cuda":
            dtype_name = "bfloat16" if torch.cuda.get_device_capability()[0] >= 8 else "half"
        else:
            dtype_name = "float"
    dtype = {"half": torch.float16, "bfloat16": torch.bfloat16,
             "float": torch.float32}[dtype_name]
    print(f"Transformers backend: device={device}, dtype={dtype_name}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        arguments.model, revision=arguments.revision, dtype=dtype).to(device)
    if arguments.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, arguments.adapter)
    model.eval()

    records, started = [], time.perf_counter()
    with (out / RECORDS).open("w") as stream, torch.inference_mode():
        for index, job in enumerate(jobs, 1):
            torch.manual_seed(job["seed"])
            input_ids = torch.tensor([job["ids"]], device=device)
            attention_mask = torch.ones_like(input_ids)
            output = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=token_cap(job["row"]["dataset"], arguments),
                do_sample=True, temperature=SAMPLING["temperature"],
                top_p=SAMPLING["top_p"], top_k=SAMPLING["top_k"],
                pad_token_id=tokenizer.eos_token_id,
            )
            tokens = output[0, input_ids.shape[1]:].tolist()
            record = record_from_tokens(job, tokens, tokenizer, eos_ids, open_id, close_id)
            records.append(record)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            print(f"{index}/{len(jobs)}  {len(tokens)} tokens", flush=True)
    finish_timing(arguments, started, records, "transformers", device)
    return records


def load_engine(arguments, longest_generation):
    """Build the vLLM engine, registering the adapter when there is one."""
    # FlashInfer's sampler compiles a kernel on the fly and needs nvcc, which
    # driver-only machines do not have. The PyTorch sampler needs no toolchain.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    rank = None
    if arguments.adapter:
        rank = json.loads((Path(arguments.adapter) / "adapter_config.json").read_text())["r"]
    dtype = resolve_vllm_dtype(arguments.dtype)
    print(f"vLLM dtype: {dtype}", flush=True)
    import torch
    if torch.cuda.device_count() < arguments.tensor_parallel_size:
        raise RuntimeError("Not enough CUDA GPUs for requested tensor parallel size")
    model = LLM(model=arguments.model, revision=arguments.revision,
                tokenizer_revision=arguments.revision,
                dtype=dtype,
                tensor_parallel_size=arguments.tensor_parallel_size,
                max_model_len=longest_generation + PROMPT_BUDGET,
                gpu_memory_utilization=getattr(arguments, 'gpu_memory_utilization', 0.70),
                max_num_seqs=getattr(arguments, 'max_num_seqs', 32),
                max_num_batched_tokens=getattr(arguments, 'max_num_batched_tokens', 2048),
                enable_chunked_prefill=True,
                enforce_eager=True,
                disable_custom_all_reduce=True,
                disable_log_stats=False,
                enable_prefix_caching=False, generation_config="vllm", seed=arguments.seed,
                enable_lora=rank is not None, max_lora_rank=rank or 16)
    if not arguments.adapter:
        return model, None
    lora = LoRARequest("candidate", 1, str(Path(arguments.adapter).resolve()))
    model.llm_engine.add_lora(lora)
    if 1 not in model.llm_engine.list_loras():
        raise ValueError("Adapter did not register")
    return model, lora


def resolve_vllm_dtype(requested):
    """Use fp16 on pre-Ampere GPUs whose hardware does not support bf16."""
    if requested != "auto":
        return requested
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("vLLM evaluation requires a CUDA GPU")
    major, _ = torch.cuda.get_device_capability()
    return "bfloat16" if major >= 8 else "half"


def token_cap(dataset, arguments):
    """The generation budget for one dataset: the flag if given, else its default."""
    if arguments.max_tokens:
        return arguments.max_tokens
    return MAX_TOKENS.get(dataset, DEFAULT_MAX_TOKENS)


def caps_in_use(questions, arguments):
    """{dataset: cap} for the datasets this run actually touches."""
    return {d: token_cap(d, arguments) for d in sorted({q["dataset"] for q in questions})}


def build_jobs(questions, tokenizer, arguments):
    """One job per (question, sample), seeded from the cohort so runs can be paired."""
    jobs = []
    for row in questions:
        ids = tokenizer.apply_chat_template(row["messages"], tokenize=True,
                                            add_generation_prompt=True, enable_thinking=True)
        for sample in range(arguments.samples):
            jobs.append({"row": row, "ids": ids, "sample": sample,
                         "seed": arguments.seed + row["seed_index"] * 1000 + sample})
    return jobs


def to_record(job, output, lora, tokenizer, eos_ids, open_id, close_id):
    used = output.lora_request
    wrong_route = (used is not None) if lora is None else (used is None or used.lora_int_id != 1)
    if wrong_route:
        raise ValueError("Adapter routing mismatch")
    return record_from_tokens(job, list(output.outputs[0].token_ids), tokenizer,
                              eos_ids, open_id, close_id)


def record_from_tokens(job, tokens, tokenizer, eos_ids, open_id, close_id):
    final, reasoning, boundary = split_thinking(tokens, open_id, close_id, eos_ids)
    return {"id": job["row"]["id"], "dataset": job["row"]["dataset"],
            "sample_index": job["sample"], "seed": job["seed"],
            "output_tokens": len(tokens), "reasoning_tokens": reasoning,
            "truncated": not (tokens and tokens[-1] in eos_ids), "boundary_error": boundary,
            "text": tokenizer.decode(tokens, skip_special_tokens=True),
            "response": tokenizer.decode(final, skip_special_tokens=True)}


def grade_all(questions, records, grading):
    """Grade MBPP+ in one batched pass, then everything else row by row."""
    by_id = {q["id"]: q for q in questions}
    programs = [r for r in records if by_id[r["id"]]["grader"] == "mbpp"]
    if programs:
        print(f"Grading {len(programs)} MBPP+ programs on "
              f"{grading['workers']} workers", flush=True)
        grade_programs(by_id, programs, grading)
    for record in records:
        row = by_id[record["id"]]
        if row["grader"] == "mbpp":
            continue
        try:
            correct, error = grade_one(row, record, grading["checkers"])
            record.update(correct=correct, parse_error=error)
        except Exception as error:  # a broken grader must not silently drop a row
            record.update(correct=None, parse_error=None, grading_error=str(error))


def grade_one(row, record, checkers):
    grader = row["grader"]
    if grader == "numeric":
        return grade_numeric(record["text"], row["answer"], record["truncated"])
    if grader == "bfcl":
        return grade_call(row, record["response"], *checkers["bfcl"], truncated=record["truncated"])
    if grader == "ifeval":
        if record["truncated"]:
            return False, "truncated"
        return checkers["ifeval"](row, record["response"])
    raise ValueError(f"Unknown grader: {grader!r}")


def grade_programs(by_id, records, grading):
    """Execution dominates MBPP+, so it runs as one parallel batch, not row by row."""
    items, parse_errors = [], {}
    for index, record in enumerate(records):
        row = by_id[record["id"]]
        code, error = mbpp_code(record["response"], row["entry_point"], grading["checkers"]["mbpp"],
                                record["truncated"])
        if error:
            parse_errors[index] = error
        else:
            items.append((index, row["id"], code))
    suite = grading["suite"]
    verdicts = sandbox.grade_programs(grading["mode"], suite["evalplus"], suite["mbpp"],
                                      items, grading["workers"])
    for index, record in enumerate(records):
        if index in parse_errors:
            record.update(correct=False, parse_error=parse_errors[index])
        elif verdicts.get(index) is None or verdicts[index].get("error"):
            record.update(correct=None, parse_error=None,
                          grading_error=(verdicts.get(index) or {}).get("error", "no verdict"))
        else:
            verdict = verdicts[index]
            record.update(correct=verdict["base"] == "pass" and verdict["plus"] == "pass",
                          parse_error=None)


def collect(questions, records, samples):
    """-> ({dataset: summary}, {dataset: why it could not be graded}).

    Both are needed: the questions say which datasets to report, the records
    hold what happened. A dataset with no gradeable record is reported as failed rather
    than quietly missing from the table.
    """
    results, failed = {}, {}
    for dataset in sorted({q["dataset"] for q in questions}):
        subset = [r for r in records if r["dataset"] == dataset]
        errors = Counter(r["grading_error"] for r in subset if r.get("grading_error"))
        if any(r["correct"] is None for r in subset):
            # Ungraded rows are not a random subset (crashes and timeouts hit the
            # heavy programs), so a score over the rest would be biased. One
            # benchmark's grader failing must not discard the others.
            failed[dataset] = {"rows": len(subset), "errors": dict(errors.most_common(3))}
            continue
        results[dataset] = dict(summarize(subset), samples=samples,
                                confidence=confidence(subset), grading_errors=dict(errors))
    if not results:
        raise SystemExit("Every benchmark failed to grade: " + json.dumps(failed, indent=2))
    return results, failed


def save_results(out, arguments, suite, grading, results, failed, caps):
    """Write results.json, which has to carry enough to interpret the scores later."""
    (out / RESULTS).write_text(json.dumps({
        "model": arguments.model, "revision": arguments.revision,
        "adapter": str(Path(arguments.adapter).resolve()) if arguments.adapter else None,
        "benchmarks": arguments.benchmarks,
        "samples": arguments.samples, "seed": arguments.seed, "sampling": SAMPLING,
        "backend": arguments.backend, "device": arguments.device, "dtype": arguments.dtype,
        "tensor_parallel_size": getattr(arguments, "tensor_parallel_size", 1),
        "data_parallel_size": getattr(arguments, "data_parallel_size", 1),
        "batch_size": getattr(arguments, "batch_size", BATCH),
        "max_num_seqs": getattr(arguments, "max_num_seqs", 32),
        "max_num_batched_tokens": getattr(arguments, "max_num_batched_tokens", 2048),
        "gpu_memory_utilization": getattr(arguments, "gpu_memory_utilization", 0.70),
        "enforce_eager": True,
        "generation": getattr(arguments, "generation_stats", None),
        "max_tokens": caps, "max_model_len": max(caps.values()) + PROMPT_BUDGET,
        "sandbox": {"mode": grading["mode"], "why": grading["why"]},
        "revisions": suite["revisions"], "cohort_sha256": suite["cohort_sha256"],
        "cohort_total": suite["total"], "smoke": suite.get("smoke", False),
        "selection": suite.get("selection", "full"),
        "results": results, "failed_benchmarks": failed,
    }, indent=2) + "\n")


def report(results, suite, failed=None):
    """One table of scores, each with the interval a leaderboard row needs."""
    if suite.get("smoke"):
        # Refusing to print scores is the point: a number from a handful of
        # questions would be indistinguishable on a leaderboard from a real one.
        # A benchmark that failed to grade must still be named: a smoke run exists to
        # find exactly that before a full run spends hours generating.
        graded = sum(r["n"] for r in results.values())
        lines = [f"Smoke run on {suite['total']} questions: generation completed, "
                 f"{graded} rows graded."]
        for name, detail in sorted((failed or {}).items()):
            lines.append(f"\n{name}: NOT GRADED, all {detail['rows']} rows failed")
            for message, count in detail["errors"].items():
                lines.append(f"  {count}x {str(message).splitlines()[0][:140]}")
        if failed:
            lines.append("\nFix the above before a full run; this benchmark would score "
                         "nothing there either.")
        lines.append("No scores are reported. Drop --smoke to evaluate.")
        return "\n".join(lines)
    lines = [f"questions: {suite.get('selection', 'full')} "
             f"({suite['cohort_sha256'][:8]})  {suite['total']} prompts",
             f"{'benchmark':<22}{'accuracy':>22}{'mean tokens':>20}"]
    for name, result in sorted(results.items()):
        lines.append(f"{name:<22}{with_error(result, 'accuracy'):>22}"
                     f"{with_error(result, 'mean_output_tokens'):>20}")
    lines.append("\nIntervals are 95%, resampling whole prompts. Two checkpoints whose "
                 "intervals overlap\nare not separated by this run.")
    for name, result in sorted(results.items()):
        notes = warnings(result)
        if notes:
            lines.append(f"\n{name}: " + "; ".join(notes))
    for name, detail in (failed or {}).items():
        lines.append(f"\n{name}: NOT GRADED, all {detail['rows']} rows failed")
        for message, count in detail["errors"].items():
            lines.append(f"  {count}x {str(message).splitlines()[0][:140]}")
    return "\n".join(lines)


def with_error(result, field):
    value = result[field]
    text = f"{value:.2%}" if field == "accuracy" else f"{value:.1f}"
    bounds = result["confidence"][field]
    if not bounds:
        return text
    half = (bounds[1] - bounds[0]) / 2
    return text + (f" +/-{half * 100:.2f}" if field == "accuracy" else f" +/-{half:.1f}")


def warnings(result):
    notes = []
    if result["truncation_rate"]:
        notes.append(f"{result['truncation_rate']:.2%} truncated "
                     "(raise --max-tokens; capped answers score as wrong)")
    if result["parse_failure_rate"]:
        notes.append(f"{result['parse_failure_rate']:.2%} unreadable")
    if result["ungraded"]:
        notes.append(f"{result['ungraded']} rows ungraded (grader errors)")
    return notes


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


if __name__ == "__main__":
    main()
