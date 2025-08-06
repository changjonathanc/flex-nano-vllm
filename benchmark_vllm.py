# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "vllm",
#     "transformers",
#     "datasets",
#     "matplotlib",
#     "tqdm",
#     "pandas",
# ]
# ///

# Usage:
# Run benchmark with minimal overhead: python bench_utils.py
# Run benchmark with profiling: ENABLE_PROFILING=1 python bench_utils.py
# NOTE: this script serves 2 purposes:
# 1. it can be used to benchmark vLLM's performance, run with isolated inline dependencies.
# 2. outside of __main__, it contains utils for producing the same payload for benchmarking.

from tqdm import tqdm
from datasets import load_dataset
import random
from dataclasses import dataclass

@dataclass
class BenchmarkRequest:
    """Simple, framework-agnostic request data"""
    text: str
    max_new_tokens: int

# Standard prompts used in benchmarks
long_prompt = """
The 12 months of the year are: January, February, March,
""".strip()

short_prompt = "The first 20 prime numbers are: 2, 3,"


def generate_benchmark_data(tokenizer, n_requests=512, max_input_length=512, min_tokens=128, max_tokens=512):
    """Generate benchmark data by skipping prompts that are too long."""
    from datasets import load_dataset
    
    data = load_dataset("Open-Orca/OpenOrca")["train"]
    benchmark_requests = []
    
    attempts = 0
    while len(benchmark_requests) < n_requests:
        # Deterministic sampling using hash
        idx = hash(f"req_{attempts}") % len(data)
        
        system_prompt = data[idx]["system_prompt"] or ""
        question = data[idx]["question"] or ""
        prompt = f"{idx}: {system_prompt} {question}".strip()
        
        # Check length and skip if too long
        tokens = tokenizer.encode(prompt)
        if len(tokens) <= max_input_length:
            prompt_hash = hash(prompt)
            max_new_tokens = min_tokens + (abs(prompt_hash >> 16) % (max_tokens - min_tokens + 1))
            benchmark_requests.append(BenchmarkRequest(text=prompt, max_new_tokens=max_new_tokens))
        
        attempts += 1
        if attempts > n_requests * 10:  # Safety valve to prevent infinite loop
            break
    
    # Add standard prompts
    benchmark_requests.extend([
        BenchmarkRequest(text=long_prompt, max_new_tokens=max_tokens),
        BenchmarkRequest(text=short_prompt, max_new_tokens=max_tokens)
    ])
    
    return benchmark_requests


def print_step_stats(steps, name):
    """Helper to print timing statistics for a collection of steps."""
    if not steps:
        return
    print(f"\n{name}:")
    print(f"  Count: {len(steps)}")
    print(f"  Total: {sum(steps):.4f}s")
    print(f"  Mean:  {sum(steps)/len(steps):.4f}s")
    print(f"  Min:   {min(steps):.4f}s")
    print(f"  Max:   {max(steps):.4f}s")


def generate_with_timing(llm, sequences, sampling_params, collect_detailed_metrics=False):
    """
    Generate with timing, optionally collecting detailed metrics.
    
    Note: We track total step time rather than trying to separate prefill/decode
    because vLLM can do both types of work within a single step, making such
    separation misleading for performance analysis.
    """
    outputs = []
    total_step_time = 0.0
    step_times = []
    
    # Optional detailed metrics
    metrics_data = {} if not collect_detailed_metrics else {
        'steps': [], 'requests_running': [], 'requests_waiting': [], 'preemptions': []
    }
    
    # Add requests
    for i, prompt in enumerate(sequences):
        sp = sampling_params[i] if isinstance(sampling_params, list) else sampling_params
        llm.llm_engine.add_request(str(i), prompt, sp)
    
    step_count = 0
    
    while llm.llm_engine.has_unfinished_requests():
        step_start = time.perf_counter()
        step_outputs = llm.llm_engine.step()
        step_duration = time.perf_counter() - step_start
        step_count += 1
        
        total_step_time += step_duration
        step_times.append(step_duration)
        
        # Collect outputs
        for output in step_outputs:
            if output.finished:
                outputs.append(output)
                
        # Optional detailed metrics
        if collect_detailed_metrics:
            metrics = llm.llm_engine.get_metrics()
            metrics_data['steps'].append(step_count)
            # Extract key metrics
            running = waiting = preemptions = 0
            for metric in metrics:
                if "requests_running" in metric.name:
                    running = metric.value
                elif "requests_waiting" in metric.name:
                    waiting = metric.value
                elif "preemptions" in metric.name:
                    preemptions = metric.value
            metrics_data['requests_running'].append(running)
            metrics_data['requests_waiting'].append(waiting)
            metrics_data['preemptions'].append(preemptions)
    
    return total_step_time, step_times, outputs, metrics_data


if __name__ == "__main__":
    import os
    import time
    
    os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"
    os.environ['VLLM_USE_V1'] = '1'
    
    ENABLE_PROFILING = os.environ.get("ENABLE_PROFILING", "0") == "1"
    
    from vllm import LLM, SamplingParams as VLLMSamplingParams
    from transformers import AutoTokenizer

    MODEL_ID = "google/gemma-2-2b"
    llm = LLM(MODEL_ID, dtype="bfloat16", gpu_memory_utilization=0.9, max_num_seqs=256, max_model_len=2048, disable_log_stats=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    n_requests = 512
    max_input_length = 512

    benchmark_requests = generate_benchmark_data(tokenizer, n_requests, max_input_length)
    
    sequences = [req.text for req in benchmark_requests]
    sampling_params_list = [
        VLLMSamplingParams(temperature=0.0, top_p=1.0, max_tokens=req.max_new_tokens)
        for req in benchmark_requests
    ]
    
    # Warmup
    llm.generate(["warmup"], VLLMSamplingParams(max_tokens=1), use_tqdm=False)



    print(f"\n--- RUNNING {'WITH' if ENABLE_PROFILING else 'WITHOUT'} DETAILED METRICS ---")
    
    # Reset memory stats to track benchmark-specific usage  
    start_time = time.time()
    total_step_time, step_times, outputs, metrics_data = generate_with_timing(
        llm, sequences, sampling_params_list, collect_detailed_metrics=ENABLE_PROFILING
    )
    total_time = time.time() - start_time

    total_output_length = sum(len(o.outputs[0].token_ids) for o in outputs)
    prompt_tok = sum(len(o.prompt_token_ids) for o in outputs)
    
    print_step_stats(step_times, "\nstep")
    
    # Get memory usage
    print("\n--- PERFORMANCE METRICS ---")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {total_output_length / total_time:.1f} tokens/s")
    print(f"Request throughput: {len(sequences) / total_time:.2f} req/s")
    print(f"Total throughput (prompt+new): {(prompt_tok + total_output_length) / total_time:.1f} tokens/s")
        
    if ENABLE_PROFILING and metrics_data:
        print("\n--- DETAILED METRICS ---")
        import pandas as pd
        pd.DataFrame(metrics_data).to_csv('vllm_metrics.csv', index=False)
        print("Metrics saved to 'vllm_metrics.csv'")

