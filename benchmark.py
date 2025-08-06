"""
This script is used to test the correctness and benchmark the paged attention implementation.


simplified interface of Inference class:
Usage:
llm = Inference(...)
sequences = [Sequence(text) for text in texts]
llm.generate(sequences, sampling_params=SamplingParams(max_new_tokens=max_new_tokens), use_tqdm=True, capture_cudagraph=True)
outputs = debug_print_outputs(sequences, tokenizer)




## correctness:
1. run with different 2 sequences with different lengths, the output should match huggingface .generate()
        this output has been verified to be correct outside of this script
2. run the same sequence with .generate() and paged attention, the output should match
        this is to test the paged attention Inference class does not have side effects that can alter the output across .generate() calls

### correctness with dynamic batching
3. run with different number of requests, and the same 2 sequences are mixed in the batch
        the output should match 1.

### correctness with cuda graph
4. after cudagraph capture, run some batch of requests, and the same 2 sequences are mixed in the batch
    the output should match 1.

### tests we don't cover, but might be useful to have

- PageTable unit tests
- tests with output length longer than one page (128 tokens)

"""

from transformers import AutoTokenizer
import time
from flex_nano_vllm import Gemma2ForCausalLM
from flex_nano_vllm.inference import Inference, Sequence, SamplingParams
from benchmark_vllm import generate_benchmark_data, long_prompt, short_prompt

import torch
from rich.console import Console
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

console = Console()
torch.set_float32_matmul_precision("high")


def get_profiler_context():
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    profiler_context = profile(
        activities=activities,
        schedule=schedule(wait=0, warmup=10, active=10, repeat=10),
        on_trace_ready=tensorboard_trace_handler("trace_dir"),
        record_shapes=False,
        profile_memory=False,
        with_stack=True,
        with_flops=False,
    )
    return profiler_context


# long_prompt and short_prompt are now imported from bench_utils


def debug_print_outputs(sequences, tokenizer, slice=slice(None), reference=None, prefix_match=False):
    results = []
    for i, seq in enumerate(sequences[slice]):
        output_ids = seq.output_ids
        output_decoded = tokenizer.decode(output_ids, skip_special_tokens=True)
        input_decoded = tokenizer.decode(seq.input_ids, skip_special_tokens=False)

        # If reference provided, only print on mismatch
        if reference is not None and i < len(reference):
            # Check for exact match or prefix match
            if prefix_match:
                matches = output_decoded.startswith(reference[i])
            else:
                matches = output_decoded == reference[i]

            if matches:
                match_type = "prefix match" if prefix_match else "match"
                console.print(f"i={i} ✓ {match_type}", style="green", markup=False)
            else:
                mismatch_type = "PREFIX MISMATCH" if prefix_match else "MISMATCH"
                console.print(f"i={i} ✗ {mismatch_type}", style="red bold", markup=False)
                console.print(f"  expected: {reference[i][:32]}...", style="red", markup=False)
                console.print(f"  got:      {output_decoded[:32]}...", style="red", markup=False)
                console.print(f"  input:    {input_decoded}", style="dim", markup=False)
        else:
            # Normal detailed output when no reference
            console.print(f"{i=} {input_decoded=} {output_decoded[:32]=}", style="bold ", markup=False)

        results.append(output_decoded)
    return results


if __name__ == "__main__":
    # Load model and tokenizer
    model_id = "google/gemma-2-2b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = Gemma2ForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto").eval()
    # model = torch.compile(model)

    B = 8
    max_new_tokens = 8
    # settings to match vLLM (see benchmark_vllm.py)
    paged_attn_max_batch_size = 512  # match vLLM max_num_seqs=256
    max_seq_length = 2048
    max_input_length = 1024
    token_allocation = 45_360  # 50% memory usage, this number is derived by running uv run benchmark_vllm.py and checking the logs (on a 3090)
    token_allocation = 140_432  # 100% memory usage, this number is derived by running uv run benchmark_vllm.py and checking the logs (on a 3090)

    prefill_length_limit = 1024 * 8  # helps control peak memory usage for prefill

    page_size = 128
    n_pages = int(token_allocation) // page_size

    print("initializing vllm inference")
    llm = Inference(
        model,
        tokenizer,
        max_batch_size=paged_attn_max_batch_size,
        max_seq_length=max_seq_length,
        n_pages=n_pages,
        kernel_options={"BLOCK_M": 32, "BLOCK_N": 32},
        prefill_length_limit=prefill_length_limit,
    )

    print("test 1")
    ## test 1
    sequences = [Sequence(long_prompt), Sequence(short_prompt)]
    llm.generate(sequences, sampling_params=SamplingParams(max_new_tokens=max_new_tokens), use_tqdm=True)
    results = debug_print_outputs(sequences, tokenizer)
    del sequences
    torch.cuda.empty_cache()

    print("test 2, same batch")
    sequences = [Sequence(long_prompt), Sequence(short_prompt)]
    llm.generate(sequences, sampling_params=SamplingParams(max_new_tokens=max_new_tokens), use_tqdm=True)
    results2 = debug_print_outputs(sequences, tokenizer, reference=results)
    for i in range(len(results)):
        assert results[i] == results2[i], f"{i=}, {results[i]=}, {results2[i]=}"
    del sequences
    torch.cuda.empty_cache()

    print("test 2.1: reverse order")
    sequences = [Sequence(short_prompt), Sequence(long_prompt)]
    llm.generate(sequences, sampling_params=SamplingParams(max_new_tokens=max_new_tokens), use_tqdm=True)
    # Debug in the right order: [long_prompt, short_prompt] to match reference
    reordered_sequences = [sequences[1], sequences[0]]  # [long_prompt, short_prompt]
    results21 = debug_print_outputs(reordered_sequences, tokenizer, reference=results)
    for i in range(len(results)):
        assert results[i] == results21[i], f"{i=}, {results[i]=}, {results21[i]=}"
    del sequences
    torch.cuda.empty_cache()

    print("test 3: batch with other sequence")
    sequences = [Sequence(short_prompt), Sequence(long_prompt), Sequence("hi")]
    llm.generate(sequences, sampling_params=SamplingParams(max_new_tokens=max_new_tokens), use_tqdm=True)
    # Debug just the sequences we care about in the right order: [long_prompt, short_prompt]
    comparison_sequences = [sequences[1], sequences[0]]  # [long_prompt, short_prompt]
    results3 = debug_print_outputs(comparison_sequences, tokenizer, reference=results)
    del sequences
    for i in range(len(results)):
        assert results[i] == results3[i], f"{i=}, {results[i]=}, {results3[i]=}"
    torch.cuda.empty_cache()

    print("test 4: batch with other sequence, capture cudagraph")
    sequences = [
        Sequence("this is a test messaage hello "),
        Sequence(short_prompt),
        Sequence("test"),
        Sequence(long_prompt),
        Sequence("hello world "),
    ]
    llm.generate(sequences, sampling_params=SamplingParams(max_new_tokens=max_new_tokens), use_tqdm=True, capture_cudagraph=True)
    # Debug just the sequences we care about in the right order: [long_prompt, short_prompt]
    comparison_sequences = [sequences[3], sequences[1]]  # [long_prompt, short_prompt]
    results4 = debug_print_outputs(comparison_sequences, tokenizer, reference=results)
    del sequences
    for i in range(len(results)):
        assert results[i] == results4[i], f"{i=}, {results[i]=}, {results4[i]=}"
    torch.cuda.empty_cache()

    # Generate test batch for cudagraph
    test_requests = generate_benchmark_data(tokenizer, n_requests=4, max_input_length=max_input_length)
    sequences = [Sequence(req.text) for req in test_requests]
    llm.generate(sequences, sampling_params=SamplingParams(max_new_tokens=16), use_tqdm=True, capture_cudagraph=True)
    # Just capture cudagraph, no need to show verbose output
    # debug_print_outputs(sequences, tokenizer)

    print("after cudagraph")
    test_requests2 = generate_benchmark_data(tokenizer, n_requests=4, max_input_length=max_input_length)
    sequences = [Sequence(req.text) for req in test_requests2] + [Sequence(long_prompt), Sequence(short_prompt)]
    print("replay cudagraph, & profile")
    with get_profiler_context() as prof:
        llm.generate(sequences, sampling_params=SamplingParams(max_new_tokens=16), use_tqdm=True, profiler=prof)
    results_prefill = debug_print_outputs(sequences, tokenizer, slice=slice(-2, None), reference=results, prefix_match=True)
    for i in range(len(results_prefill)):
        assert results_prefill[i][: len(results[i])] == results[i], f"{i=}, {results[i]=}, {results_prefill[i][:len(results[i])]=}"
    del sequences
    torch.cuda.empty_cache()

    ## benchmark throughput
    n_requests = 512
    max_input_length = 512
    # Use shared benchmark data generation
    benchmark_requests = generate_benchmark_data(
        tokenizer,
        n_requests=n_requests,
        max_input_length=max_input_length,
    )

    # Convert to flex-nano-vllm format
    sequences = [Sequence(req.text) for req in benchmark_requests]
    sampling_params = [SamplingParams(max_new_tokens=req.max_new_tokens) for req in benchmark_requests]

    print("\n--- RUNNING BENCHMARK ---")

    # Reset memory stats to track benchmark-specific usage
    torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    llm.generate(sequences, sampling_params=sampling_params, use_tqdm=False, save_metrics_csv=True, print_stats=True)
    total_time = time.time() - start_time

    total_output_length = sum(seq.output_length for seq in sequences)
    total_input_length = sum(len(seq.input_ids) for seq in sequences)

    # Get memory usage
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    current_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024

    print("\n--- PERFORMANCE METRICS ---")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {total_output_length / total_time:.1f} tokens/s")
    print(f"Request throughput: {len(sequences) / total_time:.2f} req/s")
    print(f"Total throughput (prompt+new): {(total_input_length + total_output_length) / total_time:.1f} tokens/s")
    print(f"Peak memory: {peak_memory_mb:.1f} MB")
    print(f"Current memory: {current_memory_mb:.1f} MB")

    print("\nafter benchmark")
    results_final = debug_print_outputs(sequences, tokenizer, slice=slice(n_requests, n_requests + 2), reference=results, prefix_match=True)

    # Verify correctness
    for i in range(len(results_final)):
        assert results[i] == results_final[i][: len(results[i])], f"{i=}, {results[i]=}, {results_final[i][:len(results[i])]=}"
