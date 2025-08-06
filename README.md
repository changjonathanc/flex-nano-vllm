# flex-nano-vllm

FlexAttention based, minimal vllm-style inference engine for fast Gemma 2 inference.

## Introduction

This project has no flash-attn dependency, no custom triton kernel. Everything is implemented with FlexAttention. The code is commented, the structure is flat. Stay tuned for a more detailed blog post.

## Code Structure

```
flex-nano-vllm/
├── benchmark.py                   # Testing and benchmarking script.
├── benchmark_vllm.py              # vLLM comparison benchmark (uses uv inline dependency to run vLLM).
└── flex_nano_vllm/
    ├── inference.py               # Main inference engine, uses paged attention.
    ├── modeling_gemma2.py         # Gemma2 model implementation, copied from transformers.
    └── paged_attention.py         # Paged attention implementation, including page table and paged kv cache. Based on attention-gym.
```

## Quick Start

```
uv sync

# run test and benchmark
uv run benchmark.py

# compare with vllm
uv run benchmark_vllm.py

# enable profiling to save more metrics to a csv file
# ENABLE_PROFILING=1 uv run benchmark_vllm.py
```


## Results

Test configuration:
- PyTorch version: 2.7.1+cu128
- GPU: RTX 3090 x 1 (24GB)
- Model: google/gemma-2-2b
- Workload: 512 requests, max 512 input tokens, variable output tokens (128-512)
- Configs tested: vLLM at 50% & 90% GPU memory, flex-nano-vllm with same page allocation as vLLM

| Implementation | Output Tokens/s | Request/s | Total Throughput* |
|---------------|----------------|-----------|------------------|
| vLLM v1, 90% GPU memory | 3,772 | 15.26 | 6,401 | 
| flex-nano-vllm, 90% GPU memory | 3,266 | 13.83 | 5,794 | 
| vLLM v1, 50% GPU memory | 3,020 | 13.74 | 5,448 | 
| flex-nano-vllm, 50% GPU memory | 2,146 | 9.34 | 3,851 |

*Total throughput includes both input and output tokens  

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Third-party code incorporated in this project retains its original licenses. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for details.

## Acknowledgments

- [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm): this project is inspired by nano-vllm.
- [pytorch-labs/attention-gym](https://github.com/pytorch-labs/attention-gym): The paged attention implementation is based on attention-gym.
- [huggingface/transformers](https://github.com/huggingface/transformers): I copied the gemma2 model from transformers and modified it to use flex attention / paged attention.
- [vllm-project/vllm](https://github.com/vllm-project/vllm): vLLM has support for flex attention backend, which helped me find a useful flag in flex_attention.
