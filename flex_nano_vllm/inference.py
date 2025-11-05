from collections import deque
import time
import torch
from tqdm import tqdm
from torch.nn.attention.flex_attention import BlockMask
import torch.nn.attention.flex_attention
import torch.nn.functional as F
from rich.console import Console

from flex_nano_vllm.paged_attention import PageTable, PagedKVCache

from dataclasses import dataclass

console = Console()
print(f"torch version: {torch.__version__}")


@dataclass
class SamplingParams:
    max_new_tokens: int = -1


def sample(logits_BV, greedy=True, to_cpu=False):
    # NOTE: use greedy=True to ensure deterministic sampling
    assert logits_BV.ndim == 2
    B, V = logits_BV.shape
    probs = torch.softmax(logits_BV, dim=-1)
    if not greedy:
        indices = torch.multinomial(probs, num_samples=1)  # shape: [B, 1]
        logits = torch.gather(logits_BV, dim=-1, index=indices)
        probs = torch.gather(probs, dim=-1, index=indices)
    else:
        probs, indices = torch.topk(probs, k=1, dim=-1)
        logits = torch.gather(logits_BV, dim=-1, index=indices)
    if to_cpu:
        indices = indices.to("cpu", non_blocking=True).view(B)
        logits = logits.to("cpu", non_blocking=True).view(B)
        probs = probs.to("cpu", non_blocking=True).view(B)
        torch.cuda.synchronize()
    return indices.tolist(), logits.tolist(), probs.tolist()


class Sequence:
    def __init__(self, text: str):
        self.done = False
        self.text = text
        self._output_ids = []
        self._output_logits = []
        self._output_probs = []
        self.input_ids = []
        self.finished = False

        self.input_length = None
        self.inputs = None

    def add_next_token(self, token_id: torch.Tensor, logits: torch.Tensor, probs: torch.Tensor):
        #assert token_id.ndim == 0
        #assert logits.ndim == 0
        self._output_ids.append(token_id)
        self._output_logits.append(logits)
        self._output_probs.append(probs)

    def copy(self):
        return Sequence(self.text)

    @property
    def output_ids(self):
        return torch.tensor(self._output_ids, dtype=torch.int64)

    @property
    def output_logits(self):
        return torch.tensor(self._output_logits, dtype=torch.float32)

    @property
    def output_probs(self):
        return torch.tensor(self._output_probs, dtype=torch.float32)

    @property
    def output_length(self):
        return len(self._output_ids)

    @property
    def total_length(self):
        return self.input_length + self.output_length

    @property
    def total_token_ids(self):
        if self.output_length:
            return torch.cat([self.input_ids, self.output_ids], dim=0)
        return self.input_ids

    @property
    def last_token_id(self):
        return self._output_ids[-1]


def process_sampling_params(sequences: list[Sequence], sampling_params: SamplingParams | list[SamplingParams] | None):
    if sampling_params is None:
        sampling_params = SamplingParams()
    if isinstance(sampling_params, SamplingParams):
        sampling_params = [sampling_params] * len(sequences)

    assert len(sampling_params) == len(sequences), "sampling_params must be a list of the same length as sequences"

    for seq, param in zip(sequences, sampling_params):
        seq.params = param


class Inference:
    def __init__(self, model, tokenizer, max_batch_size, max_seq_length, n_pages, page_size=128, prefill_length_limit=-1, kernel_options=None):
        self.page_table = PageTable(n_pages=n_pages, page_size=page_size, max_batch_size=max_batch_size)

        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id # cache this because it's not efficient to call tokenizer.eos_token_id every time
        self.device = model.device
        assert max_seq_length % page_size == 0, "max_seq_length must be divisible by page_size"
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        self.kernel_options = kernel_options
        self.prefill_length_limit = prefill_length_limit  # NOTE: control the peak memory usage of prefill

        for layer in self.model.model.layers:
            layer.self_attn.kv_cache = PagedKVCache(
                self.page_table,
                n_heads=self.model.model.config.num_key_value_heads,
                head_dim=self.model.model.config.head_dim,
                dtype=self.model.dtype,
            ).to(self.device, non_blocking=True)

        self.cudagraph_captured = False

        self.input_pos = torch.zeros(self.max_batch_size, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        self.block_mask = self.page_table.create_causal_blockmask(B=self.max_batch_size, L=self.max_seq_length)

    def _prefill_sequences(
        self, input_ids: torch.Tensor, input_pos: torch.Tensor, batch_idx_tensor: torch.Tensor, logits_to_keep: torch.Tensor
    ) -> torch.Tensor:
        # 1. no cuda graph
        # 2. construct block mask and apply it in logical space
        # 3. only write to kv cache, no read

        # NOTE: for batch/packed prefill, we need to pass batch_idx_tensor as [1, L]
        # input_ids is [1, L], concatenated from all sequences
        # batch_idx_tensor is [1, L]
        # position_ids is [1, L]
        # logits_to_keep is [num_sequences] instead of [1]

        ## padding: if there's padding
        # input_ids should be padded with any valid token id
        # input_pos should be padded with 0
        # batch_idx_tensor should be padded with 0 # reserved in page table

        assert input_ids.shape[0] == 1, "input_ids must be [1, L]"
        assert input_pos.shape == input_ids.shape, f"input_pos must be [1, L], got {input_pos.shape=}, {input_ids.shape=}"
        assert batch_idx_tensor.shape == input_ids.shape, f"batch_idx_tensor must be [1, L], got {batch_idx_tensor.shape=}, {input_ids.shape=}"

        mask = self.page_table.create_prefill_blockmask_no_paging(batch_idx_tensor)
        outputs = self.model.model(
            input_ids=input_ids,
            position_ids=input_pos + 1,  # NOTE: gemma2 uses 1-based position ids
            # logits_to_keep=logits_to_keep,
            flex_attn_block_mask=mask,
            flex_attn_input_pos=input_pos,
            flex_attn_batch_idx=batch_idx_tensor,
            flex_attn_kernel_options=self.kernel_options
            | {"FORCE_USE_FLEX_ATTENTION": True},  # NOTE: force torch compile to not use flash decoding code path
        )
        return self.model.lm_head(outputs.last_hidden_state[:, logits_to_keep, :])

        """
        outputs = self.model(
            input_ids=input_ids,
            position_ids=input_pos + 1, # NOTE: gemma2 uses 1-based position ids
            logits_to_keep=logits_to_keep,
            flex_attn_block_mask=mask,
            flex_attn_input_pos=input_pos,
            flex_attn_batch_idx=batch_idx_tensor,
            flex_attn_kernel_options=self.kernel_options | {'FORCE_USE_FLEX_ATTENTION': True}, # NOTE: force torch compile to not use flash decoding code path
        )
        return outputs.logits
        """

    def prefill_sequences(self, sequences: list[Sequence]) -> torch.Tensor:
        input_ids = torch.cat([seq.total_token_ids for seq in sequences], dim=0)
        input_pos = torch.cat([torch.arange(seq.total_length, dtype=torch.long) for seq in sequences], dim=0)
        batch_idx_tensor = torch.cat([torch.ones(seq.total_length, dtype=torch.long) * seq.batch_idx for seq in sequences], dim=0)
        input_lengths = torch.tensor([seq.total_length for seq in sequences], dtype=torch.int32).to(self.device, non_blocking=True)
        logits_to_keep = input_lengths.cumsum(dim=0) - 1

        num_pad = 128 - input_ids.shape[0] % 128
        if num_pad > 0:
            input_ids = F.pad(input_ids.view(-1), (0, num_pad), mode="constant", value=0)
            input_pos = F.pad(input_pos.view(-1), (0, num_pad), mode="constant", value=0)
            batch_idx_tensor = F.pad(batch_idx_tensor.view(-1), (0, num_pad), mode="constant", value=0)
            # logits_to_keep is not padded, it should have shape [num_sequences]

        input_ids = input_ids.view(1, -1).to(self.device, non_blocking=True)
        input_pos = input_pos.view(1, -1).to(self.device, non_blocking=True)
        batch_idx_tensor = batch_idx_tensor.view(1, -1).to(self.device, non_blocking=True)
        logits_to_keep = logits_to_keep.view(-1).to(self.device, non_blocking=True)

        logits = self._prefill_sequences(input_ids, input_pos, batch_idx_tensor, logits_to_keep)
        return logits

    def get_decoding_block_mask(self, batch_idx: torch.Tensor):
        """
        Args:
            batch_idx: [B]
        Returns:
            block_mask: [B, H, ROWS=1, MAX_BLOCKS_IN_COL]
            input_pos: [B]

        This function slices the
            full block mask self.block_mask:  [max_batch_size, H, MAX_BLOCKS_IN_ROW, MAX_BLOCKS_IN_COL]
            using self.input_pos: [max_batch_size]
            and batch_idx: [B]
        """

        # NOTE: this function is entirely in logical space
        def causal_offset(off: torch.Tensor):
            def offset(b, h, q_idx, kv_idx):
                return q_idx + off[b] >= kv_idx

            return offset

        block_mask = self.block_mask
        input_pos = self.input_pos[batch_idx]
        # batch_idx: [B], input_pos: [B]
        assert batch_idx.ndim == 1, "batch_idx must be 1D"
        assert input_pos.ndim == 1, "input_pos must be 1D"
        (B,) = batch_idx.shape
        input_block_idx = input_pos // block_mask.BLOCK_SIZE[0]  # [B]
        kv_num_blocks = block_mask.kv_num_blocks[batch_idx, :, input_block_idx].view(B, 1, 1)
        kv_indices = block_mask.kv_indices[batch_idx, :, input_block_idx].view(B, 1, 1, -1)
        full_kv_num_blocks, full_kv_indices = None, None
        if block_mask.full_kv_num_blocks is not None:
            full_kv_num_blocks = block_mask.full_kv_num_blocks[batch_idx, :, input_block_idx].view(B, 1, 1)  # noqa
            full_kv_indices = block_mask.full_kv_indices[batch_idx, :, input_block_idx].view(B, 1, 1, -1)  # noqa
        seq_length = (1, block_mask.seq_lengths[1])
        mask = BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            BLOCK_SIZE=block_mask.BLOCK_SIZE,
            mask_mod=causal_offset(input_pos),
            seq_lengths=seq_length,
        )
        return mask, input_pos

    def _decode_step(self, batch_idx: torch.Tensor, input_ids: torch.Tensor):
        B = input_ids.shape[0]
        mask, input_pos = self.get_decoding_block_mask(batch_idx)
        mask = self.page_table.convert_logical_block_mask(mask, batch_idx)
        outputs = self.model(
            input_ids=input_ids.view(B, 1),
            position_ids=(input_pos + 1).view(B, 1),  # NOTE: position_ids is needed for decoding. For Gemma2, it's 1-based
            flex_attn_block_mask=mask,
            flex_attn_input_pos=input_pos.view(B, 1),
            flex_attn_batch_idx=batch_idx.view(-1),
            flex_attn_kernel_options=self.kernel_options,
        )
        return outputs.logits

    def decode_step(self, batch_idx: torch.Tensor, input_ids: torch.Tensor, input_pos: torch.Tensor):
        assert input_ids.ndim == 1, "input_ids must be 1D"
        assert batch_idx.ndim == 1, "batch_idx must be 1D"
        assert input_ids.shape[0] == batch_idx.shape[0], "input_ids and batch_idx must have the same length"
        self.input_pos.zero_()
        self.input_pos[batch_idx] = input_pos
        if not self.cudagraph_captured:
            return self._decode_step(batch_idx, input_ids)
        else:
            bs = input_ids.size(0)
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()  # batch idx is 0 for undefined batch idx, so we never overwrite any kv-cache
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["batch_idx"][:bs] = batch_idx
            graph.replay()
            replayed = graph_vars["outputs"][:bs]
            return replayed

    def _check_done(self, sequences: list[Sequence]):
        for seq in sequences:
            is_eos = seq.last_token_id == self.eos_token_id
            is_max_len = seq.input_length + seq.output_length >= self.max_seq_length
            is_max_new = seq.output_length == seq.params.max_new_tokens
            if is_eos or is_max_len or is_max_new:
                seq.finished = True
                self.done.append(seq)
                self.page_table.erase(seq.batch_idx)
        return [seq for seq in sequences if not seq.finished]

    def run_one_step(self):
        # try to prefill a new sequence, if we can schedule it
        batch = []
        prefill_length_sum = 0
        # NOTE: for each sequence, only allocate & reserve just enough pages for the current sequence length
        # We do not reserve pages for the future tokens. This allows maximizing the batch size of decoding.
        # We support preemption in case we run out of pages during decoding
        while (
            self.waiting
            and self.page_table.can_reserve(self.waiting[0].total_length)  # + self.waiting[0].params.max_new_tokens)
            and (not batch or self.prefill_length_limit == -1 or prefill_length_sum + self.waiting[0].total_length < self.prefill_length_limit)
        ):
            prefill_length_sum += self.waiting[0].total_length

            seq = self.waiting.popleft()
            batch_idx_int = self.page_table.allocate()
            batch_idx_tensor = torch.tensor([batch_idx_int], device=self.device, dtype=torch.long)
            self.page_table.reserve(batch_idx_int=batch_idx_int, batch_idx=batch_idx_tensor, seq_len=seq.total_length)  # + seq.params.max_new_tokens
            seq.batch_idx = batch_idx_int
            batch.append(seq)
        if batch:
            logits_1LV = self.prefill_sequences(batch)
            next_token, logits, probs = sample(logits_1LV[0, :, :], to_cpu=True)
            for b in range(len(batch)):
                batch[b].add_next_token(next_token[b], logits[b], probs[b])
            self.running.extend(self._check_done(batch))
            return "prefill"
        # reserve new block for running sequences if needed
        # if a new block is needed, but there's no space, preempt the newest sequence
        # running: [oldest, ... , newest]  waiting: [oldest, ... , newest]
        while self.running:
            seq = self.running.popleft()
            if self.page_table.capacity[seq.batch_idx] >= seq.total_length:
                # no need to reserve new pages
                batch.append(seq)
            elif self.page_table.can_reserve(seq.total_length, batch_idx_int=seq.batch_idx):
                # reserve new pages
                self.page_table.reserve(
                    batch_idx_int=seq.batch_idx,
                    batch_idx=torch.tensor([seq.batch_idx], device=self.device, dtype=torch.long),
                    seq_len=seq.total_length,
                )
                batch.append(seq)
            else:
                # no space to run this sequence, preempt the newest sequence
                self.running.appendleft(seq)  # first put this sequence back
                newest = self.running.pop()  # then pop the newest sequence
                self.waiting.appendleft(newest)
                self.page_table.erase(newest.batch_idx)

        B = len(batch)
        # now we do decoding
        batch_idx = torch.tensor([seq.batch_idx for seq in batch], dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
        input_ids = torch.tensor([seq.last_token_id for seq in batch], dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
        input_pos = torch.tensor([seq.total_length - 1 for seq in batch], dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        self.counts.append(B)
        logits_BLV = self.decode_step(batch_idx, input_ids, input_pos)
        next_token, logits, probs = sample(logits_BLV[:, -1, :], to_cpu=True)

        for i in range(B):
            batch[i].add_next_token(next_token[i], logits[i], probs[i])
        self.running = deque(self._check_done(batch))
        return "decode"

    def tokenize(self, sequences: list[Sequence]):
        self.tokenizer.padding_side = "right"
        for seq in sequences:
            seq.input_ids = self.tokenizer([seq.text], return_tensors="pt")["input_ids"].squeeze(0)
            seq.input_length = seq.input_ids.shape[0]

    @torch.inference_mode()
    def generate(
        self,
        sequences: list[Sequence],
        use_tqdm=False,
        profiler=None,
        greedy=False,
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
        capture_cudagraph=False,
        save_metrics_csv=False,
        print_stats=False,
    ):
        self.counts = []
        self.metrics_data = {"step": [], "requests_running": [], "requests_waiting": [], "step_type": []}
        # preprocess the sequences
        self.tokenize(sequences)
        self.waiting = deque(sequences)
        self.running = deque()
        self.done = deque()
        process_sampling_params(sequences, sampling_params)

        if capture_cudagraph and not self.cudagraph_captured:
            self.capture_decode_cudagraph()
            self.cudagraph_captured = True

        total_sequences = len(self.waiting)
        times = []
        step_count = 0
        with tqdm(total=total_sequences, disable=not use_tqdm, desc="Generating") as pbar:
            prev_done = 0
            while self.waiting or self.running:
                step_count += 1
                # Track metrics before step
                self.metrics_data["step"].append(step_count)
                self.metrics_data["requests_running"].append(len(self.running))
                self.metrics_data["requests_waiting"].append(len(self.waiting))

                time_start = time.perf_counter()
                step_type = self.run_one_step()
                time_end = time.perf_counter()

                # Track step type
                self.metrics_data["step_type"].append(step_type)
                times.append({"step_type": step_type, "time": time_end - time_start})
                if profiler:
                    profiler.step()
                # Update progress bar based on newly completed sequences
                curr_done = len(self.done)
                if curr_done > prev_done:
                    pbar.update(curr_done - prev_done)
                    prev_done = curr_done
        if print_stats:
            self.print_time_stats(times)

        # Save metrics to CSV if requested
        if save_metrics_csv:
            import pandas as pd

            df = pd.DataFrame(self.metrics_data)
            df.to_csv("flex_nano_vllm_metrics.csv", index=False)
            print("Metrics saved to 'flex_nano_vllm_metrics.csv'")

    def capture_decode_cudagraph(self):
        """
        capture cudagraph for decoding
        """
        max_bs = self.max_batch_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
        batch_idx = torch.arange(max_bs, dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
        # NOTE: here we use logits as the final output, but we can consider using last hidden state as the output
        outputs = torch.zeros((max_bs, 1, self.model.model.config.vocab_size), pin_memory=True).to(self.device, non_blocking=True)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))  # 8 vs 16 doesn't make much difference
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            print(f"capturing cudagraph for {bs} sequences")
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            # warmup
            outputs[:bs] = self._decode_step(batch_idx[:bs], input_ids[:bs])  # warmup
            # capture
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self._decode_step(batch_idx[:bs], input_ids[:bs])  # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()

        self.graph_vars = dict(
            input_ids=input_ids,
            batch_idx=batch_idx,
            outputs=outputs,
        )
        # in our code, the page table tensors are modified in-place, so we don't need to put them in graph vars

    def _calculate_time_stats(self, times: list[dict]) -> dict:
        stats = {}
        for step in ["decode", "prefill"]:
            step_times = [t["time"] for t in times if t["step_type"] == step]
            stats[step] = {
                "count": len(step_times),
                "total": sum(step_times),
                "mean": sum(step_times) / len(step_times) if step_times else 0,
                "min": min(step_times) if step_times else 0,
                "max": max(step_times) if step_times else 0,
            }
        stats["total_time"] = sum(t['time'] for t in times)
        return stats

    def print_time_stats(self, times: list[dict]):
        stats = self._calculate_time_stats(times)
        print("\nTime statistics by step type:")
        for step, metrics in stats.items():
            if step == "total_time":
                continue
            print(f"\n{step}:")
            print(f"  Count: {metrics['count']}")
            print(f"  Total: {metrics['total']:.4f}s")
            print(f"  Mean:  {metrics['mean']:.4f}s")
            print(f"  Min:   {metrics['min']:.4f}s")
            print(f"  Max:   {metrics['max']:.4f}s")
        print(f"\nTotal time: {stats['total_time']:.4f}s")
