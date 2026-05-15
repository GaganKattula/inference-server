# Inference Server

A minimal vLLM-style LLM inference server implementing PagedAttention, continuous batching, and prefill/decode separation. Serves LLaMA 3.2 3B. Benchmarked pssioagainst vLLM on identical hardware and workload.

## Features

- **Paged KV-cache** — block-based memory management eliminating KV cache fragmentation
- **Continuous batching** — iteration-level scheduling supporting heterogeneous request lengths
- **Prefill/decode separation** — separate handling of compute-bound and memory-bound phases
- **GQA support** — grouped query attention with correct head broadcasting

## Architecture

```
src/
  config.py           — ModelConfig dataclass
  model.py            — LLaMA decoder (AttentionGQA, RoPE, SwiGLU, RMSNorm)
  paged_cache.py      — BlockAllocator, BlockTable, paged attention kernel
  scheduler.py        — continuous batching scheduler
tests/
  test_weight_parity.py      — logits within atol=1e-4 of HuggingFace LLaMA
  test_paged_equivalence.py  — paged KV-cache equivalence vs contiguous
  test_scheduler.py          — scheduler + paged attention equivalence gate
```

## Tests

```bash
pytest tests/
```
