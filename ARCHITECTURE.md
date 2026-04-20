# Architecture: plato-inference-runtime

## Language Choice: Rust

### Why Rust

Inference scheduling is in the hot path of every user query.
Python's asyncio overhead + per-request object allocation adds 5-20ms latency.

| Metric | Python (asyncio) | Rust (sync + heap) |
|--------|-----------------|-------------------|
| Schedule 10K requests | ~120ms | ~8ms |
| Batch 100 requests | ~15ms | ~1ms |
| Memory per request | ~500 bytes | ~80 bytes |

### Why not vLLM / TGI

vLLM and TGI are excellent for production LLM serving (PagedAttention, continuous
batching, multi-GPU). But: 100K+ lines, CUDA dependency, complex deployment.

Our runtime manages PLATO's smaller adapter-based inference:
- 7B base model (3.5GB Q4) + 100MB adapters
- <10 concurrent users per node
- Batch sizes 1-32

We'd integrate with vLLM when we need: PagedAttention, multi-GPU, or OpenAI API.

### Architecture

```
register_model() → submit(prompt) → priority queue → process_batch() → InferenceResult
                    ↓                   ↓
               auto-load model    priority-sorted batch
                    ↓
               evict_idle() when memory pressure
```

### Resource Management

- **LRU eviction**: Unload least-recently-used models when memory is full
- **Auto-loading**: Submit requests for unloaded models → auto-load
- **Priority queue**: High-priority requests processed first in batch
- **Batch timeout**: Process partial batch after timeout even if not full
