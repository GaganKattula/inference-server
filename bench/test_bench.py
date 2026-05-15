import torch
from bench.load_gen import generate_request
from src.config import ModelConfig
from bench.measure import run_benchmark, create_qwen_decoder
from src.scheduler import Request, Scheduler
from src.model import Decoder, load_hf_weights
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer
import copy

"""
  Setup:
  - 3 requests, not 20
  - lam=1.0, prompt_len_range=(3, 6), max_tokens_new=5, seed=0
  - Tiny fake model — a function that returns torch.randn(B, T, vocab_size) logits. No Qwen, no weight loading.
  - vocab_size=100 — small enough to be fast
  - chunk_size=3, block_size=4, num_blocks=50
"""


hf_config = LlamaConfig(
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,      # GQA: 4 query heads, 2 KV heads
        intermediate_size=512,
        vocab_size=1000,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        head_dim=64
    )




    # 1. Created model
hf_model = LlamaForCausalLM(hf_config)

our_config = ModelConfig(
            embedding_dim=hf_config.hidden_size,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=hf_config.head_dim,
            num_layers=hf_config.num_hidden_layers,
            d_ff=hf_config.intermediate_size,
            vocab_size=hf_config.vocab_size,
            context_length=hf_config.max_position_embeddings,
            rms_norm_eps=hf_config.rms_norm_eps,
            rope_theta=hf_config.rope_theta,
            act_fn="silu",
            dtype="float16",
            num_blocks=20,
            block_size=4)
    
our_model = Decoder(our_config)
load_hf_weights(our_model, hf_model.state_dict())
our_model.eval()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
our_model = our_model.to(device)
print(f"Model loaded on: {device}")

lam = 1.0
seed = 0
num_requests = 3
prompt_len_range=(3, 6) # short prompts for speed
max_tokens_new=5 # keeps decode phase short
vocab_size=151936 # Qwen vocab
chunk_size = 5

# 1 . Load Generation
requests = generate_request(lam = lam, seed = seed, num_requests = num_requests,
                             prompt_len_range = prompt_len_range, max_tokens_new = max_tokens_new, 
                             vocab_size = vocab_size )
requests2 = copy.deepcopy(requests)
requests3 = copy.deepcopy(requests)

print(f"Request Arrival time: {[req.arrival_time for _, req in requests]}")
print(f"Request Prompt Number of Tokens: {[req.num_prompt_tokens for _, req in requests]}")

# Config 1: static batching + contiguous
scheduler_static = Scheduler(block_size=16, num_blocks=10000, chunk_size=chunk_size,
                                cache_type="contiguous", static_batching=True)

# Config 2: continuous + contiguous
scheduler_cb = Scheduler(block_size=16, num_blocks=10000, chunk_size=chunk_size,
                           cache_type="contiguous", static_batching=False)

# Config 3: continuous + paged
scheduler_paged = Scheduler(block_size=16, num_blocks=10000, chunk_size=chunk_size,
                              cache_type="paged", static_batching=False)

#qwen_model, qwen_config, device = create_qwen_decoder()  # device = mps or cpu

# 3. Run Benchmark

# SB = static batching + contiguous cache
# CBC = Continuous batching + contiguous cache
# CBP = Continuous batching + paged cache

sb_metrics = run_benchmark(scheduler=scheduler_static, requests=requests,
                            model=our_model, config=hf_config, seed=seed, device=device)

cbc_metrics = run_benchmark(scheduler=scheduler_cb, requests=requests2,
                            model=our_model, config=hf_config, seed=seed, device=device)

cbp_metrics = run_benchmark(scheduler=scheduler_paged, requests=requests3,
                            model=our_model, config=hf_config, seed=seed, device=device)

assert all(req.status == "finished" for _, req in requests)
assert all(req.status == "finished" for _, req in requests2)
assert all(req.status == "finished" for _, req in requests3)

print(f"max_new_tokens: {[req.max_new_tokens for _, req in requests]}")

print(f"Static Batching:      TTFT={sb_metrics[0]:.3f}s  TPOT={sb_metrics[1]:.3f}s Throughput={sb_metrics[2]:.2f} req/s")
print(f"Continuous+Contiguous: TTFT={cbc_metrics[0]:.3f}s  TPOT={cbc_metrics[1]:.3f}s Throughput={cbc_metrics[2]:.2f} req/s")
print(f"Continuous+Paged:      TTFT={cbp_metrics[0]:.3f}s  TPOT={cbp_metrics[1]:.3f}s Throughput={cbp_metrics[2]:.2f} req/s")

##############################################

