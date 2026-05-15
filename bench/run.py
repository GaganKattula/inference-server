from bench.load_gen import generate_request
from bench.measure import run_benchmark, create_qwen_decoder
from src.scheduler import Request, Scheduler
import copy
import os
import json

"""
 1. Generate test load
 2. Create scheduler instances for unique test configurations
 3. Run benchamrk and compare metrics
 
"""

lam = 2.0
seed = 42
num_requests = 20
prompt_len_range=(10, 50) # short prompts for speed
max_tokens_new=20 # keeps decode phase short
vocab_size=151936 # Qwen vocab
chunk_size = 5
lam_range = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]

qwen_model, qwen_config, device = create_qwen_decoder()  # device = mps or cpu

sb=dict()
cbc=dict()
cbp=dict()

for lam in lam_range:
# 1 . Load Generation
    requests = generate_request(lam = lam, seed = seed, num_requests = num_requests,
                             prompt_len_range = prompt_len_range, max_tokens_new = max_tokens_new, 
                             vocab_size = vocab_size )

    requests2 = copy.deepcopy(requests)
    requests3 = copy.deepcopy(requests)
# 2. Schedulers

# Config 1: static batching + contiguous
    scheduler_static = Scheduler(block_size=16, num_blocks=10000, chunk_size=chunk_size,
                                cache_type="contiguous", static_batching=True)

# Config 2: continuous + contiguous
    scheduler_cb = Scheduler(block_size=16, num_blocks=10000, chunk_size=chunk_size,
                           cache_type="contiguous", static_batching=False)

# Config 3: continuous + paged
    scheduler_paged = Scheduler(block_size=16, num_blocks=10000, chunk_size=chunk_size,
                              cache_type="paged", static_batching=False)

    

# 3. Run Benchmark

# SB = static batching + contiguous cache
# CBC = Continuous batching + contiguous cache
# CBP = Continuous batching + paged cache

    sb_metrics = run_benchmark(scheduler=scheduler_static, requests=requests,
                            model=qwen_model, config=qwen_config, seed=seed, device=device)

    cbc_metrics = run_benchmark(scheduler=scheduler_cb, requests=requests2,
                            model=qwen_model, config=qwen_config, seed=seed, device=device)

    cbp_metrics = run_benchmark(scheduler=scheduler_paged, requests=requests3,
                            model=qwen_model, config=qwen_config, seed=seed, device=device)
    
    sb[lam] = sb_metrics
    cbc[lam] = cbc_metrics
    cbp[lam] = cbp_metrics


    print("\n")
    print(f"METRICS FOR LAM VALUE : {lam}\n")
    print("==============================================================================================================================")
    print("\n")
    print(f"Static Batching:      TTFT={sb_metrics[0]:.3f}s  TPOT={sb_metrics[1]:.3f}s Throughput={sb_metrics[2]:.2f} req/s")
    print(f"Continuous+Contiguous: TTFT={cbc_metrics[0]:.3f}s  TPOT={cbc_metrics[1]:.3f}s Throughput={cbc_metrics[2]:.2f} req/s")
    print(f"Continuous+Paged:      TTFT={cbp_metrics[0]:.3f}s  TPOT={cbp_metrics[1]:.3f}s Throughput={cbp_metrics[2]:.2f} req/s")
    print("\n")
    print("==============================================================================================================================\n")

# Convert metrics dicts to string keys and list values
sb_clean = {str(k): (list(v) if isinstance(v, tuple) else v) for k, v in sb.items()}
cbc_clean = {str(k): (list(v) if isinstance(v, tuple) else v) for k, v in cbc.items()}
cbp_clean = {str(k): (list(v) if isinstance(v, tuple) else v) for k, v in cbp.items()}

# pack it in one dict
results = {"sb": sb_clean, "cbc": cbc_clean, "cbp": cbp_clean}

# store json
file_path = "/Users/gagan/Documents/studysessions/appliedlearning/inference-server/bench/results.json"
os.makedirs(os.path.dirname(file_path), exist_ok=True)

with open(file_path, "w") as f:
    json.dump(results, f, indent=4)