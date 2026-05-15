from src.scheduler import Request, Scheduler
from src.config import ModelConfig
from src.utils import collate_batch
from typing import List, Tuple
from bench.load_gen import generate_request
import torch
import time
from collections import deque
from transformers import AutoModelForCausalLM
from src.model import Decoder, load_hf_weights

"""
Define Eval Metrics

- TTFT - time taken to first token ( first token after prefill)
- TPOT - time per output token ( new token per decode step)
- throughput (number of requests processed per unit time)


Measurement method

- TTFT: average request arrival time and time the first token prediction time for all prefill requests

        first_token_time - arrival_time -> average across all prefill requests

- TPOT: average time between each output token prediction for decode requests
        request_finish_time - first_token_time / # max_tokens_generated -> average across all decode requests

- throughput: average time between arrival time and time when request status is changed ot finished
        total requests completed / total wall clock time ( requests/sec)

        wall clock time
            start - when the benchmark loop begins
            end - when last request is marked finished

- TPOT per request = (finish_time - first_token_time) / tokens_generated
- TTFT per request = first_token_time - arrival_time


block_cost = num_layers * num_kv_heads * head_dim * 2 * 2 * block_size
    
FOR QWEN2.5 - 1.5B 

    Block cost ( 16 tokens/block)   28*2*128*2*2*16 = 458752 bytes ~ 0.0004272461 GB
    Model weights                   1.5B * 2 bytes = 3GB
    Memory budget = 5GB
    num_blocks = 5GB /  0.0004272461 GB


"""
# min, max = 5, 20

# prompt_len_range = (min, max)
# seed = 42

# block_size = 16
# num_blocks = 10000
#     #prompt_tokens = [1, 450, 999, 310, 657, 338, 123, 204, 302, 142]  # arbitrary token IDs within vocab_size=1000
#     #num_prompt_tokens = len(prompt_tokens)
# max_new_tokens = 20
# chunk_size = 5


# Three configs:
#           static batching, 
#           continuous batching + contiguous cache, 
#           continuous batching + paged cache


# # Config 1: static batching + contiguous
# scheduler_static = Scheduler(block_size=16, num_blocks=10000, chunk_size=chunk_size,
#                                 cache_type="contiguous", static_batching=True)

# # Config 2: continuous + contiguous
# scheduler_cb = Scheduler(block_size=16, num_blocks=10000, chunk_size=chunk_size,
#                            cache_type="contiguous", static_batching=False)

# # Config 3: continuous + paged
# scheduler_paged = Scheduler(block_size=16, num_blocks=10000, chunk_size=chunk_size,
#                               cache_type="paged", static_batching=False)




# requests = generate_request(lam=2.0, num_requests=100, vocab_size=768, prompt_len_range= prompt_len_range, max_tokens_new= 15, seed=seed) # list of tuples with arrival times and request objects
def run_benchmark(model,
                    scheduler: Scheduler,
                        requests: List[Tuple[float, Request]],
                            config: ModelConfig, seed: int,
                            device: torch.device = torch.device("cpu"))-> dict:
    
    pending = deque(requests)
    torch.manual_seed(seed)
    eval_metrics = dict()

    start_time = time.time()

    

    # Check for requests OR if running_list exists 
    while pending or scheduler.running_list:
        

    # Keep looping as long as there are either:
    #   - requests not yet admitted (pending)
    #   - requests admitted but not yet finished (scheduler.running_list)

        current_time = time.time() - start_time
        
        # admit arrived requests
        while pending and pending[0][0] <= current_time:
            arrival_time, req = pending.popleft()
            arrival_time = start_time + arrival_time
            req.arrival_time = arrival_time
            scheduler.add_request(req)
        # scheduler.step() → forward → scheduler.update()
            #print(f"ADMITTED: {req.request_id[:8]} at {current_time:.2f}s")
        p_batches, d_batches, prefill_reqs, decode_reqs = scheduler.step() # admits requests
            
        p_batch_token_ids, p_batch_positions, p_batch_tables = p_batches #unpack tuple
        d_batch_token_ids, d_batch_positions, d_batch_tables = d_batches #unpack tuple

        # PREFILL: collate variable-length chunks into a padded batch, one forward call for all
        # NOTE: attention_mask disabled — RoPE not yet updated for batched (B, T) positions.
        # Padding positions will attend incorrectly but relative metrics across configs are valid.
        # TODO: fix RoPE for batched positions, re-enable attention_mask | DONE
        if p_batch_token_ids:
            token_ids, attention_mask = collate_batch(p_batch_token_ids)
            attention_mask = attention_mask.to(device)
            token_ids = token_ids.to(device)                        # (B, T_max) on device
            positions, _ = collate_batch(p_batch_positions, pad_value=0)
            positions = positions.to(device)                     # (T_max,) — use first request positions, RoPE expects 1D
            logits, _ = model.forward(token_ids=token_ids, positions=positions,attention_mask=attention_mask,
                                      block_table=None, seq_len=None )
            scheduler.update(logits, prefill_reqs)

        # DECODE: each request generates one token — length 1, no padding needed
        if d_batch_token_ids:
            token_ids = torch.stack(d_batch_token_ids).to(device)          # (B, 1) on device
            positions = torch.stack(d_batch_positions).to(device)                    # (1,) — RoPE expects 1D
            logits, _ = model.forward(token_ids=token_ids, positions=positions,attention_mask=None,
                                      block_table=None, seq_len=None)
            scheduler.update(logits, decode_reqs)

    
     
    end_time = time.time()
    
    #print(f"prefill batch size: {len(prefill_reqs)}, decode batch size: {len(decode_reqs)}")
    #finished_requests = [req for _, req in requests]

    finished = [(at, req) for at, req in requests if req.status == "finished"]
    ttft = sum(req.first_token_time - req.arrival_time for _, req in finished) / len(finished)
    tpot = sum((req.finish_time - req.first_token_time) / req.tokens_generated for _, req in finished) / len(finished)
    throughput = len(finished) / (end_time - start_time)

    eval_metrics = (ttft, tpot, throughput)
    
    return eval_metrics


# LOAD MODEL

def create_qwen_decoder():

    # Step 1 - Instantiate Model from HF
    qwen2_5_1_5B =  AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B')
    # Step 2 - ModelConfig for Qwen 2.5 1.5B
    qwen_config = ModelConfig( embedding_dim= 1536,
                            num_heads=12,
                            num_kv_heads= 2,
                            num_layers= 28,
                            d_ff= 8960,
                            vocab_size=151936,
                            context_length= 131072,
                            rms_norm_eps= 1e-6,
                            rope_theta= 1000000.0,
                            head_dim= 128,
                            act_fn="silu",
                            dtype="float16",
                            num_blocks=10000,
                            block_size=16)
    # Step 3 - Instantiate Decoder with the config
    qwen_model = Decoder(qwen_config)

    # Step 4 - load_hf_weights
    load_hf_weights(qwen_model, qwen2_5_1_5B.state_dict() )
    qwen_model.eval()

    # Step 5 - Move model to MPS (Apple Silicon GPU) if available, else CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    qwen_model = qwen_model.to(device)
    print(f"Model loaded on: {device}")

    return qwen_model, qwen_config, device  # return device so forward loop can move tensors




