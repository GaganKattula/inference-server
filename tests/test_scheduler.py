from src.config import ModelConfig
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer
from src.model import Decoder, load_hf_weights
from src.scheduler import Scheduler, Request
from src.paged_cache import BlockAllocator, BlockTable, attention_kernel
import torch
import time


"""
  1. Create a tiny LLaMA config and model
  2. Create a scheduler with enough blocks for the test
  3. Submit one request via add_request()
  4. Run a generation loop: step() → model.forward() → update() until finished
  5. Also run the same prompt through the model directly in a simple greedy loop (no scheduler)
  6. Assert both produce identical output token sequences

"""
def test_scheduler_equivalence():
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

    # 

    block_size = 4
    num_blocks = 20
    prompt_tokens = [1, 450, 999, 310, 657, 338]  # arbitrary token IDs within vocab_size=1000
    num_prompt_tokens = len(prompt_tokens)
    max_new_tokens = 5

    # 2. Created scheduler instance
    scheduler = Scheduler(block_size, num_blocks)
    # 3. Created a request instance
    request = Request(request_id = 0, prompt_tokens=prompt_tokens, num_prompt_tokens=num_prompt_tokens,
                      max_new_tokens = max_new_tokens, block_table =None, arrival_time= time.time() )
    
    # 4. Paged Generation loop
    caches = None

    scheduler.add_request(request)  # enqueue once before the loop
    while request.status != "finished":
        batch_token_ids, batch_positions, batch_tables = scheduler.step()
        token_ids = torch.cat(batch_token_ids).unsqueeze(0)
        positions = torch.cat(batch_positions) #.unsqueeze(0)  {leads to wrong dim being repeated in expand step}  # (1, T) — batch dim for model   # (1, T)
        logits, caches = our_model.forward(token_ids=token_ids, positions=positions, block_table=request.block_table, seq_len=num_prompt_tokens + request.tokens_generated)
        scheduler.update(logits)


    # 5. Generation w/o scheduler

    # naive greedy — no scheduler, contiguous cache
    ref_tokens = []
    caches = None
    token_ids = torch.tensor(prompt_tokens).unsqueeze(0)  # (1, T)
    positions = torch.arange(len(prompt_tokens)) # The model's RoPE always expects positions as (T,)  # (1, T)

    for _ in range(max_new_tokens):
        logits, caches = our_model(token_ids, positions, caches)
        next_token = logits[0, -1, :].argmax().item()
        ref_tokens.append(next_token)
        token_ids = torch.tensor([[next_token]])
        positions = torch.tensor([len(prompt_tokens) + len(ref_tokens) - 1])


    assert request.output_token_ids == ref_tokens, (
        f"Paged output {request.output_token_ids} != reference {ref_tokens}"
    )