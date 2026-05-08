"""Debug scripts for weight parity issues.

Run individual functions to diagnose why logits diverge.
"""

import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from src.config import ModelConfig
from src.model import Decoder, load_hf_weights

MODEL_NAME = "unsloth/Llama-3.2-3B"


def check_weight_tying():
    """Check if HF model ties embedding and lm_head weights."""
    config = AutoConfig.from_pretrained(MODEL_NAME)
    print(f"tie_word_embeddings: {config.tie_word_embeddings}")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    keys = list(model.state_dict().keys())
    print(f"Has lm_head.weight: {'lm_head.weight' in keys}")
    print(f"Total keys: {len(keys)}")
    for k in keys:
        if 'embed' in k or 'lm_head' in k:
            print(f"  {k}: {model.state_dict()[k].shape}")


def check_key_mapping():
    """Print HF keys vs our keys and find any mismatches."""
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    hf_keys = set(hf_model.state_dict().keys())

    config = ModelConfig.from_hf(MODEL_NAME)
    model = Decoder(config)
    our_keys = set(model.state_dict().keys())

    # Show what's in HF but not mapped to us
    print("=== HF keys ===")
    for k in sorted(hf_keys):
        print(f"  {k}")

    print(f"\n=== Our keys ===")
    for k in sorted(our_keys):
        print(f"  {k}")

    # Load and check which keys were actually loaded
    load_hf_weights(model, MODEL_NAME)
    loaded = set(model.state_dict().keys())
    rope_keys = {k for k in loaded if 'costable' in k or 'sintable' in k}

    print(f"\nHF keys: {len(hf_keys)}")
    print(f"Our keys (total): {len(our_keys)}")
    print(f"Our keys (learned): {len(our_keys - rope_keys)}")
    print(f"Our keys (RoPE buffers): {len(rope_keys)}")


def check_weight_values():
    """Compare actual weight tensor values between HF and our model."""
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    config = ModelConfig.from_hf(MODEL_NAME)
    model = Decoder(config)
    load_hf_weights(model, MODEL_NAME)

    # Check a few critical weights
    checks = [
        ("model.embed_tokens.weight", "embedding_matrix.weight"),
        ("model.layers.0.self_attn.q_proj.weight", "layers.0.attention.W_q.weight"),
        ("model.layers.0.mlp.gate_proj.weight", "layers.0.ffn.Wgate.weight"),
        ("model.norm.weight", "final_norm.gamma"),
    ]

    # Check if lm_head exists in HF
    if "lm_head.weight" in hf_model.state_dict():
        checks.append(("lm_head.weight", "lmhead.weight"))
        print("lm_head.weight EXISTS in HF state_dict")
    else:
        print("lm_head.weight MISSING from HF state_dict — likely tied to embeddings")
        # Check if our lmhead is still randomly initialized
        embed_w = model.state_dict()["embedding_matrix.weight"]
        lmhead_w = model.state_dict()["lmhead.weight"]
        print(f"  embedding_matrix.weight mean: {embed_w.mean():.6f}")
        print(f"  lmhead.weight mean: {lmhead_w.mean():.6f}")
        print(f"  Are they the same tensor? {torch.equal(embed_w, lmhead_w)}")

    for hf_key, our_key in checks:
        if hf_key not in hf_model.state_dict():
            continue
        hf_tensor = hf_model.state_dict()[hf_key]
        our_tensor = model.state_dict()[our_key]
        diff = (hf_tensor - our_tensor).abs().max().item()
        print(f"{hf_key} → {our_key}: max_diff={diff:.8f} shape={hf_tensor.shape}")


def check_layer_by_layer():
    """Run forward pass and compare intermediate outputs to find divergence."""
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    hf_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    config = ModelConfig.from_hf(MODEL_NAME)
    model = Decoder(config)
    load_hf_weights(model, MODEL_NAME)
    model.eval()

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    token_ids = inputs["input_ids"]
    T = token_ids.shape[1]
    positions = torch.arange(T)

    with torch.no_grad():
        # Compare embeddings
        hf_embeds = hf_model.model.embed_tokens(token_ids)
        our_embeds = model.embedding_matrix(token_ids)
        diff = (hf_embeds - our_embeds).abs().max().item()
        print(f"Embedding diff: {diff:.8f}")

        # Run HF full forward with output_hidden_states to get per-layer outputs
        hf_out = hf_model(token_ids, output_hidden_states=True)
        hf_logits = hf_out.logits
        hf_hidden_states = hf_out.hidden_states  # tuple of (n_layers + 1) tensors

        # Run our model layer by layer
        our_hidden = our_embeds
        for i, layer in enumerate(model.layers):
            our_hidden, _ = layer(our_hidden, positions)
            hf_layer_hidden = hf_hidden_states[i + 1]  # +1 because [0] is embeddings
            diff = (hf_layer_hidden - our_hidden).abs().max().item()
            print(f"Layer {i} output diff: {diff:.8f}")
            if diff > 0.01:
                print(f"  *** DIVERGENCE DETECTED at layer {i} ***")
                break

        # Final norm
        our_normed = model.final_norm(our_hidden)
        hf_normed = hf_model.model.norm(hf_hidden_states[-1])
        diff = (hf_normed - our_normed).abs().max().item()
        print(f"Final norm diff: {diff:.8f}")

        # Logits
        our_logits, _ = model(token_ids, positions)
        diff = (hf_logits - our_logits).abs().max().item()
        print(f"Final logits diff: {diff:.8f}")


def check_sublayer_layer0():
    """Narrow divergence within layer 0: norm → attention → residual → norm → FFN → residual."""
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    hf_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    config = ModelConfig.from_hf(MODEL_NAME)
    model = Decoder(config)
    load_hf_weights(model, MODEL_NAME)
    model.eval()

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    token_ids = inputs["input_ids"]
    T = token_ids.shape[1]
    positions = torch.arange(T)

    hf_layer = hf_model.model.layers[0]
    our_layer = model.layers[0]

    with torch.no_grad():
        x = model.embedding_matrix(token_ids)  # already verified identical

        # Step 1: attention norm
        our_normed = our_layer.attn_norm(x)
        hf_normed = hf_layer.input_layernorm(x)
        diff = (hf_normed - our_normed).abs().max().item()
        print(f"Attn norm diff: {diff:.8f}")

        # Step 2: attention Q, K, V projections
        our_q = our_layer.attention.W_q(our_normed)
        hf_q = hf_layer.self_attn.q_proj(hf_normed)
        diff = (hf_q - our_q).abs().max().item()
        print(f"Q projection diff: {diff:.8f}")

        our_k = our_layer.attention.W_k(our_normed)
        hf_k = hf_layer.self_attn.k_proj(hf_normed)
        diff = (hf_k - our_k).abs().max().item()
        print(f"K projection diff: {diff:.8f}")

        # Step 3: reshape into heads
        B = x.shape[0]
        our_q_heads = our_q.reshape(B, T, config.num_heads, config.head_dim).transpose(1, 2)
        our_k_heads = our_k.reshape(B, T, config.num_kv_heads, config.head_dim).transpose(1, 2)
        print(f"Our Q heads shape: {our_q_heads.shape}")
        print(f"Our K heads shape: {our_k_heads.shape}")

        # Step 4: RoPE — compare our RoPE output vs HF's
        our_q_rotated = our_layer.attention.rope(our_q_heads, positions)
        our_k_rotated = our_layer.attention.rope(our_k_heads, positions)

        # HF RoPE — get their rotary embedding
        hf_q_heads = hf_q.reshape(B, T, config.num_heads, config.head_dim).transpose(1, 2)
        hf_k_heads = hf_k.reshape(B, T, config.num_kv_heads, config.head_dim).transpose(1, 2)

        # HF applies RoPE internally — let's get cos/sin from their rotary_emb
        hf_rope = hf_layer.self_attn.rotary_emb
        hf_cos_sin = hf_rope(hf_k_heads, positions.unsqueeze(0))
        hf_cos, hf_sin = hf_cos_sin

        print(f"HF cos shape: {hf_cos.shape}")
        print(f"Our cos shape (from table): {our_layer.attention.rope.costable[positions].shape}")

        # Compare cos/sin values
        our_cos = our_layer.attention.rope.costable[positions]
        # HF cos might have different shape — let's check
        print(f"HF cos[0,:5]: {hf_cos[0, 0, :5]}")
        print(f"Our cos[0,:5]: {our_cos[0, :5]}")

        cos_diff = (hf_cos.squeeze() - our_cos).abs().max().item() if hf_cos.squeeze().shape == our_cos.shape else "shapes differ"
        print(f"Cos table diff: {cos_diff}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        func = sys.argv[1]
        if func == "tying":
            check_weight_tying()
        elif func == "keys":
            check_key_mapping()
        elif func == "values":
            check_weight_values()
        elif func == "layers":
            check_layer_by_layer()
        elif func == "sublayer":
            check_sublayer_layer0()
        else:
            print(f"Unknown function: {func}")
            print("Usage: python tests/debug_weight_parity.py [tying|keys|values|layers|sublayer]")
    else:
        print("Running all debug checks...")
        check_weight_tying()
        print("\n" + "="*60 + "\n")
        check_weight_values()
