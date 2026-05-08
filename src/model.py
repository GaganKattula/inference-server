import torch
from torch import nn
from src.config import ModelConfig
from typing import List, Dict, Any, Optional, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer

class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float):
        """
        dim: hidden dimension size
        eps: prevent division by zero
        gamma: nn.Parameter of shape (dim,)
        
        Returns:

        x * norm * self.gamma - normalized tensor 

        """
        super().__init__()
        self.dim=dim
        self.eps=eps
        self.gamma = nn.Parameter(torch.ones(self.dim))

    def forward(self, x: torch.Tensor):

        norm = torch.rsqrt(torch.mean(x.pow(2), dim=-1, keepdim=True)+ self.eps)

        return x * norm * self.gamma

class RoPE(nn.Module):
    
    def __init__(self, head_dim: int, max_seq_len: int, rope_theta: float):

        super().__init__()
        self.head_dim=head_dim
        self.max_seq_len=max_seq_len
        self.base=rope_theta
        self.frequencies = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.positions= torch.arange(self.max_seq_len)
        self.angles= self.positions[:, None] * self.frequencies[None, :]
        cos_table = torch.cos(self.angles)
        sin_table = torch.sin(self.angles)
        cos_table = torch.cat([cos_table, cos_table], dim=-1)
        sin_table = torch.cat([sin_table, sin_table], dim=-1)
        self.register_buffer('costable', cos_table)
        self.register_buffer('sintable', sin_table)
    

    def forward(self, x: torch.Tensor, positions: torch.Tensor ): # Q or K matrices already projected to shape (B, n_heads, T, head_dim)
        
        #  fetch the cos and sin values for positions in the sequence and store in cos and sin
        cos = self.costable[positions] #(seq_len, 128)
        sin = self.sintable[positions] #(seq_len, 128)
        
        cos = cos.unsqueeze(0).unsqueeze(0)     # (1, 1, seq_len, 128) — ready to broadcast with x
        
        sin = sin.unsqueeze(0).unsqueeze(0)     # (1, 1, seq_len, 128)

        return x * cos + self._rotate_half(x) * sin

    def _rotate_half(self, x: torch.Tensor):
      
      x1 = x[..., :x.shape[-1]//2]   # first half:  [x0, x2]
      x2 = x[..., x.shape[-1]//2:]   # second half: [x1, x3]

      return torch.cat((-x2, x1), dim=-1)  # [-x1, -x3, x0, x2]
        
class AttentionGQA(nn.Module):



    def __init__(self, num_heads: int, d_model: int, num_kv_heads: int,
                  head_dim: int, max_seq_len: int, rope_theta: float):

        super().__init__()
        self.num_heads=num_heads
        self.d_model=d_model
        self.num_kv_heads=num_kv_heads
        self.head_dim=head_dim
        self.max_seq_len=max_seq_len
        self.rope_theta=rope_theta
        self.n_rep = num_heads // num_kv_heads
        self.W_q = nn.Linear(d_model, head_dim * num_heads, bias=False)
        self.W_k = nn.Linear(d_model, head_dim * num_kv_heads, bias=False) 
        self.W_v = nn.Linear(d_model, head_dim * num_kv_heads, bias=False)
        self.W_o = nn.Linear(head_dim * num_heads, d_model, bias=False)
        self.rope = RoPE(head_dim, max_seq_len, rope_theta)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):


        query = self.W_q(x) # shape ( B, T, head_dim * num_heads)
        key = self.W_k(x) # shape ( B, T, head_dim * num_kv_heads)
        value = self.W_v(x) # shape ( B, T, head_dim * num_kv_heads)

        B = x.shape[0]
        T = x.shape[1]

        query = query.reshape(B, T, self.num_heads, self.head_dim).transpose(2,1)
        key = key.reshape(B, T ,self.num_kv_heads, self.head_dim).transpose(2,1)
        value = value.reshape(B, T, self.num_kv_heads, self.head_dim).transpose(2,1)

        
        key = self.rope(key, positions)
        query = self.rope(query, positions)
        
        if cache is not None:
            cached_k, cached_v = cache
            key = torch.cat([cached_k, key], dim=2)      # append along T
            value = torch.cat([cached_v, value], dim=2)   # append along T

        updated_cache = (key, value)    # always store the full K, V for next step

        # expand key and value - repeat using the repeat factor "self.n_rep = num_heads // num_kv_heads"
        key = torch.repeat_interleave(key, self.n_rep, dim=1) # repeat along dim=1 — the heads dimension
        value = torch.repeat_interleave(value, self.n_rep, dim=1) # (B, 8, T, 128) → repeat_interleave on dim=1 → (B, 24, T, 128)

        scale = self.head_dim**0.5

        T_k = key.shape[2]   # could be longer than T if cache was used
        T_q = query.shape[2]

        # Create a matrix of -inf
        mask = torch.full((T_q, T_k), float('-inf'))
        # Zero out the lower triangle + diagonal — those are positions we CAN attend to
        """
        When T_q = 1 (decode) and T_k = 50(cached), torch.triu with diagonal=1 on a (1, 50) matrix would mask out everything except position 0.
But during decode, the single new token should attend to all previous positions. The fix: offset the
  diagonal by T_k - T_q:

        During prefill (T_q == T_k), this is diagonal=1 — same as before. During decode (T_q=1, T_k=50), this is
        diagonal=50 — nothing gets masked in a (1, 50) matrix, which is correct since the new token can see
        everything.
        """
        mask = torch.triu(mask, diagonal=T_k - T_q + 1)

        attn_scores = (query @ key.transpose(-2, -1))/ scale
        attn_scores = torch.softmax(attn_scores + mask, dim=-1)
        output = attn_scores @ value
        
        # Transpose back and Reshape to concatenate heads
        output = output.transpose(1, 2).reshape(B, T, self.d_model)

        # Apply self.W_o
        output = self.W_o(output) 


        return (output, updated_cache)
    
class SwiGLU_FFN(nn.Module):

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.Wup = nn.Linear(d_model, d_ff, bias=False)
        self.Wgate = nn.Linear(d_model, d_ff, bias=False)
        self.Wdown = nn.Linear(d_ff, d_model, bias=False)
        

    def forward(self, x: torch.Tensor):

        gate = self.Wgate(x)
        swish = torch.nn.functional.silu(gate)
        
        out = self.Wdown(swish * self.Wup(x))

        return out

class TransformerBlock(nn.Module):

    """
    Input Args:
        x - input embeddings (B, T, d_model)

    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config=config
        self.attn_norm = RMSNorm(dim=config.embedding_dim, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(dim=config.embedding_dim, eps=config.rms_norm_eps)
        self.attention = AttentionGQA(num_heads=config.num_heads, d_model=config.embedding_dim,
                                         num_kv_heads=config.num_kv_heads, head_dim=config.head_dim,
                                         max_seq_len=config.context_length, rope_theta=config.rope_theta)
        self.ffn = SwiGLU_FFN(d_model=config.embedding_dim, d_ff=config.d_ff)
        

    
    def forward(self, x,  positions, cache=None):
        
        # Normalize the input
        residual1 = x
        x = self.attn_norm(x)
        x, cache = self.attention(x, positions, cache)
        # Residual Connection
        x = x + residual1

        residual2 = x

        x = self.ffn_norm(x)
        x = self.ffn(x)
        # Residual Connection
        x = x + residual2

        out = x

        return out, cache

class Decoder(nn.Module):
    """                                     
     Layer                                           Shape

    Embedding Layer                           (B, T, d_model)
        Transfomer Blocks ( 0-27) 
            pre-norm
            attention
            residual
            ffn-norm
            ffn
                up                            (B, T, d_ff)
                down                          (B, T, d_model)
            residual             
    Final RMS Norm Layer                      (B, T, d_model)
    
    LM Head                                   (B, T, vocab_size) -> logits

    
    """

    def __init__(self,config: ModelConfig ):
        super().__init__()

        self.embedding_matrix = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.final_norm = RMSNorm(eps=config.rms_norm_eps, dim=config.embedding_dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.lmhead = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

        

    def forward(self, token_ids: torch.Tensor, positions, caches=None):
        # token_ids (B. T)
        x = self.embedding_matrix(token_ids) # input embeddings for the given sequence (B, T, d_model)

        updated_caches = []
        for i, layer in enumerate(self.layers):
            cache = caches[i] if caches is not None else None
            x, updated_cache = layer(x, positions, cache)
            updated_caches.append(updated_cache)


        x = self.final_norm(x)
        logits = self.lmhead(x)

        

        return logits, updated_caches

        


def load_hf_weights(model: Decoder, model_name: str):
    """Load HF pretrained weights into our Decoder by renaming state_dict keys.

    HF and our model have the same weights but different key names.
    This function renames HF keys to match our model, then loads them.
    """
    from transformers import AutoModelForCausalLM

    # Step 1: Download HF model and get its weights as a dictionary
    # Keys are strings like "model.layers.0.self_attn.q_proj.weight"
    # Values are the actual weight tensors
    hf_state_dict = AutoModelForCausalLM.from_pretrained(model_name).state_dict()

    # Step 2: Build a new dictionary with our key names
    new_dict = {}

    for hf_key, tensor in hf_state_dict.items():
        # Each iteration processes ONE key from HF's ~200 keys.
        # We start with the HF key and apply replacements one by one.
        # Each .replace() returns a NEW string — it never modifies the original.
        # If the substring isn't found, .replace() returns the string unchanged.

        # First: strip the "model." prefix that HF adds to everything.
        # "model.layers.0.self_attn.q_proj.weight" → "layers.0.self_attn.q_proj.weight"
        # "model.embed_tokens.weight" → "embed_tokens.weight"
        # "lm_head.weight" → "lm_head.weight" (no "model." prefix, so unchanged)
        my_key = hf_key.replace("model.", "", 1)  # replace only the first occurrence

        # Now rename the module-specific parts.
        # Only ONE of these will match for any given key. The rest do nothing.
        my_key = my_key.replace("embed_tokens", "embedding_matrix")
        my_key = my_key.replace("self_attn.q_proj", "attention.W_q")
        my_key = my_key.replace("self_attn.k_proj", "attention.W_k")
        my_key = my_key.replace("self_attn.v_proj", "attention.W_v")
        my_key = my_key.replace("self_attn.o_proj", "attention.W_o")
        my_key = my_key.replace("mlp.gate_proj", "ffn.Wgate")
        my_key = my_key.replace("mlp.up_proj", "ffn.Wup")
        my_key = my_key.replace("mlp.down_proj", "ffn.Wdown")
        my_key = my_key.replace("input_layernorm", "attn_norm")
        my_key = my_key.replace("post_attention_layernorm", "ffn_norm")
        my_key = my_key.replace("lm_head", "lmhead")

        # Final norm: after stripping "model.", the key is "norm.weight".
        # Must NOT match "attn_norm" or "ffn_norm" — only the standalone "norm.weight".
        if my_key.startswith("norm."):
            my_key = "final_norm." + my_key[len("norm."):]

        # HF norms use ".weight", our RMSNorm uses ".gamma"
        if "norm" in my_key:
            my_key = my_key.replace(".weight", ".gamma")

        # Store the tensor under our key name
        new_dict[my_key] = tensor

    # Step 3: Load into our model
    # strict=False because our model has RoPE buffers (costable, sintable)
    # that aren't in HF's weights — they're computed, not learned
    model.load_state_dict(new_dict, strict=False)

