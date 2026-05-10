from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class ModelConfig():

    embedding_dim: int # d_model - embedding dimension
    context_length: int
    num_heads: int # number of query heads
    num_kv_heads: int # number of kv heads
    rope_theta: float
    num_layers: int
    head_dim: int
    act_fn: str
    d_ff: int #intermediate size
    vocab_size: int
    rms_norm_eps: float
    dtype: str # torch.float32, torch.float16
    num_blocks: int
    block_size: int

    @classmethod
    def llama_3_2_3b(cls):
        return cls(
            embedding_dim=3072,
            context_length=131072,
            num_heads=24,
            num_kv_heads=8,
            rope_theta=500000.0,
            num_layers=28,
            head_dim=128,
            d_ff=8192,
            vocab_size=128256,
            rms_norm_eps=1e-05,
            act_fn='silu',
            dtype="bfloat16"
            
                    )


    @classmethod
    def from_hf(cls, model_name:str,  dtype:str = "bfloat16"):
        
        config_dict = AutoConfig.from_pretrained(model_name)


        return cls(
            embedding_dim = config_dict.hidden_size,
            head_dim = config_dict.head_dim,
            context_length = config_dict.max_position_embeddings,
            num_heads = config_dict.num_attention_heads,
            num_kv_heads = config_dict.num_key_value_heads,
            rope_theta = config_dict.rope_theta,
            d_ff = config_dict.intermediate_size,
            vocab_size = config_dict.vocab_size,
            rms_norm_eps = config_dict.rms_norm_eps,
            act_fn = config_dict.hidden_act,
            num_layers = config_dict.num_hidden_layers, 
            dtype = dtype
            )






