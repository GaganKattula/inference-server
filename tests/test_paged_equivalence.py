from src.paged_cache import BlockAllocator, BlockTable, attention_kernel
import torch

def test_attention_kernel():

    # input params
    seq_len = 32
    block_size = 16
    n_heads = 4
    n_kv_heads = 2
    head_dim = 8
    num_blocks = 10
    scale = head_dim ** 0.5
    n_rep =  n_heads // n_kv_heads

    # Populate block table
    blocktable = BlockTable(block_size)
    allocator = BlockAllocator(num_blocks)
    """
    for a seq_len - 32
    store all 32 token in logical -> physical block mapping table
    1 block is 16 tokens -> 32 tokens is 2 blocks
    
    """

    block0 = allocator.allocate()
    block1 = allocator.allocate()

    blocktable.add_block(block0)
    blocktable.add_block(block1)


   
    
    # Initalize K and V matrices
    K = torch.randn((seq_len, n_kv_heads, head_dim))
    V = torch.randn((seq_len, n_kv_heads, head_dim))
    
    Q = torch.randn((1, n_heads, 1, head_dim))
    # Initialize KV cache
    K_cache = torch.zeros(num_blocks, block_size, n_kv_heads, head_dim)
    V_cache = torch.zeros(num_blocks, block_size, n_kv_heads, head_dim)

    token_indices = torch.arange(0,seq_len)
    logical_blocks = token_indices // block_size

    slots = token_indices % block_size
    physical_blocks = torch.tensor([blocktable[i.item()] for i in logical_blocks])

 
    # Store KV cache in phycical blocks
    K_cache[physical_blocks, slots, :, :] = K[ token_indices, :, : ]
    V_cache[physical_blocks, slots, :, :] = V[ token_indices, :, : ]


    # Contiguous approach

    K_contiguous = K.unsqueeze(0).transpose(1, 2)  # (1, n_kv_heads, seq_len, head_dim)
    V_contiguous = V.unsqueeze(0).transpose(1, 2)  # (1, n_kv_heads, seq_len, head_dim)
    

    K_contiguous = torch.repeat_interleave(K_contiguous, n_rep, dim=1)
    V_contiguous = torch.repeat_interleave(V_contiguous, n_rep, dim=1)

    contiguous_attn_scores = torch.softmax((Q @ K_contiguous.transpose(-2,-1) / scale), dim=-1)
    contiguous_output = contiguous_attn_scores @ V_contiguous

    # Paged setup

    paged_output, paged_attn_scores = attention_kernel(Q, K_cache, V_cache, seq_len=seq_len, 
                                                       block_size=block_size, block_table=blocktable)
    

    # Assert both outputs for equivalence
    
    assert torch.allclose(contiguous_output, paged_output, atol=1e-5)

