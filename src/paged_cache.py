from collections import deque
import torch

class BlockAllocator():
    """
    Manages the physical pool; Maintains a free list;
    Allocates a block- returns an index; Frees a block- returns index to free list

    - free_list: list of available physical block indices [0, 1, 2, ..., N-1] initially
    - allocate() → returns one block index, removes it from free_list
    - free(block_idx) → returns block index to free_list
    """
    def __init__(self, num_blocks: int):

        self.num_blocks=num_blocks
        self.free_list=deque(range(num_blocks))      
    def allocate(self):

        if len(self.free_list) == 0:
            raise RuntimeError("BlockAllocator: out of free blocks")
        
        return self.free_list.pop()
    def free(self, block_idx):

        self.free_list.append(block_idx)
    @property
    def num_free(self): # check how many blocks are available
        return len(self.free_list)
    
class BlockTable():

    """
    Maps logical block index -> physcial block index
    Tracks how many tokens are in the current block

    
    """
    def __init__(self, block_size: int):
        self.block_size=block_size
        self.mapper=dict()
        self.num_tokens_in_lastblock=0


    def add_block(self, physical_block_idx: int):
        # scheduler calls this with a freshly allocated physical index
        # Appends it to the mapping, resets the token counter for the new block
        logical_idx = len(self.mapper)  # next available logical index
        self.mapper[logical_idx] = physical_block_idx # map the next sequential logical idx to physical block
        self.num_tokens_in_lastblock = 0 # num_tokens_in_lastblock should reset to 0

    def append_token(self):
        # called once per generated token
        # Increments num_tokens_in_last_block
        self.num_tokens_in_lastblock+=1 # adds token to block
        if self.num_tokens_in_lastblock == self.block_size: # checks if block size is reached
            return True
        return False

    def get_all_blocks(self) -> list[int]:

        return list(self.mapper.values())

    def __getitem__(self,logical_idx: int) -> int:
        """
         returns physical block index for logical block i. 

         Using __getitem__ lets the attention kernel write block_table[i] which is clean.
        """
        return self.mapper[logical_idx]
    


"""
physical_block = block_table[t // block_size]
slot = t % block_size
k_vec = K_cache[physical_block, slot, :, :]  # (n_kv_heads, head_dim)

  gathered_k = []
  for t in range(seq_len):
      physical_block = block_table[t // block_size]
      slot = t % block_size
      gathered_k.append(K_cache[physical_block, slot, :, :])
"""

def attention_kernel(Q, K_cache, V_cache, seq_len: int
                    , block_size: int, block_table: BlockTable):
    
    _, num_heads, _, _ = Q.shape
    

    # Build index arrays
    token_indices = torch.arange(0,seq_len)
    logical_blocks = token_indices // block_size

    slots = token_indices % block_size
    physical_blocks = torch.tensor([block_table[i.item()] for i in logical_blocks])

    # single indexing operation to get K and V in a contiguous tensor
    K = K_cache[physical_blocks, slots, :, :] #(seq_len, n_kv_heads, head_dim)
    V = V_cache[physical_blocks, slots, :, :]
    # Q is of shape  (1, n_heads, 1, head_dim)
    _, n_kv_heads, head_dim = K.shape
    # K = K.reshape(1, n_kv_heads, seq_len, head_dim)
    K = K.unsqueeze(0).transpose(1, 2)  # (seq_len, n_kv_heads, head_dim) → (1, n_kv_heads, seq_len,head_dim)
    # V = V.reshape(1, n_kv_heads, seq_len, head_dim)
    V = V.unsqueeze(0).transpose(1, 2)  # (seq_len, n_kv_heads, head_dim) → (1, n_kv_heads, seq_len,head_dim)

    n_rep = num_heads // n_kv_heads
    K = torch.repeat_interleave(K, n_rep, dim=1)
    V = torch.repeat_interleave(V, n_rep, dim=1)

    scale = head_dim ** 0.5
    attn_scores = torch.softmax((Q @ K.transpose(-2,-1))/ scale, dim = -1)# (B, T, T)

    output = attn_scores @ V

    return output, attn_scores

    

    

