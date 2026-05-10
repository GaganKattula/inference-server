from dataclasses import dataclass, field
from typing import List, Optional
from src.paged_cache import BlockTable, BlockAllocator
from collections import deque
import math
import torch


@dataclass
class Request:
    """
    Object representing each input sequence being passed
    for inference through the model.

    """
    request_id: int                     # — unique identifier
    prompt_tokens: List[int]            # — input token ids (the actual tensor)
    num_prompt_tokens: int              # — length of prompt
    max_new_tokens: int                 # — how many tokens to generate
    arrival_time: float                 # — for priority/fairness decisions
    block_table: Optional[BlockTable] = None             # — BlockTable instance, assigned at admission
    tokens_generated: int = 0               # — int, starts at 0, increments each decode step
    status: str = "waiting"                         # — "waiting" | "running" | "finished"
    output_token_ids: List[int]= field(default_factory=list) # - default_factory=list creates a fresh empty list for each instance.

class Scheduler:
    """
    Three data structures to maintain:
    - waiting queue to hold incoming requests to be processed
    - running list of requests admitted and being processed
    - allocator instance to create blocks for new requests : manage physical block pool

    Args:
    - token_ids
    - positionsLis
    - block_tables - list of blocktables (1 per running request)

    Function:
        method called every iteration - step()
        step(): server loop calls step(), feeds those three things to the model, gets logits back, then calls
            another scheduler method to process the results (update caches, record generated tokens, mark finished
            requests). 
    
    """

    def __init__(self, block_size: int, num_blocks: int):
  
        self.num_blocks=num_blocks
        self.block_size=block_size
        self.wait_queue=deque()
        self.running_list=[]
        self.allocator=BlockAllocator(self.num_blocks)
        
    def step(self):

        # ADMIT REQUESTS
        while self.wait_queue:
            next_req = self.wait_queue[0]          # peek without popping
            blocks_needed = math.ceil(next_req.num_prompt_tokens / self.block_size)
            if blocks_needed > self.allocator.num_free:
                break                               # not enough memory, stop admitting
            curr_req = self.wait_queue.popleft()   # now actually pop
            table = BlockTable(self.block_size)
            for _ in range(blocks_needed):
                table.add_block(self.allocator.allocate())
            curr_req.block_table = table
            curr_req.status = "running"
            # after allocating prompt blocks, the last block isn't empty — it
            # has num_prompt_tokens % block_size tokens already written into it
            table.num_tokens_in_lastblock = curr_req.num_prompt_tokens % self.block_size
            self.running_list.append(curr_req)

        # BUILD THE BATCH
        """
        what makes up a batch
        """

        batch_token_ids = []
        batch_positions = []
        # blocktables is rebuilt from scratch on every call to step()
        # local variable, not a persistent list.
        batch_tables = []

        for req in self.running_list:

            if req.tokens_generated == 0:
                # PREFILL
                token_ids = torch.tensor(req.prompt_tokens)
                positions = torch.arange(req.num_prompt_tokens)
            
                    # assemble batch
                batch_token_ids.append(token_ids)
                batch_positions.append(positions)
                batch_tables.append(req.block_table)

            elif req.tokens_generated < req.max_new_tokens:
                token_ids = torch.tensor([req.output_token_ids[-1]]) #(last generated token only)
                positions = torch.tensor([req.num_prompt_tokens + req.tokens_generated - 1])
                    # assemble batch
                batch_token_ids.append(token_ids)                
                batch_positions.append(positions)
                batch_tables.append(req.block_table)

        return batch_token_ids, batch_positions, batch_tables
            
    def update(self, logits: torch.Tensor):
        """
        1. Sample next token: argmax(logits) → token ID
        2. Append to output_token_ids
        3. Increment tokens_generated
        4. Call block_table.append_token() — if it returns True, allocate a new block
        5. Check if tokens_generated == max_new_tokens → if so, set status = "finished", free blocks, remove
            from running_list
        """

        for i, req in enumerate(self.running_list):

            next_token_id = logits[i, -1, :].argmax().item()  # -1: last position (prefill or decode)
            req.tokens_generated += 1

            if req.block_table.append_token():
                req.block_table.add_block(self.allocator.allocate())

            req.output_token_ids.append(next_token_id)

            
            if req.tokens_generated == req.max_new_tokens:
                req.status = "finished"
        finished = [req for req in self.running_list if req.status == "finished"]
        for req in finished:
            for block in req.block_table.get_all_blocks():
                self.allocator.free(block)
            self.running_list.remove(req)

        return 
    
    def add_request(self, request: Request):
        """
        Server needs a way to submit incoming requests to the scheduler
        """
        self.wait_queue.append(request)
        return