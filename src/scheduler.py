from dataclasses import dataclass, field
from typing import List, Optional
from src.paged_cache import BlockTable, BlockAllocator
from collections import deque
import math, time
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
    block_table: Optional[BlockTable]=None  # — BlockTable instance, assigned at admission
    tokens_generated: int = 0               # — int, starts at 0, increments each decode step
    status: str = "waiting"                 # — "waiting" | "running" | "finished"
    output_token_ids: List[int]= field(default_factory=list) # - default_factory=list creates a fresh empty list for each instance.
    tokens_processed: int = 0               # - tracks how many prompt tokens have been processed so far, starting at 0 and incrementing by chunk size each step
    first_token_time: Optional[float]=None  # - when update() processes the first completed prefill
    finish_time: Optional[float]=None       # - when req.status == "finished" inside update()

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

    def __init__(self, block_size: int, num_blocks: int, chunk_size: int, cache_type: str, static_batching: bool=False ):
  
        self.num_blocks=num_blocks
        self.block_size=block_size
        self.chunk_size=chunk_size
        self.wait_queue=deque()
        self.running_list=[]
        self.allocator=BlockAllocator(self.num_blocks)
        self.cache_type=cache_type
        self.static_batching=static_batching

        
    def step(self):
        """
        - Admits requests from wait_queue
        - Allocated blocks 
        - Builds batch - token ids, positions, block_tables (flat batch - prefill+decode )
        - Returns the batch
        """

        if not self.static_batching or   not self.running_list: #checks for static batching flag
        # true is continuous barching  |  true is running list is empty  
        # runs this branch for continuous 
            if self.cache_type == "paged":
                # continuous batching + paged cache AND static batching + paged cache paths



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

            elif self.cache_type == "contiguous":

                while self.wait_queue:
                    curr_req = self.wait_queue.popleft()
                    curr_req.status = "running"
                    self.running_list.append(curr_req)  
                    #print(f"SCHEDULER ADMITTED: {curr_req.request_id[:8]}")

        # BUILD THE BATCH
        """
        what makes up a batch
        """

        #batch_token_ids = []
        prefill_batch_token_ids = []
        #batch_positions = []
        prefill_batch_positions = []

        # blocktables is rebuilt from scratch on every call to step()
        # local variable, not a persistent list.
        #batch_tables = []
        prefill_batch_tables = []

        decode_batch_token_ids = []
        decode_batch_positions = []
        decode_batch_tables = []
        prefill_req = []
        decode_req = []

        p_batches = ([], [], [])
        d_batches = ([], [], [])

        for req in self.running_list:
            #if req.request_id[:8] in ['911012a9', 'b6346e8d']:
                #print(f"CHECK {req.request_id[:8]}: processed={req.tokens_processed}, prompt={req.num_prompt_tokens}, generated={req.tokens_generated}, max={req.max_new_tokens}, status={req.status}")

            if req.tokens_processed < req.num_prompt_tokens: # filter prefill requests
                # PREFILL
                chunk_end = min(req.tokens_processed + self.chunk_size, req.num_prompt_tokens)
                token_ids = torch.tensor(req.prompt_tokens[req.tokens_processed: chunk_end ])
                positions = torch.arange(req.tokens_processed, chunk_end)
            
                    # assemble batch
                prefill_batch_token_ids.append(token_ids)
                prefill_batch_positions.append(positions)
                prefill_batch_tables.append(req.block_table)

                req.tokens_processed = chunk_end 
                prefill_req.append(req)
                

            elif req.tokens_processed == req.num_prompt_tokens and req.tokens_generated < req.max_new_tokens: # filter decode requests
                #token_ids = torch.tensor([req.output_token_ids[-1]]) #(last generated token only)
                if req.tokens_generated == 0:
                    token_ids = torch.tensor([req.prompt_tokens[-1]])
                else:
                    token_ids = torch.tensor([req.output_token_ids[-1]])
                positions = torch.tensor([req.num_prompt_tokens + req.tokens_generated - 1])
                    # assemble batch
                decode_batch_token_ids.append(token_ids)                
                decode_batch_positions.append(positions)
                decode_batch_tables.append(req.block_table)

                decode_req.append(req)

                


            p_batches = (prefill_batch_token_ids, prefill_batch_positions, prefill_batch_tables)
            d_batches = (decode_batch_token_ids, decode_batch_positions, decode_batch_tables)

        return p_batches, d_batches, prefill_req, decode_req
            
    def update(self, logits: torch.Tensor, requests: List[Request]):

        """
        1. Sample next token: argmax(logits) → token ID
        2. Append to output_token_ids
        3. Increment tokens_generated
        4. Call block_table.append_token() — if it returns True(if block is full), allocate a new block
        5. Check if tokens_generated == max_new_tokens -> if so, set status = "finished", 
                                                                free blocks, 
                                                                remove from running_list                                                       
        """

        for i, req in enumerate(requests):
            
            if req.tokens_processed < req.num_prompt_tokens:
                
                continue  # mid-prefill chunk, no token generated yet
            #print(f"UPDATE {req.request_id[:8]}: processed={req.tokens_processed}, prompt={req.num_prompt_tokens}, generated={req.tokens_generated}, max={req.max_new_tokens}")

            next_token_id = logits[i, -1, :].argmax().item()  # -1: last position (prefill or decode)
            req.tokens_generated += 1

            if self.cache_type == "paged" and req.block_table:
                if req.block_table.append_token():
                    req.block_table.add_block(self.allocator.allocate())

            req.output_token_ids.append(next_token_id)

            if req.tokens_generated == 1: # used to calculate TTFT
                req.first_token_time = time.time() # first token pred - first decode step 
             
            

            
            if req.tokens_generated == req.max_new_tokens:
                req.status = "finished"
                req.finish_time = time.time()
        finished = [req for req in self.running_list if req.status == "finished"]
        for req in finished:
            if req.block_table: # only free blocks for paged cache
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