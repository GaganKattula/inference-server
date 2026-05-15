

from typing import List, Tuple
from src.scheduler import Request
import torch
import uuid
import time

def generate_request(lam: float, vocab_size: int, num_requests: int, prompt_len_range: Tuple[int, int], max_tokens_new: int, seed: int) -> List[Tuple[float, Request]]:
    
    torch.manual_seed(seed)

    U = torch.rand(num_requests)
    gaps = -torch.log(U) / lam
    
    arrival_times = torch.cumsum(gaps, dim=0)
    
    
    requests = []

    for i in range(num_requests):

        
        prompt_len = torch.randint(low=prompt_len_range[0], high=prompt_len_range[1] + 1, size=(1,)).item()
        token_ids = torch.randint(size=(prompt_len,), high=vocab_size).tolist()
        request_id = uuid.uuid4().hex
        max_new_tokens = torch.randint(low=1,size=(1,), high=max_tokens_new+1) # prevent zero sampling | .randint samples from [0, N)
        num_prompt_tokens = prompt_len

        req = Request(
        request_id=request_id,
        prompt_tokens=token_ids,
        num_prompt_tokens=num_prompt_tokens,
        max_new_tokens=max_new_tokens.item(),
        arrival_time=arrival_times[i].item()
                    )
        requests.append(req)
        
    return [(arrival_times[i].item(), req) for i, req in enumerate(requests)]
    
