import torch
from typing import List, Tuple


def collate_batch(
    token_ids_list: List[torch.Tensor],
    pad_value: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length token sequences to a uniform batch.

    Matches the HuggingFace collator pattern: padding happens outside the model,
    the model receives a pre-padded tensor + attention_mask.

    Args:
        token_ids_list: List of 1-D token tensors, each shape (T_i,). Variable length.
        pad_value: Token ID used for padding positions (default 0).

    Returns:
        token_ids:      (B, T_max) — padded token IDs, right-padded with pad_value
        attention_mask: (B, T_max) — 1 for real tokens, 0 for padding positions

    Shape note:
        T_max = max sequence length in the batch.
        Single-sequence input is a no-op: B=1, T_max=T, no padding added.
    """
    T_max = max(t.shape[0] for t in token_ids_list)

    padded_ids = []
    masks = []

    for t in token_ids_list:
        pad_len = T_max - t.shape[0]
        # Right-pad token IDs with pad_value
        padded_ids.append(torch.cat([t, torch.full((pad_len,), pad_value, dtype=t.dtype)]))
        # 1 for real tokens, 0 for padding
        masks.append(torch.cat([torch.ones(t.shape[0], dtype=torch.bool),
                                 torch.zeros(pad_len, dtype=torch.bool)]))

    return torch.stack(padded_ids), torch.stack(masks)  # (B, T_max), (B, T_max)
