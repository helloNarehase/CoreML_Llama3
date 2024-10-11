import torch
import torch.nn as nn
from typing import List, Optional, Tuple

class KVCache(nn.Module):

    def __init__(
        self,
        shape: Tuple[int, ...],
        device="cpu",
        dtype=torch.float32,
    ) -> None:
        """KV cache of shape (#layers, batch_size, #kv_heads, context_size, head_dim)."""
        super().__init__()
        self.past_seen_tokens: int = 0
        self.k_cache: torch.Tensor = torch.zeros(shape, dtype=dtype, device=device)
        self.v_cache: torch.Tensor = torch.zeros(shape, dtype=dtype, device=device)

    def update_and_fetch(
        self,
        k_state: torch.Tensor,
        v_state: torch.Tensor,
        layer_idx: int,
        slice_indices: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update key/value cache tensors for slice [slice_indices[0], slice_indices[1]).
        Return slice of key/value cache tensors from [0, slice_indices[1]).
        """
        if len(slice_indices) != 2:
            raise ValueError(f"Expect tuple of integers [start, end), got {slice_indices=}.")
        begin, end = slice_indices
        
        self.k_cache[layer_idx, :, begin:end] = k_state.to(self.k_cache)
        self.v_cache[layer_idx, :, begin:end] = v_state.to(self.v_cache)

        
        k_cache: torch.Tensor = self.k_cache[layer_idx, :, :end].to(k_state)
        v_cache: torch.Tensor = self.v_cache[layer_idx, :, :end].to(v_state)
        return k_cache, v_cache