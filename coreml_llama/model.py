import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple

from dataclasses import dataclass

from .llamaCache import KVCache

@dataclass
class ModelArgs:
    dim: int = 2048
    n_layers: int = 16
    n_heads: int = 32
    n_kv_heads: Optional[int] = 8
    vocab_size: int = 128256
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = 1.5
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0

    max_batch_size: int = 1
    max_seq_len: int = 2048
    use_scaled_rope:bool = True

def precompute_freqs_cis_real(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    
    # 각도를 기반으로 cos과 sin을 계산
    cos_vals = torch.cos(freqs).to(torch.float16)
    sin_vals = torch.sin(freqs).to(torch.float16)
    
    return cos_vals, sin_vals

def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    
    shape = [1] + [x.shape[1], 1, x.shape[-1] // 2]
    return freqs_cis.view(*shape)

def apply_rotary_emb_with_real(
    xq: torch.Tensor,
    cos_vals: torch.Tensor,
    sin_vals: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq와 xk의 마지막 차원을 2로 나누어 real과 imag 부분으로 나눔
    xq_real = xq[..., 0::2]
    xq_imag = xq[..., 1::2]
    # cos_vals와 sin_vals를 broadcast 가능하게 변환
    # print(xq_real.shape)
    cos_vals = _reshape_for_broadcast(cos_vals, xq)
    sin_vals = _reshape_for_broadcast(sin_vals, xq)

    # 복소수 곱셈 구현 (실수 및 허수 계산)
    xq_out_real = xq_real * cos_vals - xq_imag * sin_vals
    xq_out_imag = xq_real * sin_vals + xq_imag * cos_vals

    # 다시 실수 및 허수 부분을 합쳐서 반환
    xq_out = torch.cat([xq_out_real, xq_out_imag], dim=-1)

    return xq_out.type_as(xq)

class RoPE(nn.Module):
    def __init__(
        self,
        dim: int,
        end: int,
        theta: float = 500000.0
    ):
        super().__init__()
        self.dim = dim
        self.end = end
        self.theta = theta

        magnitudes, angles = precompute_freqs_cis_real(dim, end, theta)
        self.register_buffer('magnitudes', magnitudes)
        self.register_buffer('angles', angles)
        

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        """쿼리와 키 텐서에 rotary embedding을 적용"""
        # print(x.shape, start_pos)
        _bsz, seqlen, _, _ = x.shape
        
        # 필요한 주파수 값을 슬라이스로 추출
        magnitudes = self.magnitudes[start_pos : start_pos + seqlen]
        angles = self.angles[start_pos : start_pos + seqlen]
        
        x_out = apply_rotary_emb_with_real(x, magnitudes, angles)
        return x_out

class TrainedRoPE(RoPE):
    def __init__(
        self,
        dim: int,
        end: int,
        theta: float = 500000.0
    ):
        super().__init__(dim, end, theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """쿼리와 키 텐서에 rotary embedding을 적용"""
        # print(x.shape, start_pos)
        _bsz, seqlen, _, _ = x.shape
        
        # 필요한 주파수 값을 슬라이스로 추출
        magnitudes = self.magnitudes[: seqlen]
        angles = self.angles[: seqlen]
        
        x_out = apply_rotary_emb_with_real(x, magnitudes, angles)
        return x_out


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def make_mask(max_seq_len = 100, seq_len = 10):
    mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len), -2.3819763e38).to(torch.float16)
    mask_tensor = torch.triu(mask_tensor, diagonal=1)
    input_positions_tensor = torch.arange(0, seq_len,
                                            dtype=torch.int64)
    curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
    return curr_mask_tensor


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads


        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )
        self.rope = RoPE(
            args.dim // args.n_heads,
            args.max_seq_len * 2,
            args.rope_theta,
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        layer_idx: int,
        cache_kv: KVCache,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq = self.rope(xq, start_pos)
        xk = self.rope(xk, start_pos)

        keys, values = cache_kv.update_and_fetch(
            xk,
            xv,
            layer_idx,
            slice_indices=(start_pos, start_pos + seqlen),
        )

        
        # keys = cache_k[layer_idx, :, : start_pos + seqlen].to(xq)
        # values = cache_v[layer_idx, :, : start_pos + seqlen].to(xq)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)

        output = torch.nn.functional.scaled_dot_product_attention(
            xq,
            keys,
            values,
            attn_mask=mask.to(xq),
        )
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class Trained_Attention(Attention):
    def __init__(self, args: ModelArgs):
        super().__init__(args)

        self.rope = TrainedRoPE(
            args.dim // args.n_heads,
            args.max_seq_len * 2,
            args.rope_theta,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq = self.rope(xq)
        xk = self.rope(xk)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)

        output = torch.nn.functional.scaled_dot_product_attention(
            xq,
            keys,
            values,
            attn_mask=mask.to(xq),
        )
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim, hidden_dim, bias=False, 
        )
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=False, 
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False, 
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    
class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        layer_idx: int,
        cache_kv: KVCache,
        mask: Optional[torch.Tensor],
    ):
        # h = x
        h = torch.add(x, self.attention(self.attention_norm(x), start_pos, layer_idx, cache_kv, mask))
        h = torch.clamp(h, min=-1e9, max=1e9)
        
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
    

class Trained_TransformerBlock(TransformerBlock):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__(layer_id, args)
        self.attention = Trained_Attention(args)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = torch.add(x, self.attention(self.attention_norm(x), mask))
        h = torch.clamp(h, min=-1e9, max=1e9)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.maxlen = params.max_seq_len
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(
            params.vocab_size, params.dim
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(
            params.dim, params.vocab_size, bias=False, 
        )

        self.n_kv_heads = params.n_heads if params.n_kv_heads is None else params.n_kv_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.head_dim = params.dim // params.n_heads
        
        self.caches_shape = (
            params.n_layers,
            params.max_batch_size,
            params.max_seq_len,
            self.n_local_kv_heads,
            self.head_dim,
        )

        self.Kcache = torch.zeros(self.caches_shape, dtype=torch.float32)
        self.Vcache = torch.zeros(self.caches_shape, dtype=torch.float32)

        self.register_buffer("cache_k_s", self.Kcache)
        self.register_buffer("cache_v_s", self.Vcache)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor, cache:KVCache):
        offset = mask.shape[-1] - tokens.size(1)
        h = self.tok_embeddings(tokens)
        for idx in range(self.params.n_layers):
            layer = self.layers[idx]
            h = layer(h, offset, idx, cache, mask)
        h = self.norm(h)
        logits = self.output(h)
        return logits.float()
    

class Trained_Transformer(Transformer):
    def __init__(self, params: ModelArgs):
        super().__init__(params)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(Trained_TransformerBlock(layer_id, params))

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        h = self.tok_embeddings(tokens)
        for idx in range(self.params.n_layers):
            layer = self.layers[idx]
            h = layer(h, mask)
        h = self.norm(h)
        logits = self.output(h)
        return logits.float()

class Llama_CoreML(nn.Module):
    def __init__(self, transformer:Transformer) -> None:
        super().__init__()
        self.transformer = transformer
        
        self.kv_cache = KVCache(self.transformer.caches_shape, dtype=torch.float16)

        self.register_buffer("keyCache", self.kv_cache.k_cache)
        self.register_buffer("valueCache", self.kv_cache.v_cache)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        logits = self.transformer(tokens, mask, self.kv_cache)
        return logits[:, -1, :]