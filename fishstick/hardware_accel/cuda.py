from typing import Optional, Dict, Any, List, Tuple, Callable, Union
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class CUDAMemoryPool:
    def __init__(self, device: Optional[int] = None):
        self.device = device or torch.cuda.current_device()
        self.pools: Dict[int, List[Tensor]] = {}
        self.max_pool_size = 100

    def allocate(
        self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32
    ) -> Tensor:
        key = (shape, dtype)

        if key in self.pools and self.pools[key]:
            tensor = self.pools[key].pop()
            return tensor

        tensor = torch.empty(shape, dtype=dtype, device=self.device)
        return tensor

    def release(self, tensor: Tensor) -> None:
        if tensor.numel() == 0:
            return

        key = (tuple(tensor.shape), tensor.dtype)

        if key not in self.pools:
            self.pools[key] = []

        if len(self.pools[key]) < self.max_pool_size:
            self.pools[key].append(tensor)

    def clear(self) -> None:
        self.pools.clear()

    def get_stats(self) -> Dict[str, Any]:
        total_cached = sum(
            sum(t.numel() * t.element_size() for t in pool)
            for pool in self.pools.values()
        )
        return {
            "total_cached_bytes": total_cached,
            "num_pools": len(self.pools),
            "pool_sizes": {str(k): len(v) for k, v in self.pools.items()},
        }


_memory_pool = CUDAMemoryPool()


@contextmanager
def cuda_memory_pool(enabled: bool = True):
    if not enabled or not torch.cuda.is_available():
        yield
        return

    initial_stats = _memory_pool.get_stats()
    try:
        yield _memory_pool
    finally:
        pass


class CUDAAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

        self._use_memory_efficient = True

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self._use_memory_efficient:
            attn = self._memory_efficient_attention(
                q, k, v, attn_mask, key_padding_mask
            )
        else:
            attn = self._standard_attention(q, k, v, attn_mask, key_padding_mask)

        attn = attn.transpose(1, 2).reshape(B, N, C)
        attn = self.proj(attn)
        attn = self.dropout(attn)

        return attn

    def _standard_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        scale = self.head_dim**-0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        if attn_mask is not None:
            attn = attn + attn_mask

        attn = F.softmax(attn, dim=-1)

        return torch.matmul(attn, v)

    def _memory_efficient_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        if hasattr(F, "scaled_dot_product_attention"):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )

        return self._standard_attention(q, k, v, attn_mask, key_padding_mask)


class CUDAStream:
    def __init__(self, priority: int = 0):
        self.priority = priority
        self.stream = (
            torch.cuda.Stream(priority=priority) if torch.cuda.is_available() else None
        )

    def wait_stream(self, stream: "CUDAStream") -> None:
        if self.stream and stream.stream:
            self.stream.wait_stream(stream.stream)

    def synchronize(self) -> None:
        if self.stream:
            self.stream.synchronize()

    def __enter__(self):
        if self.stream:
            self.stream.__enter__()
        return self

    def __exit__(self, *args):
        if self.stream:
            self.stream.__exit__(*args)


@dataclass
class KernelBenchmarkResult:
    name: str
    avg_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput: float
    memory_used_mb: float


def benchmark_kernel(
    kernel_fn: Callable,
    args: Tuple = (),
    kwargs: Optional[Dict] = None,
    warmup_runs: int = 10,
    benchmark_runs: int = 100,
    device: str = "cuda",
) -> KernelBenchmarkResult:
    kwargs = kwargs or {}

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    for _ in range(warmup_runs):
        _ = kernel_fn(*args, **kwargs)

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    mem_used = []

    for _ in range(benchmark_runs):
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start = (
            torch.cuda.Event(enable_timing=True)
            if device == "cuda" and torch.cuda.is_available()
            else None
        )
        end = (
            torch.cuda.Event(enable_timing=True)
            if device == "cuda" and torch.cuda.is_available()
            else None
        )

        if start and end:
            start.record()

        _ = kernel_fn(*args, **kwargs)

        if start and end:
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

            mem = torch.cuda.max_memory_allocated() / (1024**2)
            mem_used.append(mem)
        else:
            import time

            start_time = time.perf_counter()
            _ = kernel_fn(*args, **kwargs)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)
            mem_used.append(0)

    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

    return KernelBenchmarkResult(
        name=kernel_fn.__name__,
        avg_time_ms=avg_time,
        std_time_ms=std_time,
        min_time_ms=min(times),
        max_time_ms=max(times),
        throughput=1000.0 / avg_time if avg_time > 0 else 0,
        memory_used_mb=sum(mem_used) / len(mem_used),
    )


class CustomCUDAKernel:
    @staticmethod
    def matmul_kernel(a: Tensor, b: Tensor) -> Tensor:
        return torch.mm(a, b)

    @staticmethod
    def fused_gelu(x: Tensor) -> Tensor:
        return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1.0 + 0.044715 * x * x)))

    @staticmethod
    def layer_norm(
        x: Tensor,
        weight: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        eps: float = 1e-5,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) * torch.rsqrt(var + eps)

        if weight is not None:
            x_norm = x_norm * weight
        if bias is not None:
            x_norm = x_norm + bias

        return x_norm, mean.squeeze(-1), var.squeeze(-1)

    @staticmethod
    def softmax(x: Tensor, dim: int = -1) -> Tensor:
        e_x = torch.exp(x - x.max(dim=dim, keepdim=True).values)
        return e_x / e_x.sum(dim=dim, keepdim=True)


class MemoryOptimizer:
    @staticmethod
    def enable_gradient_checkpointing(model: nn.Module) -> None:
        for module in model.modules():
            if hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable()

    @staticmethod
    def clear_cache() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @staticmethod
    def get_memory_stats(device: Optional[int] = None) -> Dict[str, float]:
        if not torch.cuda.is_available():
            return {}

        if device is None:
            device = torch.cuda.current_device()

        return {
            "allocated_mb": torch.cuda.memory_allocated(device) / (1024**2),
            "reserved_mb": torch.cuda.memory_reserved(device) / (1024**2),
            "max_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
            "max_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024**2),
        }

    @staticmethod
    def set_per_process_memory_fraction(fraction: float) -> None:
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(fraction)

    @staticmethod
    def memory_efficient_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
    ) -> Tensor:
        if hasattr(F, "scaled_dot_product_attention"):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p if torch.is_grad_enabled() else 0.0,
            )

        scale = q.size(-1) ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            attn = attn + attn_mask

        attn = F.softmax(attn, dim=-1)

        if dropout_p > 0 and torch.is_grad_enabled():
            attn = F.dropout(attn, p=dropout_p)

        return torch.matmul(attn, v)


class CUDAGraphCapture:
    def __init__(self, model: nn.Module, inputs: Tuple[Tensor, ...]):
        self.model = model
        self.inputs = inputs
        self.static_inputs = None
        self.static_outputs = None
        self.graph = None
        self._captured = False

    def capture(self) -> None:
        if not torch.cuda.is_available():
            return

        self.static_inputs = tuple(
            torch.cuda.Stream().synchronize() or x.clone() for x in self.inputs
        )

        self.graph = torch.cuda.CUDAGraph()

        self.static_outputs = self.model(*self.static_inputs)
        self.model.zero_grad()

        self.graph.capture()

        self._captured = True

    def replay(self, new_inputs: Tuple[Tensor, ...]) -> Tensor:
        if not self._captured or self.static_inputs is None:
            return self.model(*new_inputs)

        for i, (static, new) in enumerate(zip(self.static_inputs, new_inputs)):
            static.copy_(new)

        self.graph.replay()

        return self.static_outputs


def optimize_cuda_operations(model: nn.Module) -> nn.Module:
    if not torch.cuda.is_available():
        return model

    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            pass

        if hasattr(module, "_forward_hook"):
            pass

    return model


def create_cuda_stream_pool(num_streams: int = 4) -> List[CUDAStream]:
    return [CUDAStream(priority=i) for i in range(num_streams)]


def get_cuda_device_properties(device: Optional[int] = None) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {}

    if device is None:
        device = torch.cuda.current_device()

    props = torch.cuda.get_device_properties(device)

    return {
        "name": props.name,
        "total_memory_mb": props.total_memory / (1024**2),
        "compute_capability": f"{props.major}.{props.minor}",
        "multi_processor_count": props.multi_processor_count,
        "max_threads_per_block": props.max_threads_per_block,
        "max_threads_per_multiprocessor": props.max_threads_per_multiprocessor,
    }
