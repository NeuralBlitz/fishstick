from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class CUDAKernelConfig:
    block_size: int = 256
    grid_size: Optional[int] = None
    shared_memory: int = 0
    num_streams: int = 1


class CudaKernel:
    def __init__(self, name: str, config: Optional[CUDAKernelConfig] = None):
        self.name = name
        self.config = config or CUDAKernelConfig()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f"Kernel {self.name} not implemented")


class CustomCUDAOp(nn.Module):
    def __init__(
        self,
        op_name: str,
        forward_fn: Optional[Callable] = None,
        backward_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.op_name = op_name
        self.forward_fn = forward_fn
        self.backward_fn = backward_fn

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.forward_fn:
            return self.forward_fn(*args, **kwargs)
        raise NotImplementedError(f"Forward not defined for {self.op_name}")


class MemoryOptimizer:
    def __init__(self, model: nn.Module):
        self.model = model
        self.original_buffers: Dict[str, Tensor] = {}
        self.saved_tensors: List[Tensor] = []

    def optimize_memory(
        self,
        strategy: str = "checkpoint",
        checkpoint_ratio: float = 0.5,
    ) -> nn.Module:
        if strategy == "checkpoint":
            return self.apply_gradient_checkpointing(checkpoint_ratio)
        elif strategy == "offload":
            return self.apply_cpu_offload()
        elif strategy == "pin_memory":
            return self.apply_pinned_memory()
        return self.model

    def apply_gradient_checkpointing(self, ratio: float = 0.5) -> nn.Module:
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        return self.model

    def apply_cpu_offload(self) -> nn.Module:
        for module in self.model.modules():
            if hasattr(module, "to"):
                module.to("cpu")
        return self.model

    def apply_pinned_memory(self) -> nn.Module:
        for param in self.model.parameters():
            if param.is_cuda:
                param.data = param.data.pin_memory()
        return self.model


class CudaMemoryPool:
    def __init__(self, device: int = 0):
        self.device = device
        self.pools: Dict[str, List[Tensor]] = {}
        self.stats = {
            "allocated": 0,
            "reserved": 0,
            "peak_allocated": 0,
        }

    def allocate(self, key: str, shape: Tuple[int, ...], dtype: torch.dtype) -> Tensor:
        if key not in self.pools or not self.pools[key]:
            tensor = torch.empty(shape, dtype=dtype, device=f"cuda:{self.device}")
            self.pools[key] = [tensor]
            return tensor

        tensor = self.pools[key].pop()
        if tensor.shape != shape or tensor.dtype != dtype:
            tensor = torch.empty(shape, dtype=dtype, device=f"cuda:{self.device}")
        return tensor

    def release(self, key: str, tensor: Tensor) -> None:
        if key not in self.pools:
            self.pools[key] = []
        self.pools[key].append(tensor)

    def clear(self) -> None:
        for key in self.pools:
            self.pools[key].clear()
        torch.cuda.empty_cache()

    def get_stats(self) -> Dict[str, int]:
        self.stats["allocated"] = torch.cuda.memory_allocated(self.device)
        self.stats["reserved"] = torch.cuda.memory_reserved(self.device)
        self.stats["peak_allocated"] = torch.cuda.max_memory_allocated(self.device)
        return self.stats


def optimize_cuda_memory(
    model: nn.Module,
    strategy: str = "default",
    **kwargs: Any,
) -> nn.Module:
    if not next(model.parameters(), None).is_cuda:
        model = model.cuda()

    if strategy == "memory_efficient":
        if hasattr(torch, "compile"):
            model = torch.compile(model, mode="reduce-overhead")
    elif strategy == "fast":
        if hasattr(torch, "compile"):
            model = torch.compile(model, mode="default")

    torch.cuda.empty_cache()
    return model


def clear_cuda_cache() -> None:
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def get_cuda_info(device: Optional[int] = None) -> Dict[str, Any]:
    if device is None:
        device = torch.cuda.current_device()

    return {
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": torch.cuda.get_device_capability(device),
        "memory_allocated": torch.cuda.memory_allocated(device),
        "memory_reserved": torch.cuda.memory_reserved(device),
        "max_memory_allocated": torch.cuda.max_memory_allocated(device),
        "max_memory_reserved": torch.cuda.max_memory_reserved(device),
        "total_memory": torch.cuda.get_device_properties(device).total_memory,
    }


class CudaStream:
    def __init__(self, device: int = 0):
        self.device = device
        self.stream = torch.cuda.Stream(device)

    def wait(self, event: "CudaEvent") -> None:
        self.stream.wait_event(event.event)

    def wait_stream(self, other: "CudaStream") -> None:
        self.stream.wait_stream(other.stream)

    def __enter__(self):
        self.stream.__enter__()
        return self

    def __exit__(self, *args: Any):
        self.stream.__exit__(*args)


class CudaEvent:
    def __init__(self, enable_timing: bool = False, device: int = 0):
        self.device = device
        self.event = torch.cuda.Event(enable_timing=enable_timing)

    def record(self, stream: Optional[CudaStream] = None) -> None:
        if stream:
            self.event.record(stream.stream)
        else:
            self.event.record()

    def elapsed_time(self, end: "CudaEvent") -> float:
        return self.event.elapsed_time(end.event)


class MemoryEfficientAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if hasattr(F, "scaled_dot_product_attention"):
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            scale = self.head_dim**-0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = F.softmax(attn, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            attn = attn @ v

        x = attn.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class CudaGraphCapture:
    def __init__(self, model: nn.Module, example_inputs: Any):
        self.model = model
        self.example_inputs = example_inputs
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.static_inputs: Optional[Tensor] = None
        self.static_outputs: Optional[Tensor] = None

    def capture(self) -> None:
        self.model.eval()
        self.static_inputs = self.example_inputs
        if isinstance(self.example_inputs, (list, tuple)):
            self.static_inputs = tuple(
                x.clone() if isinstance(x, Tensor) else x for x in self.example_inputs
            )
        elif isinstance(self.example_inputs, Tensor):
            self.static_inputs = self.example_inputs.clone()

        self.graph = torch.cuda.CUDAGraph()
        self.static_outputs = self.model(*self.static_inputs)
        self.graph.capture_begin()
        self.static_outputs = self.model(*self.static_inputs)
        self.graph.capture_end()

    def replay(self, inputs: Any) -> Tensor:
        if isinstance(inputs, (list, tuple)):
            for src, dst in zip(inputs, self.static_inputs):
                if isinstance(src, Tensor) and isinstance(dst, Tensor):
                    dst.copy_(src)
        elif isinstance(inputs, Tensor) and isinstance(self.static_inputs, Tensor):
            self.static_inputs.copy_(inputs)

        self.graph.replay()
        return self.static_outputs


def register_custom_op(
    name: str,
    forward_fn: Callable,
    backward_fn: Optional[Callable] = None,
) -> Callable:
    class CustomOp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args, **kwargs):
            return forward_fn(*args, **kwargs)

        @staticmethod
        def backward(ctx, *grad_outputs):
            if backward_fn:
                return backward_fn(*grad_outputs)
            return grad_outputs

    return CustomOp.apply
