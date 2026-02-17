"""
Pretrained Model Registry

Central registry for all pretrained models.
"""

from typing import Dict, Any, Optional, Callable, Union
import torch
from torch import nn


class ModelInfo:
    """Information about a pretrained model."""

    def __init__(
        self,
        name: str,
        framework: str,
        task: str,
        architecture: str,
        num_params: int,
        input_dim: int,
        output_dim: int,
        description: str,
        paper: str,
        weights_url: Optional[str] = None,
    ):
        self.name = name
        self.framework = framework
        self.task = task
        self.architecture = architecture
        self.num_params = num_params
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.description = description
        self.paper = paper
        self.weights_url = weights_url


class LLMInfo:
    """Information about an LLM."""

    def __init__(
        self,
        name: str,
        model_id: str,
        provider: str,
        num_params: int,
        context_length: int,
        description: str,
        quantization: Optional[str] = None,
    ):
        self.name = name
        self.model_id = model_id
        self.provider = provider
        self.num_params = num_params
        self.context_length = context_length
        self.description = description
        self.quantization = quantization


class LLMRegistry:
    """Registry for Large Language Models."""

    _registry: Dict[str, LLMInfo] = {}

    @classmethod
    def register(cls, llm_info: LLMInfo) -> None:
        """Register an LLM."""
        cls._registry[llm_info.name] = llm_info

    @classmethod
    def get(cls, name: str) -> Optional[LLMInfo]:
        """Get LLM info by name."""
        return cls._registry.get(name)

    @classmethod
    def list_llms(cls, provider: Optional[str] = None) -> list:
        """List available LLMs."""
        llms = list(cls._registry.values())
        if provider:
            llms = [m for m in llms if m.provider == provider]
        return llms

    @classmethod
    def load(cls, name: str, **kwargs) -> Any:
        """Load an LLM."""
        llm_info = cls.get(name)
        if llm_info is None:
            raise ValueError(f"LLM {name} not found in registry")

        if llm_info.provider == "huggingface":
            return cls._load_huggingface(llm_info, **kwargs)
        elif llm_info.provider == "ollama":
            return cls._load_ollama(llm_info, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {llm_info.provider}")

    @classmethod
    def _load_huggingface(cls, llm_info: LLMInfo, **kwargs) -> Any:
        """Load from HuggingFace."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("transformers required: pip install transformers")

        device = kwargs.get("device", "auto")
        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)

        tokenizer = AutoTokenizer.from_pretrained(llm_info.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            llm_info.model_id,
            torch_dtype=torch_dtype,
            device_map=device,
            **kwargs,
        )
        return {"model": model, "tokenizer": tokenizer}

    @classmethod
    def _load_ollama(cls, llm_info: LLMInfo, **kwargs) -> Any:
        """Load from Ollama."""
        try:
            import ollama
        except ImportError:
            raise ImportError("ollama required: pip install ollama")

        return ollama


def register_llm(
    name: str,
    model_id: str,
    provider: str,
    num_params: int,
    context_length: int,
    description: str,
    quantization: Optional[str] = None,
) -> Callable:
    """Decorator to register an LLM."""

    def decorator(cls):
        llm_info = LLMInfo(
            name=name,
            model_id=model_id,
            provider=provider,
            num_params=num_params,
            context_length=context_length,
            description=description,
            quantization=quantization,
        )
        LLMRegistry.register(llm_info)
        return cls

    return decorator


def list_llms(provider: Optional[str] = None) -> list:
    """List available LLMs."""
    return LLMRegistry.list_llms(provider=provider)


def load_llm(name: str, **kwargs) -> Any:
    """Load an LLM."""
    return LLMRegistry.load(name, **kwargs)


LLMRegistry.register(
    LLMInfo(
        name="llama3_8b",
        model_id="meta-llama/Meta-Llama-3-8B",
        provider="huggingface",
        num_params=8000000000,
        context_length=8192,
        description="Meta's Llama 3 8B parameter model",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="llama3_70b",
        model_id="meta-llama/Meta-Llama-3-70B",
        provider="huggingface",
        num_params=70000000000,
        context_length=8192,
        description="Meta's Llama 3 70B parameter model",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="mistral_7b",
        model_id="mistralai/Mistral-7B-v0.1",
        provider="huggingface",
        num_params=7000000000,
        context_length=8192,
        description="Mistral 7B base model",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="mixtral_8x7b",
        model_id="mistralai/Mixtral-8x7B-v0.1",
        provider="huggingface",
        num_params=46000000000,
        context_length=32000,
        description="Mixtral mixture of experts 8x7B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="qwen2_7b",
        model_id="Qwen/Qwen2-7B",
        provider="huggingface",
        num_params=7000000000,
        context_length=32768,
        description="Qwen2 7B parameter model",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="qwen2_72b",
        model_id="Qwen/Qwen2-72B",
        provider="huggingface",
        num_params=72000000000,
        context_length=32768,
        description="Qwen2 72B parameter model",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="phi3_mini",
        model_id="microsoft/Phi-3-mini-4k-instruct",
        provider="huggingface",
        num_params=3000000000,
        context_length=4096,
        description="Microsoft Phi-3 Mini 3B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="phi3_medium",
        model_id="microsoft/Phi-3-medium-4k-instruct",
        provider="huggingface",
        num_params=14000000000,
        context_length=4096,
        description="Microsoft Phi-3 Medium 14B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="gemma_2b",
        model_id="google/gemma-2b",
        provider="huggingface",
        num_params=2000000000,
        context_length=8192,
        description="Google Gemma 2B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="gemma_7b",
        model_id="google/gemma-7b",
        provider="huggingface",
        num_params=7000000000,
        context_length=8192,
        description="Google Gemma 7B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="falcon_7b",
        model_id="tiiuae/falcon-7b",
        provider="huggingface",
        num_params=7000000000,
        context_length=2048,
        description="TII Falcon 7B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="falcon_40b",
        model_id="tiiuae/falcon-40b",
        provider="huggingface",
        num_params=40000000000,
        context_length=2048,
        description="TII Falcon 40B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="m2_mini",
        model_id="mlx-community/MiniMax-M2-mini-4bit",
        provider="huggingface",
        num_params=1200000000,
        context_length=32000,
        description="MiniMax M2 Mini (MLX, 4-bit)",
        quantization="4bit",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="codellama_7b",
        model_id="codellama/CodeLlama-7b-hf",
        provider="huggingface",
        num_params=7000000000,
        context_length=16384,
        description="Code Llama 7B for code generation",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="codellama_34b",
        model_id="codellama/CodeLlama-34b-hf",
        provider="huggingface",
        num_params=34000000000,
        context_length=16384,
        description="Code Llama 34B for code generation",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="stablelm_3b",
        model_id="stabilityai/stablelm-3b-4e1t",
        provider="huggingface",
        num_params=3000000000,
        context_length=4096,
        description="Stability AI StableLM 3B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="olmo_7b",
        model_id="ai2-llm/olmo-7b-0218-hf",
        provider="huggingface",
        num_params=7000000000,
        context_length=4096,
        description="AI2 OLMo 7B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="dbrx_base",
        model_id="databricks/dbrx-base",
        provider="huggingface",
        num_params=12000000000,
        context_length=32768,
        description="Databricks DBRX Base 12B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="llama3_8b_instruct",
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        provider="huggingface",
        num_params=8000000000,
        context_length=8192,
        description="Meta's Llama 3 8B Instruct",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="mistral_7b_instruct",
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
        provider="huggingface",
        num_params=7000000000,
        context_length=32768,
        description="Mistral 7B Instruct v0.2",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="qwen2_5_7b",
        model_id="Qwen/Qwen2.5-7B",
        provider="huggingface",
        num_params=7000000000,
        context_length=32768,
        description="Qwen2.5 7B parameter model",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="qwen2_5_14b",
        model_id="Qwen/Qwen2.5-14B",
        provider="huggingface",
        num_params=14000000000,
        context_length=32768,
        description="Qwen2.5 14B parameter model",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="qwen2_5_32b",
        model_id="Qwen/Qwen2.5-32B",
        provider="huggingface",
        num_params=32000000000,
        context_length=32768,
        description="Qwen2.5 32B parameter model",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="llama3_1_8b",
        model_id="meta-llama/Llama-3.1-8B",
        provider="huggingface",
        num_params=8000000000,
        context_length=128000,
        description="Meta's Llama 3.1 8B with 128K context",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="llama3_1_70b",
        model_id="meta-llama/Llama-3.1-70B",
        provider="huggingface",
        num_params=70000000000,
        context_length=128000,
        description="Meta's Llama 3.1 70B with 128K context",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="llama3_1_405b",
        model_id="meta-llama/Llama-3.1-405B",
        provider="huggingface",
        num_params=405000000000,
        context_length=128000,
        description="Meta's Llama 3.1 405B with 128K context",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="aya_23_8b",
        model_id="CohereForAI/aya-23-8B",
        provider="huggingface",
        num_params=8000000000,
        context_length=8192,
        description="Cohere Aya 23 8B multilingual",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="aya_23_35b",
        model_id="CohereForAI/aya-23-35B",
        provider="huggingface",
        num_params=35000000000,
        context_length=8192,
        description="Cohere Aya 23 35B multilingual",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="olmo2_7b",
        model_id="ai2-llm/olmo2-7B-1124",
        provider="huggingface",
        num_params=7000000000,
        context_length=4096,
        description="AI2 OLMo 2 7B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="olmo2_13b",
        model_id="ai2-llm/olmo2-13B-1124",
        provider="huggingface",
        num_params=13000000000,
        context_length=4096,
        description="AI2 OLMo 2 13B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="smollm_corpus_1.7b",
        model_id="HuggingFaceTB/SmolLM-1.7B",
        provider="huggingface",
        num_params=1700000000,
        context_length=2048,
        description="HuggingFace SmolLM 1.7B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="smollm_corpus_360m",
        model_id="HuggingFaceTB/SmolLM-360M",
        provider="huggingface",
        num_params=360000000,
        context_length=2048,
        description="HuggingFace SmolLM 360M",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="smollm_corpus_135m",
        model_id="HuggingFaceTB/SmolLM-135M",
        provider="huggingface",
        num_params=135000000,
        context_length=2048,
        description="HuggingFace SmolLM 135M",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="llama2_7b",
        model_id="meta-llama/Llama-2-7b-hf",
        provider="huggingface",
        num_params=7000000000,
        context_length=4096,
        description="Meta Llama 2 7B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="llama2_13b",
        model_id="meta-llama/Llama-2-13b-hf",
        provider="huggingface",
        num_params=13000000000,
        context_length=4096,
        description="Meta Llama 2 13B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="llama2_70b",
        model_id="meta-llama/Llama-2-70b-hf",
        provider="huggingface",
        num_params=70000000000,
        context_length=4096,
        description="Meta Llama 2 70B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="llama2_7b_chat",
        model_id="meta-llama/Llama-2-7b-chat-hf",
        provider="huggingface",
        num_params=7000000000,
        context_length=4096,
        description="Meta Llama 2 7B Chat",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="llama2_13b_chat",
        model_id="meta-llama/Llama-2-13b-chat-hf",
        provider="huggingface",
        num_params=13000000000,
        context_length=4096,
        description="Meta Llama 2 13B Chat",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="llama2_70b_chat",
        model_id="meta-llama/Llama-2-70b-chat-hf",
        provider="huggingface",
        num_params=70000000000,
        context_length=4096,
        description="Meta Llama 2 70B Chat",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="mistral_7b_v02",
        model_id="mistralai/Mistral-7B-v0.2",
        provider="huggingface",
        num_params=7000000000,
        context_length=32768,
        description="Mistral 7B v0.2",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="mistral_7b_v03",
        model_id="mistralai/Mistral-7B-v0.3",
        provider="huggingface",
        num_params=7000000000,
        context_length=32768,
        description="Mistral 7B v0.3",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="mixtral_8x22b",
        model_id="mistralai/Mixtral-8x22B-v0.1",
        provider="huggingface",
        num_params=141000000000,
        context_length=65536,
        description="Mixtral mixture of experts 8x22B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="mistral_nemo",
        model_id="mistralai/Mistral-Nemo-Instruct-2407",
        provider="huggingface",
        num_params=12000000000,
        context_length=128000,
        description="Mistral Nemo 12B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="pixtral_12b",
        model_id="mistralai/Pixtral-12B-2409",
        provider="huggingface",
        num_params=12000000000,
        context_length=128000,
        description="Mistral Pixtral 12B (multimodal)",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="qwen_1.5_0.5b",
        model_id="Qwen/Qwen1.5-0.5B",
        provider="huggingface",
        num_params=500000000,
        context_length=32768,
        description="Qwen 1.5 0.5B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="qwen_1.5_1.8b",
        model_id="Qwen/Qwen1.5-1.8B",
        provider="huggingface",
        num_params=1800000000,
        context_length=32768,
        description="Qwen 1.5 1.8B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="qwen_1.5_4b",
        model_id="Qwen/Qwen1.5-4B",
        provider="huggingface",
        num_params=4000000000,
        context_length=32768,
        description="Qwen 1.5 4B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="qwen_1.5_14b",
        model_id="Qwen/Qwen1.5-14B",
        provider="huggingface",
        num_params=14000000000,
        context_length=32768,
        description="Qwen 1.5 14B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="qwen_1.5_32b",
        model_id="Qwen/Qwen1.5-32B",
        provider="huggingface",
        num_params=32000000000,
        context_length=32768,
        description="Qwen 1.5 32B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="qwen_1.5_72b",
        model_id="Qwen/Qwen1.5-72B",
        provider="huggingface",
        num_params=72000000000,
        context_length=32768,
        description="Qwen 1.5 72B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="qwen2_vl_2b",
        model_id="Qwen/Qwen2-VL-2B-Instruct",
        provider="huggingface",
        num_params=2000000000,
        context_length=32768,
        description="Qwen2 VL 2B (vision)",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="qwen2_vl_7b",
        model_id="Qwen/Qwen2-VL-7B-Instruct",
        provider="huggingface",
        num_params=7000000000,
        context_length=32768,
        description="Qwen2 VL 7B (vision)",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="qwen2_vl_72b",
        model_id="Qwen/Qwen2-VL-72B-Instruct",
        provider="huggingface",
        num_params=72000000000,
        context_length=32768,
        description="Qwen2 VL 72B (vision)",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="qwen_math_1.5_72b",
        model_id="Qwen/Qwen1.5-Math-72B-Instruct",
        provider="huggingface",
        num_params=72000000000,
        context_length=8192,
        description="Qwen 1.5 Math 72B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="qwen_coder_1.5_14b",
        model_id="Qwen/Qwen1.5-14B-Code",
        provider="huggingface",
        num_params=14000000000,
        context_length=32768,
        description="Qwen 1.5 Code 14B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="qwen_coder_1.5_32b",
        model_id="Qwen/Qwen1.5-32B-Code",
        provider="huggingface",
        num_params=32000000000,
        context_length=32768,
        description="Qwen 1.5 Code 32B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="gemma_2_9b",
        model_id="google/gemma-2-9b",
        provider="huggingface",
        num_params=9000000000,
        context_length=8192,
        description="Google Gemma 2 9B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="gemma_2_27b",
        model_id="google/gemma-2-27b",
        provider="huggingface",
        num_params=27000000000,
        context_length=8192,
        description="Google Gemma 2 27B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="gemma_2_it_9b",
        model_id="google/gemma-2-9b-it",
        provider="huggingface",
        num_params=9000000000,
        context_length=8192,
        description="Google Gemma 2 9B Instruct",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="gemma_2_it_27b",
        model_id="google/gemma-2-27b-it",
        provider="huggingface",
        num_params=27000000000,
        context_length=8192,
        description="Google Gemma 2 27B Instruct",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="phi3_3.8b",
        model_id="microsoft/Phi-3-mini-128k-instruct",
        provider="huggingface",
        num_params=3800000000,
        context_length=128000,
        description="Microsoft Phi-3 Mini 128K 3.8B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="phi3.5_mini",
        model_id="microsoft/Phi-3.5-mini-instruct",
        provider="huggingface",
        num_params=4000000000,
        context_length=4096,
        description="Microsoft Phi-3.5 Mini",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="phi3.5_moe",
        model_id="microsoft/Phi-3.5-MoE-instruct",
        provider="huggingface",
        num_params=42000000000,
        context_length=4096,
        description="Microsoft Phi-3.5 MoE",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="phi4_mini",
        model_id="microsoft/Phi-4-mini-instruct",
        provider="huggingface",
        num_params=4000000000,
        context_length=16384,
        description="Microsoft Phi-4 Mini",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="phi4",
        model_id="microsoft/phi-4",
        provider="huggingface",
        num_params=15000000000,
        context_length=16384,
        description="Microsoft Phi-4 15B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="command_r",
        model_id="CohereForAI/c4ai-command-r-v01",
        provider="huggingface",
        num_params=35000000000,
        context_length=4096,
        description="Cohere Command R 35B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="command_r_plus",
        model_id="CohereForAI/c4ai-command-r-plus",
        provider="huggingface",
        num_params=104000000000,
        context_length=4096,
        description="Cohere Command R+ 104B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="aya_expanse_8b",
        model_id="CohereForAI/aya-expanse-8b",
        provider="huggingface",
        num_params=8000000000,
        context_length=8192,
        description="Cohere Aya Expanse 8B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="aya_expanse_32b",
        model_id="CohereForAI/aya-expanse-32b",
        provider="huggingface",
        num_params=32000000000,
        context_length=8192,
        description="Cohere Aya Expanse 32B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="bloom_7b",
        model_id="bigscience/bloom-7b1",
        provider="huggingface",
        num_params=7000000000,
        context_length=2048,
        description="BigScience BLOOM 7B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="bloom_176b",
        model_id="bigscience/bloom",
        provider="huggingface",
        num_params=176000000000,
        context_length=2048,
        description="BigScience BLOOM 176B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="gpt_neox_20b",
        model_id="EleutherAI/gpt-neox-20b",
        provider="huggingface",
        num_params=20000000000,
        context_length=2048,
        description="EleutherAI GPT-NeoX 20B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="pythia_12b",
        model_id="EleutherAI/pythia-12b",
        provider="huggingface",
        num_params=12000000000,
        context_length=2048,
        description="EleutherAI Pythia 12B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="pythia_70b",
        model_id="EleutherAI/pythia-70b",
        provider="huggingface",
        num_params=70000000000,
        context_length=2048,
        description="EleutherAI Pythia 70B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="stablelm_2_1.6b",
        model_id="stabilityai/stablelm-2-1.6b",
        provider="huggingface",
        num_params=1600000000,
        context_length=4096,
        description="Stability AI StableLM 2 1.6B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="stablelm_2_12b",
        model_id="stabilityai/stablelm-2-12b",
        provider="huggingface",
        num_params=12000000000,
        context_length=4096,
        description="Stability AI StableLM 2 12B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="stablelm_2_zephyr_3b",
        model_id="stabilityai/StableLM-2-Zephyr-3B",
        provider="huggingface",
        num_params=3000000000,
        context_length=4096,
        description="Stability AI StableLM 2 Zephyr 3B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="stablelm_2_zephyr_12b",
        model_id="stabilityai/StableLM-2-Zephyr-12B",
        provider="huggingface",
        num_params=12000000000,
        context_length=4096,
        description="Stability AI StableLM 2 Zephyr 12B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="redpajama_7b",
        model_id="togethercomputer/RedPajama-INCITE-7B-Chat",
        provider="huggingface",
        num_params=7000000000,
        context_length=2048,
        description="RedPajama INCITE 7B Chat",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="redpajama_7b_instruct",
        model_id="togethercomputer/RedPajama-INCITE-7B-Instruct",
        provider="huggingface",
        num_params=7000000000,
        context_length=2048,
        description="RedPajama INCITE 7B Instruct",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="m2_bert",
        model_id="mlx-community/MiniMax-M2-bert-4bit",
        provider="huggingface",
        num_params=180000000,
        context_length=512,
        description="MiniMax M2 BERT (MLX, 4-bit)",
        quantization="4bit",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="olmo_1_7b",
        model_id="ai2-llm/olmo-1.7-7b-hf",
        provider="huggingface",
        num_params=7000000000,
        context_length=4096,
        description="AI2 OLMo 1.7 7B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="olmo_1_70b",
        model_id="ai2-llm/olmo-1.7-70b-hf",
        provider="huggingface",
        num_params=70000000000,
        context_length=4096,
        description="AI2 OLMo 1.7 70B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name=" Granite_3.0_8b",
        model_id="ibm/Granite-3.0-8B-Base",
        provider="huggingface",
        num_params=8000000000,
        context_length=4096,
        description="IBM Granite 3.0 8B Base",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name=" Granite_3.0_8b_instruct",
        model_id="ibm/Granite-3.0-8B-Instruct",
        provider="huggingface",
        num_params=8000000000,
        context_length=4096,
        description="IBM Granite 3.0 8B Instruct",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name=" Granite_3.0_3b",
        model_id="ibm/Granite-3.0-3B-Base",
        provider="huggingface",
        num_params=3000000000,
        context_length=4096,
        description="IBM Granite 3.0 3B Base",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name=" Granite_3.0_3b_instruct",
        model_id="ibm/Granite-3.0-3B-Instruct",
        provider="huggingface",
        num_params=3000000000,
        context_length=4096,
        description="IBM Granite 3.0 3B Instruct",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="Llama-3.3_70b",
        model_id="meta-llama/Llama-3.3-70B-Instruct",
        provider="huggingface",
        num_params=70000000000,
        context_length=128000,
        description="Meta Llama 3.3 70B Instruct",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="Llama_3.2_1b",
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        provider="huggingface",
        num_params=1000000000,
        context_length=128000,
        description="Meta Llama 3.2 1B Instruct",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="Llama_3.2_3b",
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        provider="huggingface",
        num_params=3000000000,
        context_length=128000,
        description="Meta Llama 3.2 3B Instruct",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="Llama_3.2_11b_vision",
        model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
        provider="huggingface",
        num_params=11000000000,
        context_length=128000,
        description="Meta Llama 3.2 11B Vision Instruct",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="Llama_3.2_90b_vision",
        model_id="meta-llama/Llama-3.2-90B-Vision-Instruct",
        provider="huggingface",
        num_params=90000000000,
        context_length=128000,
        description="Meta Llama 3.2 90B Vision Instruct",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="solar_10.7b",
        model_id="upstage/SOLAR-10.7B-Instruct-v1.0",
        provider="huggingface",
        num_params=10700000000,
        context_length=4096,
        description="Upstage SOLAR 10.7B Instruct",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="solar_11b",
        model_id="upstage/SOLAR-11B-16k-Instruct",
        provider="huggingface",
        num_params=11000000000,
        context_length=16384,
        description="Upstage SOLAR 11B 16K Instruct",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="arctic",
        model_id="Snowflake/snowflake-arctic-instruct",
        provider="huggingface",
        num_params=48000000000,
        context_length=4096,
        description="Snowflake Arctic 48B Instruct",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="deepseek_7b",
        model_id="deepseek-ai/DeepSeek-V2",
        provider="huggingface",
        num_params=7000000000,
        context_length=4096,
        description="DeepSeek 7B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="deepseek_67b",
        model_id="deepseek-ai/DeepSeek-67B",
        provider="huggingface",
        num_params=67000000000,
        context_length=4096,
        description="DeepSeek 67B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="deepseek_coder_6.7b",
        model_id="deepseek-ai/deepseek-coder-6.7b-instruct",
        provider="huggingface",
        num_params=6700000000,
        context_length=16384,
        description="DeepSeek Coder 6.7B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="deepseek_coder_33b",
        model_id="deepseek-ai/deepseek-coder-33b-instruct",
        provider="huggingface",
        num_params=33000000000,
        context_length=16384,
        description="DeepSeek Coder 33B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="deepseek_v2_lite",
        model_id="deepseek-ai/DeepSeek-V2-Lite",
        provider="huggingface",
        num_params=1600000000,
        context_length=16384,
        description="DeepSeek V2 Lite",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="deepseek_v2_chat",
        model_id="deepseek-ai/DeepSeek-V2-Chat",
        provider="huggingface",
        num_params=21000000000,
        context_length=16384,
        description="DeepSeek V2 Chat",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="deepseek_v3",
        model_id="deepseek-ai/DeepSeek-V3",
        provider="huggingface",
        num_params=685000000000,
        context_length=64000,
        description="DeepSeek V3 685B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="deepseek_reasoner",
        model_id="deepseek-ai/DeepSeek-R1",
        provider="huggingface",
        num_params=685000000000,
        context_length=64000,
        description="DeepSeek R1 (Reasoning) 685B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="yi_6b",
        model_id="01-ai/Yi-6B",
        provider="huggingface",
        num_params=6000000000,
        context_length=4096,
        description="01.AI Yi 6B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="yi_9b",
        model_id="01-ai/Yi-9B",
        provider="huggingface",
        num_params=9000000000,
        context_length=4096,
        description="01.AI Yi 9B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="yi_34b",
        model_id="01-ai/Yi-34B",
        provider="huggingface",
        num_params=34000000000,
        context_length=4096,
        description="01.AI Yi 34B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="yi_34b_chat",
        model_id="01-ai/Yi-34B-Chat",
        provider="huggingface",
        num_params=34000000000,
        context_length=4096,
        description="01.AI Yi 34B Chat",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="yi_coder_6b",
        model_id="01-ai/Yi-Coder-6B",
        provider="huggingface",
        num_params=6000000000,
        context_length=4096,
        description="01.AI Yi Coder 6B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="yi_coder_9b",
        model_id="01-ai/Yi-Coder-9B",
        provider="huggingface",
        num_params=9000000000,
        context_length=4096,
        description="01.AI Yi Coder 9B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="phi_1",
        model_id="microsoft/phi-1",
        provider="huggingface",
        num_params=1300000000,
        context_length=2048,
        description="Microsoft Phi-1 1.3B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="phi_1.5",
        model_id="microsoft/phi-1_5",
        provider="huggingface",
        num_params=1300000000,
        context_length=2048,
        description="Microsoft Phi-1.5 1.3B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="phi_2",
        model_id="microsoft/phi-2",
        provider="huggingface",
        num_params=2700000000,
        context_length=2048,
        description="Microsoft Phi-2 2.7B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="orion_14b",
        model_id="OrionStarAI/Orion-14B-Base",
        provider="huggingface",
        num_params=14000000000,
        context_length=4096,
        description="OrionStar Orion 14B Base",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="orion_14b_chat",
        model_id="OrionStarAI/Orion-14B-Chat",
        provider="huggingface",
        num_params=14000000000,
        context_length=4096,
        description="OrionStar Orion 14B Chat",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="blue_7b",
        model_id="KomeijiForce/BlueMoon7B",
        provider="huggingface",
        num_params=7000000000,
        context_length=4096,
        description="BlueMoon 7B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="l3_8b_stealth",
        model_id="StealthAI/stealth-llama-3-8B",
        provider="huggingface",
        num_params=8000000000,
        context_length=8192,
        description="Stealth Llama 3 8B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="bge_m3",
        model_id="BAAI/bge-m3",
        provider="huggingface",
        num_params=560000000,
        context_length=8192,
        description="BAAI BGE-M3 embedding model",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="bge_large",
        model_id="BAAI/bge-large-en-v1.5",
        provider="huggingface",
        num_params=335000000,
        context_length=512,
        description="BAAI BGE-Large embedding model",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="e5_mistral",
        model_id="intfloat/e5-mistral-7b-instruct",
        provider="huggingface",
        num_params=7000000000,
        context_length=4096,
        description="E5 Mistral 7B instruct",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="gte_qwen2_7b",
        model_id="Alibaba-NLP/gte-Qwen2-7B-instruct",
        provider="huggingface",
        num_params=7000000000,
        context_length=8192,
        description="Alibaba GTE Qwen2 7B instruct",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="jamba_1.5_mini",
        model_id="ai21labs/Jamba-1.5-Mini",
        provider="huggingface",
        num_params=12000000000,
        context_length=256000,
        description="AI21 Labs Jamba 1.5 Mini",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="jamba_1.5_large",
        model_id="ai21labs/Jamba-1.5-Large",
        provider="huggingface",
        num_params=94000000000,
        context_length=256000,
        description="AI21 Labs Jamba 1.5 Large",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="jamba_1.5_medium",
        model_id="ai21labs/Jamba-1.5-Medium",
        provider="huggingface",
        num_params=51000000000,
        context_length=256000,
        description="AI21 Labs Jamba 1.5 Medium",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="marryyou_8b",
        model_id="MarryAI/marryyou_8b",
        provider="huggingface",
        num_params=8000000000,
        context_length=4096,
        description="MarryYou 8B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="zephyr_3b",
        model_id="HuggingFaceH4/zephyr-3b",
        provider="huggingface",
        num_params=3000000000,
        context_length=4096,
        description="Zephyr 3B DPO",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="zephyr_7b",
        model_id="HuggingFaceH4/zephyr-7b-beta",
        provider="huggingface",
        num_params=7000000000,
        context_length=4096,
        description="Zephyr 7B Beta",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="openchat_3.5",
        model_id="openchat/openchat-3.5-0106",
        provider="huggingface",
        num_params=7000000000,
        context_length=4096,
        description="OpenChat 3.5 7B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="openchat_7b",
        model_id="openchat/openchat-7b",
        provider="huggingface",
        num_params=7000000000,
        context_length=4096,
        description="OpenChat 7B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="mathstral_7b",
        model_id="mistralai/Mathstral-7B-v0.1",
        provider="huggingface",
        num_params=7000000000,
        context_length=32768,
        description="Mistral Mathstral 7B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="codestral_7b",
        model_id="mistralai/CodeStral-7B-v0.1",
        provider="huggingface",
        num_params=7000000000,
        context_length=32768,
        description="Mistral CodeStral 7B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="codestral_22b",
        model_id="mistralai/CodeStral-22B-v0.1",
        provider="huggingface",
        num_params=22000000000,
        context_length=65536,
        description="Mistral CodeStral 22B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="falcon3_7b",
        model_id="tiiuae/falcon3-7b",
        provider="huggingface",
        num_params=7000000000,
        context_length=8192,
        description="TII Falcon 3 7B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="falcon3_10b",
        model_id="tiiuae/falcon3-10b",
        provider="huggingface",
        num_params=10000000000,
        context_length=8192,
        description="TII Falcon 3 10B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="falcon3_mamba_7b",
        model_id="tiiuae/Falcon3-Mamba-7B",
        provider="huggingface",
        num_params=7000000000,
        context_length=8192,
        description="TII Falcon 3 Mamba 7B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="llama3_8b_olea",
        model_id="OmegaPull/llama-3-8b-OpenEuler",
        provider="huggingface",
        num_params=8000000000,
        context_length=8192,
        description="OmegaPull Llama 3 8B OpenEuler",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="claude_3_opus",
        model_id="anthropic/claude-3-opus",
        provider="huggingface",
        num_params=175000000000,
        context_length=200000,
        description="Anthropic Claude 3 Opus",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="claude_3_sonnet",
        model_id="anthropic/claude-3-sonnet",
        provider="huggingface",
        num_params=45000000000,
        context_length=200000,
        description="Anthropic Claude 3 Sonnet",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="claude_3_haiku",
        model_id="anthropic/claude-3-haiku",
        provider="huggingface",
        num_params=20000000000,
        context_length=200000,
        description="Anthropic Claude 3 Haiku",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="claude_3.5_sonnet",
        model_id="anthropic/claude-3.5-sonnet",
        provider="huggingface",
        num_params=45000000000,
        context_length=200000,
        description="Anthropic Claude 3.5 Sonnet",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="gemini_1.5_pro",
        model_id="google/gemini-1.5-pro",
        provider="huggingface",
        num_params=190000000000,
        context_length=2000000,
        description="Google Gemini 1.5 Pro",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="gemini_1.5_flash",
        model_id="google/gemini-1.5-flash",
        provider="huggingface",
        num_params=60000000000,
        context_length=1000000,
        description="Google Gemini 1.5 Flash",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="gemini_1.5_flash_8b",
        model_id="google/gemini-1.5-flash-8b",
        provider="huggingface",
        num_params=8000000000,
        context_length=1000000,
        description="Google Gemini 1.5 Flash 8B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="gemini_2.0_flash",
        model_id="google/gemini-2.0-flash",
        provider="huggingface",
        num_params=12000000000,
        context_length=1000000,
        description="Google Gemini 2.0 Flash",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="gemini_2.0_pro",
        model_id="google/gemini-2.0-pro",
        provider="huggingface",
        num_params=180000000000,
        context_length=2000000,
        description="Google Gemini 2.0 Pro",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="gpt4",
        model_id="openai/gpt-4",
        provider="huggingface",
        num_params=180000000000,
        context_length=8192,
        description="OpenAI GPT-4",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="gpt4_turbo",
        model_id="openai/gpt-4-turbo",
        provider="huggingface",
        num_params=180000000000,
        context_length=128000,
        description="OpenAI GPT-4 Turbo",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="gpt4o",
        model_id="openai/gpt-4o",
        provider="huggingface",
        num_params=180000000000,
        context_length=128000,
        description="OpenAI GPT-4o",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="gpt4o_mini",
        model_id="openai/gpt-4o-mini",
        provider="huggingface",
        num_params=8000000000,
        context_length=128000,
        description="OpenAI GPT-4o Mini",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="gpt_o1",
        model_id="openai/o1",
        provider="huggingface",
        num_params=180000000000,
        context_length=128000,
        description="OpenAI o1 (Reasoning)",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="gpt_o1_mini",
        model_id="openai/o1-mini",
        provider="huggingface",
        num_params=8000000000,
        context_length=128000,
        description="OpenAI o1-mini (Reasoning)",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="gpt_o3",
        model_id="openai/o3",
        provider="huggingface",
        num_params=180000000000,
        context_length=200000,
        description="OpenAI o3",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="gpt_o3_mini",
        model_id="openai/o3-mini",
        provider="huggingface",
        num_params=8000000000,
        context_length=200000,
        description="OpenAI o3-mini",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="gpt35_turbo",
        model_id="openai/gpt-3.5-turbo",
        provider="huggingface",
        num_params=175000000000,
        context_length=16385,
        description="OpenAI GPT-3.5 Turbo",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="gpt35_turbo_instruct",
        model_id="openai/gpt-3.5-turbo-instruct",
        provider="huggingface",
        num_params=175000000000,
        context_length=4096,
        description="OpenAI GPT-3.5 Turbo Instruct",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="davinci_002",
        model_id="openai/text-davinci-002",
        provider="huggingface",
        num_params=175000000000,
        context_length=4096,
        description="OpenAI Davinci 002",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="davinci_003",
        model_id="openai/text-davinci-003",
        provider="huggingface",
        num_params=175000000000,
        context_length=4096,
        description="OpenAI Davinci 003",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="claude_2.1",
        model_id="anthropic/claude-2.1",
        provider="huggingface",
        num_params=175000000000,
        context_length=200000,
        description="Anthropic Claude 2.1",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="claude_2.0",
        model_id="anthropic/claude-2.0",
        provider="huggingface",
        num_params=175000000000,
        context_length=100000,
        description="Anthropic Claude 2.0",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="claude_instant",
        model_id="anthropic/claude-instant",
        provider="huggingface",
        num_params=175000000000,
        context_length=100000,
        description="Anthropic Claude Instant",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="claude_1",
        model_id="anthropic/claude-1",
        provider="huggingface",
        num_params=175000000000,
        context_length=8192,
        description="Anthropic Claude 1",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="command_r_08_2024",
        model_id="CohereForAI/c4ai-command-r08-2024-09-09",
        provider="huggingface",
        num_params=104000000000,
        context_length=131072,
        description="Cohere Command R 08-2024",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="aya_101",
        model_id="CohereForAI/aya-101",
        provider="huggingface",
        num_params=8000000000,
        context_length=8192,
        description="Cohere Aya 101",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="c4ai_command_r_plus_08_2024",
        model_id="CohereForAI/c4ai-command-r-plus-08-2024",
        provider="huggingface",
        num_params=104000000000,
        context_length=131072,
        description="Cohere Command R+ 08-2024",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="xwinlm_7b",
        model_id="Xwin-LM/Xwin-LM-7B-V0.1",
        provider="huggingface",
        num_params=7000000000,
        context_length=8192,
        description="Xwin-LM 7B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="xwinlm_13b",
        model_id="Xwin-LM/Xwin-LM-13B-V0.1",
        provider="huggingface",
        num_params=13000000000,
        context_length=8192,
        description="Xwin-LM 13B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="xwinlm_70b",
        model_id="Xwin-LM/Xwin-LM-70B-V0.1",
        provider="huggingface",
        num_params=70000000000,
        context_length=8192,
        description="Xwin-LM 70B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="wizardlm_7b",
        model_id="WizardLM/WizardLM-7B-V1.0",
        provider="huggingface",
        num_params=7000000000,
        context_length=4096,
        description="WizardLM 7B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="wizardlm_13b",
        model_id="WizardLM/WizardLM-13B-V1.2",
        provider="huggingface",
        num_params=13000000000,
        context_length=4096,
        description="WizardLM 13B V1.2",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="wizardlm_70b",
        model_id="WizardLM/WizardLM-70B-V1.0",
        provider="huggingface",
        num_params=70000000000,
        context_length=4096,
        description="WizardLM 70B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="wizardcoder_7b",
        model_id="WizardLM/WizardCoder-7B-V1.0",
        provider="huggingface",
        num_params=7000000000,
        context_length=8192,
        description="WizardCoder 7B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="wizardcoder_13b",
        model_id="WizardLM/WizardCoder-13B-V1.0",
        provider="huggingface",
        num_params=13000000000,
        context_length=8192,
        description="WizardCoder 13B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="wizardcoder_34b",
        model_id="WizardLM/WizardCoder-34B-V1.0",
        provider="huggingface",
        num_params=34000000000,
        context_length=16384,
        description="WizardCoder 34B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="vicuna_7b",
        model_id="lmsys/vicuna-7b-v1.5",
        provider="huggingface",
        num_params=7000000000,
        context_length=4096,
        description="Vicuna 7B v1.5",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="vicuna_13b",
        model_id="lmsys/vicuna-13b-v1.5",
        provider="huggingface",
        num_params=13000000000,
        context_length=4096,
        description="Vicuna 13B v1.5",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="vicuna_33b",
        model_id="lmsys/vicuna-33b-v1.3",
        provider="huggingface",
        num_params=33000000000,
        context_length=4096,
        description="Vicuna 33B v1.3",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="airoboros_7b",
        model_id="jondurbin/airoboros-7b-gpt4-2.5",
        provider="huggingface",
        num_params=7000000000,
        context_length=4096,
        description="Airoboros 7B GPT4-2.5",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="airoboros_13b",
        model_id="jondurbin/airoboros-13b-gpt4-2.5",
        provider="huggingface",
        num_params=13000000000,
        context_length=4096,
        description="Airoboros 13B GPT4-2.5",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="airoboros_70b",
        model_id="jondurbin/airoboros-70b-gpt4-2.5",
        provider="huggingface",
        num_params=70000000000,
        context_length=4096,
        description="Airoboros 70B GPT4-2.5",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="chronos_5_13b",
        model_id="Amazon/Chronos-5-13B",
        provider="huggingface",
        num_params=13000000000,
        context_length=4096,
        description="Amazon Chronos 5 13B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="chronos_5_70b",
        model_id="Amazon/Chronos-5-70B",
        provider="huggingface",
        num_params=70000000000,
        context_length=4096,
        description="Amazon Chronos 5 70B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="falcon_180b",
        model_id="tiiuae/falcon-180B",
        provider="huggingface",
        num_params=180000000000,
        context_length=2048,
        description="TII Falcon 180B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="falcon_180b_chat",
        model_id="tiiuae/falcon-180B-chat",
        provider="huggingface",
        num_params=180000000000,
        context_length=2048,
        description="TII Falcon 180B Chat",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="mpt_7b",
        model_id="mosaicml/mpt-7b",
        provider="huggingface",
        num_params=7000000000,
        context_length=4096,
        description="MosaicML MPT 7B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="mpt_7b_instruct",
        model_id="mosaicml/mpt-7b-instruct",
        provider="huggingface",
        num_params=7000000000,
        context_length=4096,
        description="MosaicML MPT 7B Instruct",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="mpt_30b",
        model_id="mosaicml/mpt-30b",
        provider="huggingface",
        num_params=30000000000,
        context_length=4096,
        description="MosaicML MPT 30B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="mpt_30b_instruct",
        model_id="mosaicml/mpt-30b-instruct",
        provider="huggingface",
        num_params=30000000000,
        context_length=4096,
        description="MosaicML MPT 30B Instruct",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="mpt_65b",
        model_id="mosaicml/mpt-65b",
        provider="huggingface",
        num_params=65000000000,
        context_length=4096,
        description="MosaicML MPT 65B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="mpt_65b_instruct",
        model_id="mosaicml/mpt-65b-instruct",
        provider="huggingface",
        num_params=65000000000,
        context_length=4096,
        description="MosaicML MPT 65B Instruct",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="guanaco_7b",
        model_id="timdettmers/guanaco-7b",
        provider="huggingface",
        num_params=7000000000,
        context_length=4096,
        description="Guanaco 7B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="guanaco_13b",
        model_id="timdettmers/guanaco-13b",
        provider="huggingface",
        num_params=13000000000,
        context_length=4096,
        description="Guanaco 13B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="guanaco_33b",
        model_id="timdettmers/guanaco-33b",
        provider="huggingface",
        num_params=33000000000,
        context_length=4096,
        description="Guanaco 33B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="guanaco_65b",
        model_id="timdettmers/guanaco-65b",
        provider="huggingface",
        num_params=65000000000,
        context_length=4096,
        description="Guanaco 65B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="baize_7b",
        project="baize/baize-7b",
        provider="huggingface",
        num_params=7000000000,
        context_length=4096,
        description="Baize 7B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="baize_13b",
        project="baize/baize-13b",
        provider="huggingface",
        num_params=13000000000,
        context_length=4096,
        description="Baize 13B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="alpaca_7b",
        project="tatsu-lab/alpaca-7b",
        provider="huggingface",
        num_params=7000000000,
        context_length=4096,
        description="Alpaca 7B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="alpaca_13b",
        project="tatsu-lab/alpaca-13b",
        provider="huggingface",
        num_params=13000000000,
        context_length=4096,
        description="Alpaca 13B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="llama_2_7b_32k",
        project="meta-llama/Llama-2-7b-32k",
        provider="huggingface",
        num_params=7000000000,
        context_length=32768,
        description="Meta Llama 2 7B 32K",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="llama_2_70b_32k",
        project="meta-llama/Llama-2-70b-32k",
        provider="huggingface",
        num_params=70000000000,
        context_length=32768,
        description="Meta Llama 2 70B 32K",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="longchat_7b_16k",
        project="lmsys/longchat-7b-16k",
        provider="huggingface",
        num_params=7000000000,
        context_length=16384,
        description="LongChat 7B 16K",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="longchat_13b_16k",
        project="lmsys/longchat-13b-16k",
        provider="huggingface",
        num_params=13000000000,
        context_length=16384,
        description="LongChat 13B 16K",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="gemma_2b_it",
        project="google/gemma-2b-it",
        provider="huggingface",
        num_params=2000000000,
        context_length=8192,
        description="Google Gemma 2B Instruct",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="gemma_7b_it",
        project="google/gemma-7b-it",
        provider="huggingface",
        num_params=7000000000,
        context_length=8192,
        description="Google Gemma 7B Instruct",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="qwen_2_0.5b",
        project="Qwen/Qwen2-0.5B",
        provider="huggingface",
        num_params=500000000,
        context_length=32768,
        description="Qwen 2 0.5B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="qwen_2_1.5b",
        project="Qwen/Qwen2-1.5B",
        provider="huggingface",
        num_params=1500000000,
        context_length=32768,
        description="Qwen 2 1.5B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="qwen_2_57b",
        project="Qwen/Qwen2-57B-A14B",
        provider="huggingface",
        num_params=57000000000,
        context_length=32768,
        description="Qwen 2 57B (14B active)",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="qwen_2_VL_2B",
        project="Qwen/Qwen2-VL-2B-Instruct",
        provider="huggingface",
        num_params=2000000000,
        context_length=32768,
        description="Qwen 2 VL 2B Instruct",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="qwen_2_5_VL_72B",
        project="Qwen/Qwen2.5-VL-72B-Instruct",
        provider="huggingface",
        num_params=72000000000,
        context_length=32768,
        description="Qwen 2.5 VL 72B Instruct",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="qwen_2_math_1.5_72B",
        project="Qwen/Qwen2-Math-72B-Instruct",
        provider="huggingface",
        num_params=72000000000,
        context_length=8192,
        description="Qwen 2 Math 72B Instruct",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="olmo2_1b",
        project="ai2-llm/olmo2-1B-1124",
        provider="huggingface",
        num_params=1000000000,
        context_length=4096,
        description="AI2 OLMo 2 1B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="olmo2_27b",
        project="ai2-llm/olmo2-27B-1124",
        provider="huggingface",
        num_params=27000000000,
        context_length=4096,
        description="AI2 OLMo 2 27B",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="llama_3_8b_instruct_long",
        project="meta-llama/Meta-Llama-3-8B-Instruct-70k",
        provider="huggingface",
        num_params=8000000000,
        context_length=70000,
        description="Meta Llama 3 8B Instruct 70K",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="llama_3.1_8b_instruct",
        project="meta-llama/Llama-3.1-8B-Instruct",
        provider="huggingface",
        num_params=8000000000,
        context_length=128000,
        description="Meta Llama 3.1 8B Instruct",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="llama_3.1_70b_instruct",
        project="meta-llama/Llama-3.1-70B-Instruct",
        provider="huggingface",
        num_params=70000000000,
        context_length=128000,
        description="Meta Llama 3.1 70B Instruct",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="llama_3.1_405b_instruct_fp8",
        project="meta-llama/Llama-3.1-405B-Instruct-FP8",
        provider="huggingface",
        num_params=405000000000,
        context_length=128000,
        description="Meta Llama 3.1 405B Instruct FP8",
        quantization="fp8",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="phi3.5_sam",
        project="microsoft/Phi-3.5-samoa-instruct",
        provider="huggingface",
        num_params=4000000000,
        context_length=4096,
        description="Microsoft Phi-3.5 Samoa",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="phi4_mini_instruct",
        project="microsoft/Phi-4-mini-instruct",
        provider="huggingface",
        num_params=4000000000,
        context_length=16384,
        description="Microsoft Phi-4 Mini Instruct",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="phi4_instruct",
        project="microsoft/phi-4-instruct",
        provider="huggingface",
        num_params=15000000000,
        context_length=16384,
        description="Microsoft Phi-4 Instruct",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="mistral_7b_instruct_v2",
        project="mistralai/Mistral-7B-Instruct-v0.2",
        provider="huggingface",
        num_params=7000000000,
        context_length=32768,
        description="Mistral 7B Instruct v0.2",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="mistral_7b_instruct_v3",
        project="mistralai/Mistral-7B-Instruct-v0.3",
        provider="huggingface",
        num_params=7000000000,
        context_length=32768,
        description="Mistral 7B Instruct v0.3",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="mistral_nemo_instruct_2407",
        project="mistralai/Mistral-Nemo-Instruct-2407",
        provider="huggingface",
        num_params=12000000000,
        context_length=128000,
        description="Mistral Nemo Instruct 2407",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="pixtral_12b_2409",
        project="mistralai/Pixtral-12B-2409",
        provider="huggingface",
        num_params=12000000000,
        context_length=128000,
        description="Mistral Pixtral 12B 2409",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="codestral_2501",
        project="mistralai/CodeStral-22B-v0.1",
        provider="huggingface",
        num_params=22000000000,
        context_length=65536,
        description="Mistral CodeStral 22B v0.1",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="ministral_8b",
        project="mistralai/Ministral-8B-Instruct-2410",
        provider="huggingface",
        num_params=8000000000,
        context_length=32768,
        description="Mistral Ministral 8B Instruct",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="ministral_3b",
        project="mistralai/Ministral-3B-Instruct-2410",
        provider="huggingface",
        num_params=3000000000,
        context_length=32768,
        description="Mistral Ministral 3B Instruct",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="seallab_7b_v1",
        project="Sea-AI/SEAL-7B-v1.0",
        provider="huggingface",
        num_params=7000000000,
        context_length=4096,
        description="Sea-AI SEAL 7B v1",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="seallab_13b_v1",
        project="Sea-AI/SEAL-13B-v1.0",
        provider="huggingface",
        num_params=13000000000,
        context_length=4096,
        description="Sea-AI SEAL 13B v1",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="seallab_70b_v1",
        project="Sea-AI/SEAL-70B-v1.0",
        provider="huggingface",
        num_params=70000000000,
        context_length=4096,
        description="Sea-AI SEAL 70B v1",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="seallab_7b_v2",
        project="Sea-AI/SEAL-7B-v2.0",
        provider="huggingface",
        num_params=7000000000,
        context_length=8192,
        description="Sea-AI SEAL 7B v2",
        quantization="bf16",
    )
)

LLMRegistry.register(
    LLMInfo(
        name="seallab_13b_v2",
        project="Sea-AI/SEAL-13B-v2.0",
        provider="huggingface",
        num_params=13000000000,
        context_length=8192,
        description="Sea-AI SEAL 13B v2",
        quantization="bf16",
    )
)


class ModelRegistry:
    """Registry for pretrained models."""

    _registry: Dict[str, ModelInfo] = {}

    @classmethod
    def register(cls, model_info: ModelInfo) -> None:
        """Register a pretrained model."""
        cls._registry[model_info.name] = model_info

    @classmethod
    def get(cls, name: str) -> Optional[ModelInfo]:
        """Get model info by name."""
        return cls._registry.get(name)

    @classmethod
    def list_models(
        cls, task: Optional[str] = None, framework: Optional[str] = None
    ) -> list:
        """List all available models."""
        models = list(cls._registry.values())

        if task:
            models = [m for m in models if m.task == task]
        if framework:
            models = [m for m in models if m.framework == framework]

        return models

    @classmethod
    def create_model(cls, name: str, **kwargs) -> nn.Module:
        """Create a model from the registry."""
        model_info = cls.get(name)
        if model_info is None:
            raise ValueError(f"Model {name} not found in registry")

        from fishstick.frameworks import (
            create_crls,
            create_toposformer,
            create_uif_i,
            create_uis_j,
            create_uia_k,
            create_crls_l,
            create_uia_m,
            create_uis_n,
            create_uia_o,
            create_uif_p,
            create_uinet_q,
            create_uif_r,
            create_usif_s,
            create_uif_t,
            create_usif_u,
            create_uif_v,
            create_mca_w,
            create_ttsik_x,
            create_ctna_y,
            create_scif_z,
        )

        framework = model_info.framework
        if framework == "crls":
            return create_crls(
                input_dim=model_info.input_dim,
                output_dim=model_info.output_dim,
                **kwargs,
            )
        elif framework == "toposformer":
            return create_toposformer(
                input_dim=model_info.input_dim,
                output_dim=model_info.output_dim,
                **kwargs,
            )
        elif framework == "uif_i":
            return create_uif_i(
                input_dim=model_info.input_dim,
                output_dim=model_info.output_dim,
                **kwargs,
            )
        elif framework == "uis_j":
            return create_uis_j(
                input_dim=model_info.input_dim,
                output_dim=model_info.output_dim,
                **kwargs,
            )
        elif framework == "uia_k":
            return create_uia_k(
                input_dim=model_info.input_dim,
                output_dim=model_info.output_dim,
                **kwargs,
            )
        elif framework == "crls_l":
            return create_crls_l(
                input_dim=model_info.input_dim,
                output_dim=model_info.output_dim,
                **kwargs,
            )
        elif framework == "uia_m":
            return create_uia_m(
                input_dim=model_info.input_dim,
                output_dim=model_info.output_dim,
                **kwargs,
            )
        elif framework == "uis_n":
            return create_uis_n(
                input_dim=model_info.input_dim,
                output_dim=model_info.output_dim,
                **kwargs,
            )
        elif framework == "uia_o":
            return create_uia_o(
                input_dim=model_info.input_dim,
                output_dim=model_info.output_dim,
                **kwargs,
            )
        elif framework == "uif_p":
            return create_uif_p(
                input_dim=model_info.input_dim,
                output_dim=model_info.output_dim,
                **kwargs,
            )
        elif framework == "uinet_q":
            return create_uinet_q(
                input_dim=model_info.input_dim,
                output_dim=model_info.output_dim,
                **kwargs,
            )
        elif framework == "uif_r":
            return create_uif_r(
                input_dim=model_info.input_dim,
                output_dim=model_info.output_dim,
                **kwargs,
            )
        elif framework == "usif_s":
            return create_usif_s(
                input_dim=model_info.input_dim,
                output_dim=model_info.output_dim,
                **kwargs,
            )
        elif framework == "uif_t":
            return create_uif_t(
                input_dim=model_info.input_dim,
                output_dim=model_info.output_dim,
                **kwargs,
            )
        elif framework == "usif_u":
            return create_usif_u(
                input_dim=model_info.input_dim,
                output_dim=model_info.output_dim,
                **kwargs,
            )
        elif framework == "uif_v":
            return create_uif_v(
                input_dim=model_info.input_dim,
                output_dim=model_info.output_dim,
                **kwargs,
            )
        elif framework == "mca_w":
            return create_mca_w(
                input_dim=model_info.input_dim,
                output_dim=model_info.output_dim,
                **kwargs,
            )
        elif framework == "ttsik_x":
            return create_ttsik_x(
                input_dim=model_info.input_dim,
                output_dim=model_info.output_dim,
                **kwargs,
            )
        elif framework == "ctna_y":
            return create_ctna_y(
                input_dim=model_info.input_dim,
                output_dim=model_info.output_dim,
                **kwargs,
            )
        elif framework == "scif_z":
            return create_scif_z(
                input_dim=model_info.input_dim,
                output_dim=model_info.output_dim,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown framework: {framework}")


def register_model(
    name: str,
    framework: str,
    task: str,
    architecture: str,
    num_params: int,
    input_dim: int,
    output_dim: int,
    description: str,
    paper: str,
    weights_url: Optional[str] = None,
) -> Callable:
    """Decorator to register a pretrained model."""

    def decorator(cls):
        model_info = ModelInfo(
            name=name,
            framework=framework,
            task=task,
            architecture=architecture,
            num_params=num_params,
            input_dim=input_dim,
            output_dim=output_dim,
            description=description,
            paper=paper,
            weights_url=weights_url,
        )
        ModelRegistry.register(model_info)
        return cls

    return decorator


def list_models(task: Optional[str] = None, framework: Optional[str] = None) -> list:
    """List available pretrained models."""
    return ModelRegistry.list_models(task=task, framework=framework)


def get_model_info(name: str) -> Optional[ModelInfo]:
    """Get information about a specific model."""
    return ModelRegistry.get(name)


def load_pretrained(name: str, device: str = "cpu", **kwargs) -> nn.Module:
    """Load a pretrained model."""
    model_info = ModelRegistry.get(name)
    if model_info is None:
        raise ValueError(f"Model {name} not found")

    model = ModelRegistry.create_model(name, **kwargs)
    model = model.to(device)

    if model_info.weights_url:
        try:
            state_dict = torch.hub.load_state_dict_from_url(
                model_info.weights_url,
                map_location=device,
            )
            model.load_state_dict(state_dict)
            print(f"Loaded pretrained weights from {model_info.weights_url}")
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")

    return model


ModelRegistry.register(
    ModelInfo(
        name="crls_mnist",
        framework="crls",
        task="image_classification",
        architecture="CRLSModel",
        num_params=300000,
        input_dim=784,
        output_dim=10,
        description="CRLS model pretrained on MNIST",
        paper="Categorical Renormalization Learning Systems",
        weights_url=None,
    )
)

ModelRegistry.register(
    ModelInfo(
        name="toposformer_cifar10",
        framework="toposformer",
        task="image_classification",
        architecture="ToposFormer",
        num_params=4800000,
        input_dim=3072,
        output_dim=10,
        description="ToposFormer pretrained on CIFAR-10",
        paper="Topos-Theoretic Neural Networks",
        weights_url=None,
    )
)

ModelRegistry.register(
    ModelInfo(
        name="uif_imdb",
        framework="uif",
        task="text_classification",
        architecture="UIFModel",
        num_params=370000,
        input_dim=30000,
        output_dim=2,
        description="UIF model for sentiment analysis",
        paper="Unified Intelligence Framework",
        weights_url=None,
    )
)

ModelRegistry.register(
    ModelInfo(
        name="uif_i_mnist",
        framework="uif_i",
        task="image_classification",
        architecture="UIF_I_Model",
        num_params=450000,
        input_dim=784,
        output_dim=10,
        description="UIF-I model pretrained on MNIST",
        paper="Unified Intelligence Framework - Variant I",
        weights_url=None,
    )
)

ModelRegistry.register(
    ModelInfo(
        name="uis_j_cifar10",
        framework="uis_j",
        task="image_classification",
        architecture="UIS_J_Model",
        num_params=2100000,
        input_dim=3072,
        output_dim=10,
        description="UIS-J model pretrained on CIFAR-10",
        paper="Unified Intelligence System - Variant J",
        weights_url=None,
    )
)

ModelRegistry.register(
    ModelInfo(
        name="uia_k_fashion",
        framework="uia_k",
        task="image_classification",
        architecture="UIA_K_Model",
        num_params=890000,
        input_dim=784,
        output_dim=10,
        description="UIA-K model pretrained on Fashion-MNIST",
        paper="Unified Intelligence Architecture - Variant K",
        weights_url=None,
    )
)

ModelRegistry.register(
    ModelInfo(
        name="crls_l_mnist",
        framework="crls_l",
        task="image_classification",
        architecture="CRLS_L_Model",
        num_params=520000,
        input_dim=784,
        output_dim=10,
        description="CRLS-L model pretrained on MNIST",
        paper="Categorical Renormalization Learning Systems - Variant L",
        weights_url=None,
    )
)

ModelRegistry.register(
    ModelInfo(
        name="uia_m_cifar100",
        framework="uia_m",
        task="image_classification",
        architecture="UIA_M_Model",
        num_params=3500000,
        input_dim=3072,
        output_dim=100,
        description="UIA-M model pretrained on CIFAR-100",
        paper="Unified Intelligence Architecture - Variant M",
        weights_url=None,
    )
)

ModelRegistry.register(
    ModelInfo(
        name="uis_n_imdb",
        framework="uis_n",
        task="text_classification",
        architecture="UIS_N_Model",
        num_params=680000,
        input_dim=30000,
        output_dim=2,
        description="UIS-N model for sentiment analysis",
        paper="Unified Intelligence System - Variant N",
        weights_url=None,
    )
)

ModelRegistry.register(
    ModelInfo(
        name="uia_o_tREC",
        framework="uia_o",
        task="text_classification",
        architecture="UIA_O_Model",
        num_params=420000,
        input_dim=5000,
        output_dim=6,
        description="UIA-O model for TREC question classification",
        paper="Unified Intelligence Architecture - Variant O",
        weights_url=None,
    )
)

ModelRegistry.register(
    ModelInfo(
        name="uif_p_mnist",
        framework="uif_p",
        task="image_classification",
        architecture="UIF_P_Model",
        num_params=380000,
        input_dim=784,
        output_dim=10,
        description="UIF-P model pretrained on MNIST",
        paper="Unified Intelligence Framework - Variant P",
        weights_url=None,
    )
)

ModelRegistry.register(
    ModelInfo(
        name="uinet_q_cifar10",
        framework="uinet_q",
        task="image_classification",
        architecture="UINet_Q_Model",
        num_params=5600000,
        input_dim=3072,
        output_dim=10,
        description="UINet-Q model pretrained on CIFAR-10",
        paper="Unified Intelligence Network - Variant Q",
        weights_url=None,
    )
)

ModelRegistry.register(
    ModelInfo(
        name="uif_r_fashion",
        framework="uif_r",
        task="image_classification",
        architecture="UIF_R_Model",
        num_params=720000,
        input_dim=784,
        output_dim=10,
        description="UIF-R model pretrained on Fashion-MNIST",
        paper="Unified Intelligence Framework - Variant R",
        weights_url=None,
    )
)

ModelRegistry.register(
    ModelInfo(
        name="usif_s_mnist",
        framework="usif_s",
        task="image_classification",
        architecture="USIF_S_Model",
        num_params=290000,
        input_dim=784,
        output_dim=10,
        description="USIF-S model pretrained on MNIST",
        paper="Unified Scale-Invariant Framework - Variant S",
        weights_url=None,
    )
)

ModelRegistry.register(
    ModelInfo(
        name="uif_t_cifar10",
        framework="uif_t",
        task="image_classification",
        architecture="UIF_T_Model",
        num_params=4100000,
        input_dim=3072,
        output_dim=10,
        description="UIF-T model pretrained on CIFAR-10",
        paper="Unified Intelligence Framework - Variant T",
        weights_url=None,
    )
)

ModelRegistry.register(
    ModelInfo(
        name="usif_u_imdb",
        framework="usif_u",
        task="text_classification",
        architecture="USIF_U_Model",
        num_params=550000,
        input_dim=30000,
        output_dim=2,
        description="USIF-U model for sentiment analysis",
        paper="Unified Scale-Invariant Framework - Variant U",
        weights_url=None,
    )
)

ModelRegistry.register(
    ModelInfo(
        name="uif_v_mnist",
        framework="uif_v",
        task="image_classification",
        architecture="UIF_V_Model",
        num_params=630000,
        input_dim=784,
        output_dim=10,
        description="UIF-V model pretrained on MNIST",
        paper="Unified Intelligence Framework - Variant V",
        weights_url=None,
    )
)

ModelRegistry.register(
    ModelInfo(
        name="mca_w_cifar100",
        framework="mca_w",
        task="image_classification",
        architecture="MCA_W_Model",
        num_params=7800000,
        input_dim=3072,
        output_dim=100,
        description="MCA-W model pretrained on CIFAR-100",
        paper="Multi-Scale Categorical Architecture - Variant W",
        weights_url=None,
    )
)

ModelRegistry.register(
    ModelInfo(
        name="ttsik_x_fashion",
        framework="ttsik_x",
        task="image_classification",
        architecture="TTSIK_X_Model",
        num_params=950000,
        input_dim=784,
        output_dim=10,
        description="TTSIK-X model pretrained on Fashion-MNIST",
        paper="Topological Time-Space Invariant Kernel - Variant X",
        weights_url=None,
    )
)

ModelRegistry.register(
    ModelInfo(
        name="ctna_y_cifar10",
        framework="ctna_y",
        task="image_classification",
        architecture="CTNA_Y_Model",
        num_params=3200000,
        input_dim=3072,
        output_dim=10,
        description="CTNA-Y model pretrained on CIFAR-10",
        paper="Categorical Topological Neural Architecture - Variant Y",
        weights_url=None,
    )
)

ModelRegistry.register(
    ModelInfo(
        name="scif_z_mnist",
        framework="scif_z",
        task="image_classification",
        architecture="SCIF_Z_Model",
        num_params=410000,
        input_dim=784,
        output_dim=10,
        description="SCIF-Z model pretrained on MNIST",
        paper="Scale-Invariant Categorical Information Framework - Variant Z",
        weights_url=None,
    )
)
