"""
External API Integrations for fishstick

Provides clients for integrating with external AI services:
- OpenAI (GPT models)
- Anthropic (Claude models)
- Google AI (PaLM/Gemini)
- Hugging Face Inference API
- Cohere
"""

from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
import os


class Provider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    COHERE = "cohere"


@dataclass
class ModelConfig:
    provider: Provider
    model_name: str
    api_key: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7


class OpenAIClient:
    """Client for OpenAI API (GPT-4, GPT-3.5-Turbo, etc.)"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "openai package not installed. Run: pip install openai"
                )
        return self._client

    def generate(
        self, prompt: str, system_message: Optional[str] = None, **kwargs
    ) -> str:
        """Generate text from prompt."""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        response = self._get_client().chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
        )
        return response.choices[0].message.content

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat with messages."""
        response = self._get_client().chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
        )
        return response.choices[0].message.content

    def embed(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """Get embeddings for text."""
        response = self._get_client().embeddings.create(
            model=model,
            input=text,
        )
        return response.data[0].embedding


class AnthropicClient:
    """Client for Anthropic API (Claude models)"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-sonnet-20240229",
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. Run: pip install anthropic"
                )
        return self._client

    def generate(
        self, prompt: str, system_message: Optional[str] = None, **kwargs
    ) -> str:
        """Generate text from prompt."""
        messages = [{"role": "user", "content": prompt}]

        response = self._get_client().messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            system=system_message or "",
            messages=messages,
        )
        return response.content[0].text

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat with messages."""
        response = self._get_client().messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            messages=messages,
        )
        return response.content[0].text


class GoogleAIClient:
    """Client for Google AI (Gemini/PaLM models)"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-pro",
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                self._client = genai
            except ImportError:
                raise ImportError(
                    "google-generativeai not installed. Run: pip install google-generativeai"
                )
        return self._client

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        client = self._get_client()
        model = client.GenerativeModel(self.model)

        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
            },
        )
        return response.text

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat with messages (uses last user message as prompt)."""
        for msg in reversed(messages):
            if msg["role"] == "user":
                return self.generate(msg["content"], **kwargs)
        return ""


class HuggingFaceClient:
    """Client for Hugging Face Inference API"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt2",
    ):
        self.api_key = api_key or os.getenv("HF_TOKEN")
        self.model = model

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using HF Inference API."""
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError(
                "huggingface_hub not installed. Run: pip install huggingface_hub"
            )

        client = InferenceClient(model=self.model, token=self.api_key)
        response = client.text_generation(
            prompt,
            max_new_tokens=kwargs.get("max_tokens", 100),
            temperature=kwargs.get("temperature", 0.7),
        )
        return response

    def embed(
        self, text: str, model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> List[float]:
        """Get embeddings using HF feature extraction."""
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            raise ImportError(
                "huggingface_hub not installed. Run: pip install huggingface_hub"
            )

        client = InferenceClient(model=model, token=self.api_key)
        return client.feature_extraction(text)


class CohereClient:
    """Client for Cohere API"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "command",
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ):
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        try:
            import cohere
        except ImportError:
            raise ImportError("cohere not installed. Run: pip install cohere")

        client = cohere.Client(self.api_key)
        response = client.generate(
            model=self.model,
            prompt=prompt,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
        )
        return response.generations[0].text

    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Get embeddings for text(s)."""
        try:
            import cohere
        except ImportError:
            raise ImportError("cohere not installed. Run: pip install cohere")

        client = cohere.Client(self.api_key)
        if isinstance(texts, str):
            texts = [texts]
        response = client.embed(texts=texts, model="embed-english-v3.0")
        return response.embeddings


def create_client(
    provider: Union[str, Provider],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> Union[
    OpenAIClient, AnthropicClient, GoogleAIClient, HuggingFaceClient, CohereClient
]:
    """Factory function to create API clients."""
    if isinstance(provider, str):
        provider = Provider(provider.lower())

    if provider == Provider.OPENAI:
        return OpenAIClient(api_key=api_key, model=model or "gpt-3.5-turbo", **kwargs)
    elif provider == Provider.ANTHROPIC:
        return AnthropicClient(
            api_key=api_key, model=model or "claude-3-sonnet-20240229", **kwargs
        )
    elif provider == Provider.GOOGLE:
        return GoogleAIClient(api_key=api_key, model=model or "gemini-pro", **kwargs)
    elif provider == Provider.HUGGINGFACE:
        return HuggingFaceClient(api_key=api_key, model=model or "gpt2")
    elif provider == Provider.COHERE:
        return CohereClient(api_key=api_key, model=model or "command", **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
