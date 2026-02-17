# External API Integrations

Clients for external AI services and APIs.

## Installation

```bash
pip install fishstick[integrations]
```

## Overview

The `integrations` module provides unified clients for various external AI services including OpenAI, Anthropic, Google AI, Hugging Face, and Cohere.

## Usage

```python
from fishstick.integrations import create_client, OpenAIClient, AnthropicClient

# Create client via factory
client = create_client("openai", api_key="your-key")

# Or use specific client
openai_client = OpenAIClient(api_key="your-key")
response = openai_client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Anthropic
anthropic = AnthropicClient(api_key="your-key")
response = anthropic.messages.create(
    model="claude-3-opus",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Available Clients

| Client | Description |
|--------|-------------|
| `OpenAIClient` | OpenAI API (GPT models) |
| `AnthropicClient` | Anthropic API (Claude models) |
| `GoogleAIClient` | Google AI (Gemini) |
| `HuggingFaceClient` | Hugging Face Inference API |
| `CohereClient` | Cohere API |

## Factory Function

```python
from fishstick.integrations import create_client

client = create_client("openai", api_key="key", **kwargs)
```

## Examples

See `examples/integrations/` for complete examples.
