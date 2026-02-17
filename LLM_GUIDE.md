# Using LLMs in the fishstick Environment

This guide shows you how to work with Large Language Models using the available tools in your environment.

## Available Tools

### Installed Libraries
- **transformers** (v4.57.6) - Hugging Face's model library
- **accelerate** (v1.12.0) - Multi-device training support
- **ollama** (v0.6.1) - Local LLM inference
- **torch** - PyTorch for model operations

### Hardware Constraints
- **CPU only** - No GPU detected
- **RAM**: Limited (affects model size)
- **Best for**: Inference and small model fine-tuning

## Quick Start Examples

### 1. Basic Text Generation with GPT-2

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load GPT-2 (small, fast, CPU-friendly)
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare prompt
prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=50,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True
    )

# Decode and print
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

**Output Example:**
```
The future of artificial intelligence is likely to be a combination of 
human and machine intelligence. As AI systems become more sophisticated, 
they will increasingly work alongside humans to solve complex problems...
```

### 2. Using Different Model Sizes

```python
# DistilGPT2 - smaller, faster (82M parameters)
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# GPT-2 Medium - better quality, slower (345M parameters)
model = AutoModelForCausalLM.from_pretrained("gpt2-medium")

# GPT-2 Large - best quality, very slow on CPU (774M parameters)
# model = AutoModelForCausalLM.from_pretrained("gpt2-large")
```

### 3. Text Classification with BERT

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

# Load BERT for sentiment analysis
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Classify text
text = "I love this new framework! It's amazing!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    probabilities = F.softmax(outputs.logits, dim=-1)
    predicted_class = probabilities.argmax().item()
    confidence = probabilities[0][predicted_class].item()

print(f"Sentiment: {'Positive' if predicted_class == 1 else 'Negative'}")
print(f"Confidence: {confidence:.2%}")
```

### 4. Question Answering

```python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# Load QA model
model_name = "distilbert-base-cased-distilled-squad"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

context = """
The fishstick framework is a mathematically rigorous AI framework that 
combines theoretical physics, formal mathematics, and advanced machine learning. 
It implements 6 unified frameworks including Hamiltonian Neural Networks, 
Sheaf-Optimized Attention, and Renormalization Group flows.
"""

question = "What does fishstick combine?"

inputs = tokenizer(question, context, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    
# Get answer span
answer_start = outputs.start_logits.argmax()
answer_end = outputs.end_logits.argmax() + 1
answer = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
)

print(f"Q: {question}")
print(f"A: {answer}")
```

### 5. Simple Fine-tuning on Custom Data

```python
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    Trainer, 
    TrainingArguments,
    TextDataset,
    DataCollatorForLanguageModeling
)

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Prepare dataset
# Create a text file with your training data
train_path = "my_data.txt"  # One text sample per line

# Create dataset
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_path,
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # For causal LM
)

# Training arguments (optimized for CPU)
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Keep small for CPU
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=100,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Train (will be slow on CPU!)
# trainer.train()

# Save model
# trainer.save_model("./gpt2-finetuned")
```

### 6. Using Ollama for Local LLMs

```bash
# Pull a model
ollama pull llama2

# Run interactively
ollama run llama2

# Or use in code
```

```python
import requests
import json

# Ollama API endpoint
url = "http://localhost:11434/api/generate"

# Prepare request
data = {
    "model": "llama2",
    "prompt": "Explain the fishstick AI framework in simple terms:",
    "stream": False
}

# Send request (requires ollama server running)
response = requests.post(url, json=data)
result = json.loads(response.text)
print(result['response'])
```

## Performance Tips for CPU

### 1. Use Smaller Models
```python
# Fast on CPU (82M parameters)
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Use half precision to save memory
model = model.half()
```

### 2. Optimize Generation
```python
# Faster generation settings
outputs = model.generate(
    **inputs,
    max_new_tokens=50,      # Limit output length
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    num_beams=1,            # Disable beam search (slower)
    use_cache=True,         # Use KV cache
)
```

### 3. Batch Processing
```python
# Process multiple texts at once
texts = ["Text 1", "Text 2", "Text 3"]
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)
```

### 4. Use torch.no_grad()
```python
# Always use for inference (saves memory)
with torch.no_grad():
    outputs = model(**inputs)
```

## Model Recommendations by Use Case

### For Fast Inference (CPU)
- `distilgpt2` (82M) - Fastest, decent quality
- `gpt2` (124M) - Good balance
- `distilbert-base-uncased` (66M) - For classification

### For Better Quality (Slow on CPU)
- `gpt2-medium` (345M) - Better but slow
- `bert-base-uncased` (110M) - Good for understanding tasks

### Avoid on CPU
- `gpt2-large` (774M) - Too slow
- `gpt2-xl` (1.5B) - Way too slow
- Modern LLMs (LLaMA, GPT-3, etc.) - Need GPU

## Example: Complete Chat Bot

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class SimpleChatBot:
    def __init__(self, model_name="distilgpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def chat(self, prompt, max_length=100):
        # Format prompt
        formatted_prompt = f"User: {prompt}\nAssistant:"
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=len(inputs['input_ids'][0]) + max_length,
                num_return_sequences=1,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0][len(inputs['input_ids'][0]):], 
            skip_special_tokens=True
        )
        
        return response.strip()

# Use it
bot = SimpleChatBot()
response = bot.chat("What is machine learning?")
print(f"Bot: {response}")
```

## Troubleshooting

### Out of Memory
```python
# Use smaller model
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Use half precision
model = model.half()

# Reduce batch size
batch_size = 1
```

### Slow Generation
```python
# Limit sequence length
max_length = 50

# Disable beam search
num_beams = 1

# Use smaller model
```

### Model Download Issues
```python
# Cache models locally
from transformers import cache_dir
# Models saved to ~/.cache/huggingface/
```

## Next Steps

1. **Try the examples** above to get familiar
2. **Experiment with prompts** - LLMs are sensitive to prompt formatting
3. **Fine-tune on your data** - Even small models can be useful with domain-specific training
4. **Consider GPU access** - For serious training, you'll need CUDA support

## Resources

- [Hugging Face Models](https://huggingface.co/models)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [Accelerate Documentation](https://huggingface.co/docs/accelerate)

---

**Note**: CPU-only environments are best for inference and experimentation. For training large models, GPU access is essential!