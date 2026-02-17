# Installed LLM Models - Complete Catalog

This document lists all **29 models** successfully installed in your fishstick environment.

## **Download Summary**
- ✅ **Successfully downloaded**: 29/32 models
- ❌ **Failed**: 3/32 models (translation models - missing sentencepiece)
- 📦 **Storage location**: `~/.cache/huggingface/`
- ⚡ **Status**: Ready to use instantly (no re-download needed)

---

## **1. Text Generation Models (7 models)**

Perfect for creative writing, chatbots, code generation, and text completion.

### GPT-2 Family
| Model | Size | Speed | Best For |
|-------|------|-------|----------|
| `distilgpt2` | 82M | ⚡⚡⚡ Fast | General text, testing |
| `gpt2` | 124M | ⚡⚡ Good | General text generation |
| `gpt2-medium` | 345M | ⚡ Slow | Better quality text |
| `gpt2-xl` | 1.5B | 🐌 Very Slow | Best quality (avoid on CPU) |

### Other Generative Models
| Model | Size | Description |
|-------|------|-------------|
| `EleutherAI/gpt-neo-125M` | 125M | Open-source GPT alternative |
| `EleutherAI/pythia-160m` | 160M | EleutherAI's interpretable model |
| `sshleifer/tiny-gpt2` | ~1M | Ultra-small for testing |

### Usage Example
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("distilgpt2")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=30)
print(tokenizer.decode(outputs[0]))
```

---

## **2. BERT & Understanding Models (8 models)**

For classification, named entity recognition, question answering, and understanding tasks.

### BERT Variants
| Model | Size | Description |
|-------|------|-------------|
| `bert-base-uncased` | 110M | Original BERT (uncased) |
| `bert-base-cased` | 110M | Original BERT (cased) |
| `distilbert-base-uncased` | 66M | 40% smaller, 97% performance |
| `roberta-base` | 125M | Robustly optimized BERT |
| `albert-base-v2` | 12M | Lite BERT (parameter sharing) |
| `distilroberta-base` | 82M | Distilled RoBERTa |

### Tiny Models (for testing)
| Model | Size | Description |
|-------|------|-------------|
| `prajjwal1/bert-tiny` | 4M | Ultra-fast testing |
| `prajjwal1/bert-mini` | 11M | Fast testing |

### Usage Example
```python
from transformers import AutoModel, AutoTokenizer

# Get embeddings
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "fishstick is an AI framework"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
embeddings = outputs.last_hidden_state
```

---

## **3. Classification Models (3 models)**

Pre-trained for specific classification tasks.

| Model | Task | Description |
|-------|------|-------------|
| `distilbert-base-uncased-finetuned-sst-2-english` | Sentiment | Positive/Negative classification |
| `dslim/bert-base-NER` | NER | Named Entity Recognition (Person, Org, Location) |
| `dslim/distilbert-NER` | NER | Fast NER version |

### Usage Example
```python
from transformers import pipeline

# Sentiment analysis
classifier = pipeline("sentiment-analysis", 
                     model="distilbert-base-uncased-finetuned-sst-2-english")
result = classifier("I love fishstick!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]

# Named Entity Recognition
ner = pipeline("ner", model="dslim/distilbert-NER", aggregation_strategy="simple")
entities = ner("Apple Inc. was founded by Steve Jobs in Cupertino.")
```

---

## **4. Question Answering Models (2 models)**

Extract answers from context text.

| Model | Size | Description |
|-------|------|-------------|
| `distilbert-base-cased-distilled-squad` | 66M | Fast QA on SQuAD dataset |
| `bert-large-uncased-whole-word-masking-finetuned-squad` | 340M | High accuracy QA |

### Usage Example
```python
from transformers import pipeline

qa = pipeline("question-answering", 
              model="distilbert-base-cased-distilled-squad")

context = "fishstick is an AI framework developed by NeuralBlitz."
question = "Who developed fishstick?"

result = qa(question=question, context=context)
print(result['answer'])  # "NeuralBlitz"
```

---

## **5. Embedding Models (2 models)**

For similarity search, clustering, and semantic tasks.

| Model | Size | Description |
|-------|------|-------------|
| `sentence-transformers/all-MiniLM-L6-v2` | 22M | Best for similarity tasks |
| `sentence-transformers/all-distilroberta-v1` | 82M | Good quality embeddings |

### Usage Example
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = ["This is an example", "Another example sentence"]
embeddings = model.encode(sentences)

# Calculate similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
```

---

## **6. Code Models (3 models)**

For code generation, code completion, and code understanding.

| Model | Size | Description |
|-------|------|-------------|
| `microsoft/CodeGPT-small-py` | ~100M | Python code generation |
| `microsoft/codebert-base` | 125M | Code understanding |
| `Salesforce/codet5-small` | 60M | Code-to-code tasks |

### Usage Example
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("microsoft/CodeGPT-small-py")
tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-py")

code_prompt = "def fibonacci(n):"
inputs = tokenizer(code_prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

---

## **7. Multilingual Models (2 models)**

For non-English text processing.

| Model | Size | Languages |
|-------|------|-----------|
| `bert-base-multilingual-cased` | 179M | 104 languages |
| `distilbert-base-multilingual-cased` | 135M | 104 languages (fast) |

### Usage Example
```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("bert-base-multilingual-cased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# Works with 104 languages
texts = ["Hello world", "Hola mundo", "Bonjour le monde", "Hallo Welt"]
```

---

## **8. Summarization Models (2 models)**

For text summarization.

| Model | Size | Description |
|-------|------|-------------|
| `sshleifer/distilbart-cnn-12-6` | 66M | Fast summarization |
| `facebook/bart-large-cnn` | 406M | High quality summaries |

### Usage Example
```python
from transformers import pipeline

summarizer = pipeline("summarization", 
                     model="sshleifer/distilbart-cnn-12-6")

text = """Your long text here that you want to summarize..."""
summary = summarizer(text, max_length=50, min_length=10)
print(summary[0]['summary_text'])
```

---

## **Quick Reference: Model Sizes**

| Category | Best Models | Parameters | Speed |
|----------|------------|------------|-------|
| **Fast Inference** | distilgpt2, distilbert-base | 66-82M | ⚡⚡⚡ |
| **Balanced** | gpt2, bert-base | 110-124M | ⚡⚡ |
| **Better Quality** | gpt2-medium, roberta-base | 125-345M | ⚡ |
| **Slow but Good** | gpt2-xl, bart-large | 406M-1.5B | 🐌 |
| **Tiny/Testing** | bert-tiny, tiny-gpt2 | 1-11M | ⚡⚡⚡⚡ |

---

## **Recommended Models by Task**

### For Text Generation
1. 🥇 `distilgpt2` - Best speed/quality ratio
2. 🥈 `gpt2` - Better quality
3. 🥉 `gpt2-medium` - Best quality on CPU

### For Classification
1. 🥇 `distilbert-base-uncased` - Fast and good
2. 🥈 `roberta-base` - Better accuracy

### For Question Answering
1. 🥇 `distilbert-base-cased-distilled-squad` - Fast and accurate
2. 🥈 `bert-large-uncased-whole-word-masking-finetuned-squad` - Most accurate

### For Code
1. 🥇 `microsoft/CodeGPT-small-py` - Python code gen
2. 🥈 `microsoft/codebert-base` - Code understanding

### For Embeddings
1. 🥇 `sentence-transformers/all-MiniLM-L6-v2` - Best all-around

---

## **How to Use**

### Interactive Browser
```bash
python model_browser.py
```

### List All Models
```bash
python model_browser.py --list
```

### Quick Test
```bash
python model_browser.py --quick
```

### Direct Usage
```python
from transformers import AutoModel, AutoTokenizer

# Load any model
model = AutoModel.from_pretrained("distilgpt2")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
```

---

## **Storage Information**

**Location**: `~/.cache/huggingface/`

**Total Size**: Several GB (depends on which models you use)

**Check size**:
```bash
du -sh ~/.cache/huggingface/
```

**Clear cache** (if needed):
```bash
rm -rf ~/.cache/huggingface/
```

---

## **Failed Models**

These 3 models failed to download (missing `sentencepiece` dependency):

- ❌ `Helsinki-NLP/opus-mt-en-de` (English → German)
- ❌ `Helsinki-NLP/opus-mt-en-fr` (English → French)
- ❌ `Helsinki-NLP/opus-mt-en-es` (English → Spanish)

**To install them**:
```bash
pip install sentencepiece
python download_models.py
```

---

## **Next Steps**

1. **Test models**: Run `python model_browser.py` to interactively test
2. **Fine-tune**: Use `LLM_GUIDE.md` for fine-tuning examples
3. **Integrate**: Combine with fishstick frameworks for advanced AI systems
4. **Experiment**: Try different models for your specific tasks

---

**You now have 29 state-of-the-art language models ready to use!** 🎉