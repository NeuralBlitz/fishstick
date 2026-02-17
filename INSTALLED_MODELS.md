# Installed Models - Complete Catalog

This document lists all available models in your fishstick environment.

## **Summary**
- 📦 **Storage location**: `~/.cache/huggingface/`
- ⚡ **Status**: Ready to use instantly (no re-download needed)

---

## **1. Text Generation / LLM Models**

Perfect for creative writing, chatbots, code generation, and text completion.

### GPT-2 Family
| Model | Size | Speed | Best For |
|-------|------|-------|----------|
| `distilgpt2` | 82M | ⚡⚡⚡ Fast | General text, testing |
| `gpt2` | 124M | ⚡⚡ Good | General text generation |
| `gpt2-medium` | 345M | ⚡ Slow | Better quality text |
| `gpt2-xl` | 1.5B | 🐌 Very Slow | Best quality (avoid on CPU) |

### EleutherAI Models
| Model | Size | Description |
|-------|------|-------------|
| `EleutherAI/gpt-neo-125M` | 125M | Open-source GPT alternative |
| `EleutherAI/gpt-neo-1.3B` | 1.3B | Larger GPT-Neo |
| `EleutherAI/gpt-j-6b` | 6B | GPT-J (powerful, requires GPU) |
| `EleutherAI/pythia-160m` | 160M | EleutherAI's interpretable model |
| `EleutherAI/pythia-2.8b` | 2.8B | Larger Pythia |
| `EleutherAI/gpt-neox-20b` | 20B | GPT-NeoX (requires GPU) |

### OPT Models (Meta)
| Model | Size | Description |
|-------|------|-------------|
| `facebook/opt-125m` | 125M | OPT small |
| `facebook/opt-350m` | 350M | OPT medium |
| `Salesforce/opt-1.3b` | 1.3B | OPT 1.3B |

### RedPajama Models
| Model | Size | Description |
|-------|------|-------------|
| `togethercomputer/RedPajama-INCITE-7B-Base` | 7B | Base model |
| `togethercomputer/RedPajama-INCITE-7B-Chat` | 7B | Chat model |

### T5/FLAN Models
| Model | Size | Description |
|-------|------|-------------|
| `google/flan-t5-small` | 80M | FLAN-T5 Small |
| `google/flan-t5-base` | 250M | FLAN-T5 Base |
| `google/flan-t5-large` | 780M | FLAN-T5 Large |
| `T5-base` | 220M | T5 Base |
| `T5-large` | 770M | T5 Large |

### Tiny Models (for testing)
| Model | Size | Description |
|-------|------|-------------|
| `sshleifer/tiny-gpt2` | ~1M | Ultra-small for testing |
| `prajjwal1/bert-tiny` | 4M | Ultra-fast testing |
| `prajjwal1/bert-mini` | 11M | Fast testing |

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

## **2. BERT & Understanding Models**

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

### DeBERTa Models (Microsoft)
| Model | Size | Description |
|-------|------|-------------|
| `microsoft/deberta-v3-base` | 100M | DeBERTa v3 Base |
| `microsoft/deberta-v3-large` | 300M | DeBERTa v3 Large |
| `microsoft/deberta-v2-xlarge` | 900M | DeBERTa v2 XLarge |

### Specialized Models
| Model | Size | Description |
|-------|------|-------------|
| `AllenAI/Longformer-base-4096` | 150M | Long document processing |
| `AllenAI/Longformer-large-4096` | 340M | Long document processing |

---

## **3. Classification & NER Models**

Pre-trained for specific classification tasks.

| Model | Task | Description |
|-------|------|-------------|
| `distilbert-base-uncased-finetuned-sst-2-english` | Sentiment | Positive/Negative classification |
| `dslim/bert-base-NER` | NER | Named Entity Recognition |
| `dslim/distilbert-NER` | NER | Fast NER version |

---

## **4. Question Answering Models**

Extract answers from context text.

| Model | Size | Description |
|-------|------|-------------|
| `distilbert-base-cased-distilled-squad` | 66M | Fast QA on SQuAD |
| `bert-large-uncased-whole-word-masking-finetuned-squad` | 340M | High accuracy QA |

---

## **5. Embedding Models**

For similarity search, clustering, and semantic tasks.

### Sentence Transformers
| Model | Size | Description |
|-------|------|-------------|
| `sentence-transformers/all-MiniLM-L6-v2` | 22M | Best for similarity tasks |
| `sentence-transformers/all-distilroberta-v1` | 82M | Good quality embeddings |
| `sentence-transformers/all-mpnet-base-v2` | 110M | Best quality |
| `sentence-transformers/msmarco-bert-base-diversev2` | 110M | MS MARCO ranking |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 110M | Multilingual MPNet |
| `sentence-transformers/quora-distilbert-multilingual` | 110M | Quora (Multilingual) |

### E5 Embeddings
| Model | Size | Description |
|-------|------|-------------|
| `intfloat/e5-base-v2` | 110M | E5 Base |
| `intfloat/e5-large-v2` | 340M | E5 Large |

### BGE Embeddings
| Model | Size | Description |
|-------|------|-------------|
| `BAAI/bge-small-en-v1.5` | 33M | BGE Small |
| `BAAI/bge-base-en-v1.5` | 110M | BGE Base |
| `BAAI/bge-large-en-v1.5` | 340M | BGE Large |

---

## **6. Vision Models**

For image classification, object detection, and image understanding.

### Image Classification
| Model | Size | Description |
|-------|------|-------------|
| `microsoft/resnet-50` | 25M | ResNet50 (ImageNet) |
| `microsoft/resnet-101` | 44M | ResNet101 |
| `facebook/convnext-tiny-224` | 28M | ConvNeXt Tiny |
| `facebook/convnext-small-224` | 50M | ConvNeXt Small |
| `facebook/convnext-base-224` | 89M | ConvNeXt Base |
| `facebook/convnext-large-224` | 198M | ConvNeXt Large |
| `google/vit-base-patch16-224` | 86M | ViT Base |
| `google/vit-large-patch16-224` | 304M | ViT Large |
| `google/swin-tiny-patch4-window7-224` | 28M | Swin Tiny |
| `google/swin-small-patch4-window7-224` | 50M | Swin Small |
| `google/swin-base-patch4-window7-224` | 88M | Swin Base |
| `microsoft/beit-base-patch16-224` | 86M | BEiT Base |
| `microsoft/beit-large-patch16-224` | 304M | BEiT Large |

### Object Detection
| Model | Size | Description |
|-------|------|-------------|
| `microsoft/detr-resnet-50` | 69M | DETR (Object Detection) |
| `facebook/detr-resnet-101` | 81M | DETR 101 |

### Zero-shot Image Classification (CLIP)
| Model | Size | Description |
|-------|------|-------------|
| `openai/clip-vit-base-patch32` | 151M | CLIP Base |
| `openai/clip-vit-large-patch14` | 407M | CLIP Large |

### Image Captioning
| Model | Size | Description |
|-------|------|-------------|
| `Salesforce/blip-image-captioning-base` | 224M | BLIP Base |
| `Salesforce/blip-2-opt-2.7b` | 3.9B | BLIP-2 (requires GPU) |

---

## **7. Audio/Speech Models**

For speech recognition, text-to-speech, and audio processing.

### Whisper (OpenAI)
| Model | Size | Languages |
|-------|------|-----------|
| `openai/whisper-base` | 74M | English |
| `openai/whisper-small` | 244M | Multilingual |
| `openai/whisper-medium` | 769M | Multilingual |
| `openai/whisper-large` | 1550M | Multilingual |

### Wav2Vec2 (Meta)
| Model | Size | Description |
|-------|------|-------------|
| `facebook/wav2vec2-base` | 79M | Base ASR |
| `facebook/wav2vec2-large` | 317M | Large ASR |
| `facebook/hubert-base` | 95M | HuBERT Base |
| `facebook/hubert-large` | 317M | HuBERT Large |
| `facebook/fairseq-wav2vec2-xlsr-53` | 300M | XLSR-53 (Multilingual) |

### Text-to-Speech
| Model | Size | Description |
|-------|------|-------------|
| `microsoft/speecht5_tts` | 268M | SpeechT5 TTS |
| `microsoft/speecht5_hifigan` | 24M | HiFi-GAN Vocoder |

### Multilingual
| Model | Size | Description |
|-------|------|-------------|
| `facebook/mms-1b` | 1B | MMS 1B (Massively Multilingual) |
| `jonatasgrosman/wav2vec2-large-xlsr-53-english` | 317M | English ASR |
| `elgeish/wav2vec2-large-xlsr-53-arabic` | 317M | Arabic ASR |

---

## **8. Code Models**

For code generation, code completion, and code understanding.

| Model | Size | Description |
|-------|------|-------------|
| `microsoft/CodeGPT-small-py` | ~100M | Python code generation |
| `microsoft/codebert-base` | 125M | Code understanding |
| `Salesforce/codet5-small` | 60M | Code-to-code tasks |

---

## **9. Multilingual Models**

For non-English text processing.

| Model | Size | Languages |
|-------|------|-----------|
| `bert-base-multilingual-cased` | 179M | 104 languages |
| `distilbert-base-multilingual-cased` | 135M | 104 languages (fast) |
| `google/mt5-base` | 300M | Multilingual T5 |
| `google/byt5-base` | 300M | Character-level |
| `facebook/mbart-large-50-many-to-many-mt` | 680M | Many-to-Many MT |

---

## **10. Summarization Models**

For text summarization.

| Model | Size | Description |
|-------|------|-------------|
| `sshleifer/distilbart-cnn-12-6` | 66M | Fast summarization |
| `facebook/bart-large-cnn` | 406M | High quality summaries |
| `BART-finetuned-xsum` | 406M | XSum dataset |
| `mrm8488/t5-base-finetuned-summarize-news` | 220M | News summarization |

---

## **11. Translation Models**

| Model | Description |
|-------|-------------|
| `Helsinki-NLP/opus-mt-en-de` | English → German |
| `Helsinki-NLP/opus-mt-en-fr` | English → French |
| `Helsinki-NLP/opus-mt-en-es` | English → Spanish |

---

## **12. API Integrations**

External API clients are available in `fishstick.integrations`:

### OpenAI
```python
from fishstick.integrations import OpenAIClient

client = OpenAIClient(api_key="sk-...", model="gpt-4")
response = client.generate("Write a haiku about AI")
```

### Anthropic (Claude)
```python
from fishstick.integrations import AnthropicClient

client = AnthropicClient(api_key="sk-ant-...", model="claude-3-sonnet-20240229")
response = client.generate("Explain quantum computing")
```

### Google AI (Gemini)
```python
from fishstick.integrations import GoogleAIClient

client = GoogleAIClient(api_key="...", model="gemini-pro")
response = client.generate("What is machine learning?")
```

### Hugging Face
```python
from fishstick.integrations import HuggingFaceClient

client = HuggingFaceClient(api_key="hf_...", model="gpt2")
response = client.generate("Once upon a time")
```

### Cohere
```python
from fishstick.integrations import CohereClient

client = CohereClient(api_key="...")
response = client.generate("Write a tagline for an AI company")
```

---

## **Quick Reference: Model Sizes**

| Category | Best Models | Parameters | Speed |
|----------|------------|------------|-------|
| **Fast Inference** | distilgpt2, distilbert-base, bert-tiny | 4-82M | ⚡⚡⚡⚡ |
| **Balanced** | gpt2, bert-base | 110-125M | ⚡⚡ |
| **Better Quality** | gpt2-medium, roberta-base | 125-345M | ⚡ |
| **Slow but Good** | gpt2-xl, bart-large | 406M-1.5B | 🐌 |
| **GPU Only** | GPT-J, GPT-NeoX, BLIP-2 | 6B+ | Requires GPU |

---

## **How to Use**

### Download New Models
```bash
python download_models.py
```

### Interactive Browser
```bash
python model_browser.py
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

**Check size**:
```bash
du -sh ~/.cache/huggingface/
```

---

**You now have 100+ state-of-the-art models ready to use!**

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