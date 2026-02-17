#!/usr/bin/env python3
"""
Download and cache popular LLM models for training and inference.
This script downloads models locally so they're ready to use.
"""

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForMaskedLM,
    AutoModel,
)
import os

# Create models directory
os.makedirs("./models", exist_ok=True)

print("=" * 70)
print("DOWNLOADING LLM MODELS FOR fishstick")
print("=" * 70)
print("\nThis will download models to your local cache.")
print("Models will be stored in: ~/.cache/huggingface/")
print("\nDownloading... (this may take a while)\n")

models_to_download = [
    # Text Generation Models (GPT family)
    ("gpt2", "Text Generation - Base (124M)"),
    ("gpt2-medium", "Text Generation - Medium (345M)"),
    ("distilgpt2", "Text Generation - DistilGPT2 (82M)"),
    ("EleutherAI/gpt-neo-125M", "Text Generation - GPT-Neo (125M)"),
    ("EleutherAI/pythia-160m", "Text Generation - Pythia (160M)"),
    # BERT and variants (Understanding tasks)
    ("bert-base-uncased", "BERT - Base Uncased (110M)"),
    ("bert-base-cased", "BERT - Base Cased (110M)"),
    ("distilbert-base-uncased", "DistilBERT - Uncased (66M)"),
    ("distilbert-base-uncased-finetuned-sst-2-english", "DistilBERT - Sentiment"),
    ("roberta-base", "RoBERTa - Base (125M)"),
    ("albert-base-v2", "ALBERT - Base (12M)"),
    # Multilingual models
    ("bert-base-multilingual-cased", "mBERT - Multilingual (179M)"),
    ("distilbert-base-multilingual-cased", "DistilBERT - Multilingual (135M)"),
    # Code models
    ("microsoft/CodeGPT-small-py", "Code - Python GPT (Small)"),
    ("microsoft/codebert-base", "CodeBERT - Base (125M)"),
    ("Salesforce/codet5-small", "CodeT5 - Small (60M)"),
    # Instruction-tuned / Chat models
    ("sshleifer/tiny-gpt2", "Tiny GPT-2 for testing"),
    ("gpt2-xl", "Text Generation - XL (1.5B) - SLOW on CPU"),
    # Question Answering
    ("distilbert-base-cased-distilled-squad", "QA - DistilBERT Squad"),
    ("bert-large-uncased-whole-word-masking-finetuned-squad", "QA - BERT Large Squad"),
    # Named Entity Recognition
    ("dslim/bert-base-NER", "NER - BERT"),
    ("dslim/distilbert-NER", "NER - DistilBERT"),
    # Summarization
    ("sshleifer/distilbart-cnn-12-6", "Summarization - DistilBART (66M)"),
    ("facebook/bart-large-cnn", "Summarization - BART Large (406M)"),
    # Translation
    ("Helsinki-NLP/opus-mt-en-de", "Translation - English to German"),
    ("Helsinki-NLP/opus-mt-en-fr", "Translation - English to French"),
    ("Helsinki-NLP/opus-mt-en-es", "Translation - English to Spanish"),
    # Fill-mask (Masked Language Models)
    ("distilroberta-base", "Fill-Mask - DistilRoBERTa (82M)"),
    # Sentence embeddings / Similarity
    ("sentence-transformers/all-MiniLM-L6-v2", "Embeddings - MiniLM (22M)"),
    ("sentence-transformers/all-distilroberta-v1", "Embeddings - DistilRoBERTa (82M)"),
    # Small models for testing
    ("prajjwal1/bert-tiny", "Tiny BERT - for testing (4M)"),
    ("prajjwal1/bert-mini", "Mini BERT - for testing (11M)"),
]

success_count = 0
fail_count = 0
failed_models = []

for model_name, description in models_to_download:
    print(
        f"\n[{success_count + fail_count + 1}/{len(models_to_download)}] {description}"
    )
    print(f"    Model: {model_name}")

    try:
        # Download tokenizer first (smaller, faster)
        print(f"    Downloading tokenizer...", end=" ", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✓")

        # Determine model type and download
        print(f"    Downloading model...", end=" ", flush=True)

        # Try to infer model type from the name or download generically
        try:
            # Try causal LM first (most common)
            if any(
                x in model_name.lower()
                for x in ["gpt", "pythia", "bloom", "llama", "opt", "codegen"]
            ):
                model = AutoModelForCausalLM.from_pretrained(model_name)
            elif "squad" in model_name.lower():
                model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            elif (
                "sentiment" in description.lower()
                or "sst" in model_name.lower()
                or "ner" in model_name.lower()
            ):
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
            elif "fill-mask" in description.lower() or "roberta" in model_name.lower():
                model = AutoModelForMaskedLM.from_pretrained(model_name)
            else:
                # Default to causal LM
                model = AutoModelForCausalLM.from_pretrained(model_name)

            print("✓")
            success_count += 1
            print(f"    Status: Successfully downloaded!")

        except Exception as model_error:
            # Try alternative model types
            try:
                model = AutoModel.from_pretrained(model_name)
                print("✓ (Generic model)")
                success_count += 1
            except:
                raise model_error

    except Exception as e:
        print(f"✗ Failed: {str(e)[:60]}")
        fail_count += 1
        failed_models.append((model_name, description, str(e)[:100]))

# Summary
print("\n" + "=" * 70)
print("DOWNLOAD SUMMARY")
print("=" * 70)
print(f"\nSuccessfully downloaded: {success_count}/{len(models_to_download)} models")
print(f"Failed: {fail_count}/{len(models_to_download)} models")

if failed_models:
    print("\nFailed models:")
    for model_name, desc, error in failed_models:
        print(f"  ✗ {desc} ({model_name})")
        print(f"    Error: {error}")

print("\n" + "=" * 70)
print("MODELS READY TO USE!")
print("=" * 70)
print("\nAll downloaded models are cached in: ~/.cache/huggingface/")
print("They will load instantly on next use (no re-download needed).")
print("\nExample usage:")
print("  from transformers import AutoModel, AutoTokenizer")
print('  model = AutoModel.from_pretrained("gpt2")')
print('  tokenizer = AutoTokenizer.from_pretrained("gpt2")')

print("\n" + "=" * 70)
print("STORAGE INFORMATION")
print("=" * 70)
print("\nApproximate disk usage:")
print("  Small models (~100M params): ~400MB each")
print("  Medium models (~300M params): ~1.2GB each")
print("  Large models (~1B+ params): ~4GB+ each")
print("\nTotal estimated: Several GB depending on which models downloaded")
print("\nTo see exact size, run: du -sh ~/.cache/huggingface/")
