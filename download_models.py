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
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoModelForObjectDetection,
    AutoModelForVision2Seq,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
)
import os
import traceback

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
    # ========== NEW MODELS ==========
    # Additional LLM Models
    ("EleutherAI/gpt-neo-1.3B", "Text Generation - GPT-Neo 1.3B"),
    ("EleutherAI/gpt-j-6b", "Text Generation - GPT-J (6B)"),
    ("EleutherAI/pythia-2.8b", "Text Generation - Pythia 2.8B"),
    ("facebook/opt-125m", "Text Generation - OPT (125M)"),
    ("facebook/opt-350m", "Text Generation - OPT (350M)"),
    ("Salesforce/opt-1.3b", "Text Generation - OPT 1.3B"),
    ("EleutherAI/gpt-neox-20b", "Text Generation - GPT-NeoX (20B) - SLOW"),
    ("togethercomputer/RedPajama-INCITE-7B-Chat", "Chat - RedPajama (7B)"),
    ("togethercomputer/RedPajama-INCITE-7B-Base", "Text - RedPajama Base (7B)"),
    # Additional Embedding Models
    (
        "sentence-transformers/all-mpnet-base-v2",
        "Embeddings - MPNet (110M) - Best quality",
    ),
    (
        "sentence-transformers/msmarco-bert-base-diversev2",
        "Embeddings - MS MARCO (110M)",
    ),
    (
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "Embeddings - Multilingual MPNet",
    ),
    (
        "sentence-transformers/quora-distilbert-multilingual",
        "Embeddings - Quora (Multilingual)",
    ),
    ("intfloat/e5-base-v2", "Embeddings - E5 Base (110M)"),
    ("intfloat/e5-large-v2", "Embeddings - E5 Large (340M)"),
    ("BAAI/bge-base-en-v1.5", "Embeddings - BGE Base (110M)"),
    ("BAAI/bge-large-en-v1.5", "Embeddings - BGE Large (340M)"),
    ("BAAI/bge-small-en-v1.5", "Embeddings - BGE Small (33M)"),
    ("google/flan-t5-small", "Embeddings/T5 - FLAN-T5 Small (80M)"),
    ("google/flan-t5-base", "Embeddings/T5 - FLAN-T5 Base (250M)"),
    ("google/flan-t5-large", "Embeddings/T5 - FLAN-T5 Large (780M)"),
    # Vision Models
    ("microsoft/resnet-50", "Vision - ResNet50 (ImageNet)"),
    ("microsoft/resnet-101", "Vision - ResNet101"),
    ("facebook/convnext-tiny-224", "Vision - ConvNeXt Tiny"),
    ("facebook/convnext-small-224", "Vision - ConvNeXt Small"),
    ("facebook/convnext-base-224", "Vision - ConvNeXt Base"),
    ("facebook/convnext-large-224", "Vision - ConvNeXt Large"),
    ("google/vit-base-patch16-224", "Vision - ViT Base (ImageNet)"),
    ("google/vit-large-patch16-224", "Vision - ViT Large"),
    ("google/swin-tiny-patch4-window7-224", "Vision - Swin Tiny"),
    ("google/swin-small-patch4-window7-224", "Vision - Swin Small"),
    ("google/swin-base-patch4-window7-224", "Vision - Swin Base"),
    ("microsoft/detr-resnet-50", "Vision - DETR (Object Detection)"),
    ("facebook/detr-resnet-101", "Vision - DETR 101 (Object Detection)"),
    ("facebook/detr-resnet-50-panoptic", "Vision - DETR (Panoptic)"),
    ("openai/clip-vit-base-patch32", "Vision - CLIP (Zero-shot)"),
    ("openai/clip-vit-large-patch14", "Vision - CLIP Large (Zero-shot)"),
    ("Salesforce/blip-image-captioning-base", "Vision - BLIP (Image Captioning)"),
    ("Salesforce/blip-2-opt-2.7b", "Vision - BLIP-2 (Image Captioning)"),
    ("microsoft/beit-base-patch16-224", "Vision - BEiT Base"),
    ("microsoft/beit-large-patch16-224", "Vision - BEiT Large"),
    # Audio/Speech Models
    ("openai/whisper-base", "Audio - Whisper Base (EN)"),
    ("openai/whisper-small", "Audio - Whisper Small"),
    ("openai/whisper-medium", "Audio - Whisper Medium"),
    ("openai/whisper-large", "Audio - Whisper Large"),
    ("facebook/wav2vec2-base", "Audio - Wav2Vec2 Base (ASR)"),
    ("facebook/wav2vec2-large", "Audio - Wav2Vec2 Large (ASR)"),
    ("facebook/hubert-base", "Audio - HuBERT Base"),
    ("facebook/hubert-large", "Audio - HuBERT Large"),
    ("microsoft/speecht5_tts", "Audio - SpeechT5 TTS"),
    ("microsoft/speecht5_hifigan", "Audio - SpeechT5 HiFi-GAN (Vocoder)"),
    ("facebook/fairseq-wav2vec2-xlsr-53", "Audio - XLSR-53 (Multilingual ASR)"),
    ("jonatasgrosman/wav2vec2-large-xlsr-53-english", "Audio - Wav2Vec2 English"),
    ("elgeish/wav2vec2-large-xlsr-53-arabic", "Audio - Wav2Vec2 Arabic"),
    ("facebook/mms-1b", "Audio - MMS 1B (Massively Multilingual Speech)"),
    # Multimodal Models
    ("Salesforce/blip-2-opt-2.7b", "Multimodal - BLIP-2 (Image-to-Text)"),
    ("llava-hf/llava-1.5-7b-hf", "Multimodal - LLaVA 1.5 (7B)"),
    ("llava-hf/llava-1.5-13b-hf", "Multimodal - LLaVA 1.5 (13B)"),
    ("Salesforce/instructblip-7b", "Multimodal - InstructBLIP (7B)"),
    ("Salesforce/instructblip-13b", "Multimodal - InstructBLIP (13B)"),
    ("microsoft/visual-bert-vqa", "Multimodal - VisualBERT (VQA)"),
    ("dandelin/vilt-b32-finetuned-vqa", "Multimodal - ViLT (VQA)"),
    # Additional Specialized Models
    ("microsoft/deberta-v3-base", "Specialized - DeBERTa Base (100M)"),
    ("microsoft/deberta-v3-large", "Specialized - DeBERTa Large (300M)"),
    ("microsoft/deberta-v2-xlarge", "Specialized - DeBERTa v2 XLarge (900M)"),
    ("AllenAI/Longformer-base-4096", "Specialized - Longformer (150M)"),
    ("AllenAI/Longformer-large-4096", "Specialized - Longformer Large (340M)"),
    ("princeton-nlp/spanberta-base", "Specialized - SpanBERTa"),
    ("BART-finetuned-xsum", "Summarization - BART XSum"),
    ("mrm8488/t5-base-finetuned-summarize-news", "Summarization - T5 News"),
    ("facebook/mbart-large-50-many-to-many-mt", "Translation - mBART (Many-to-Many)"),
    ("google/mt5-base", "Translation - mT5 Base"),
    ("google/byt5-base", "ByT5 - Character-level"),
    ("T5-base", "Text-to-Text - T5 Base"),
    ("T5-large", "Text-to-Text - T5 Large"),
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
            model_type = None

            # Vision models
            if any(
                x in model_name.lower()
                for x in [
                    "vit",
                    "resnet",
                    "convnext",
                    "swin",
                    "clip",
                    "detr",
                    "beit",
                    "blip",
                ]
            ):
                if "clip" in model_name.lower():
                    from transformers import CLIPProcessor, CLIPModel

                    model = CLIPModel.from_pretrained(model_name)
                    processor = CLIPProcessor.from_pretrained(model_name)
                    model_type = "vision"
                elif "detr" in model_name.lower():
                    from transformers import DetrImageProcessor, DetrForObjectDetection

                    model = DetrForObjectDetection.from_pretrained(model_name)
                    processor = DetrImageProcessor.from_pretrained(model_name)
                    model_type = "vision"
                elif "blip" in model_name.lower():
                    from transformers import BlipProcessor, BlipForConditionalGeneration

                    model = BlipForConditionalGeneration.from_pretrained(model_name)
                    processor = BlipProcessor.from_pretrained(model_name)
                    model_type = "vision"
                else:
                    model = AutoModelForImageClassification.from_pretrained(model_name)
                    processor = AutoImageProcessor.from_pretrained(model_name)
                    model_type = "vision"
            # Audio/Speech models
            elif any(
                x in model_name.lower()
                for x in ["whisper", "wav2vec", "hubert", "speecht5", "mms", "xlsr"]
            ):
                if "whisper" in model_name.lower():
                    from transformers import (
                        WhisperProcessor,
                        WhisperForConditionalGeneration,
                    )

                    model = WhisperForConditionalGeneration.from_pretrained(model_name)
                    processor = WhisperProcessor.from_pretrained(model_name)
                else:
                    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
                    processor = AutoProcessor.from_pretrained(model_name)
                model_type = "audio"
            # T5/FLAN models
            elif "t5" in model_name.lower() or "flan" in model_name.lower():
                from transformers import T5ForConditionalGeneration

                model = T5ForConditionalGeneration.from_pretrained(model_name)
                model_type = "text-to-text"
            # mBART/mT5
            elif "mbart" in model_name.lower() or "mt5" in model_name.lower():
                from transformers import MBartForConditionalGeneration

                model = MBartForConditionalGeneration.from_pretrained(model_name)
                model_type = "translation"
            # Question Answering
            elif "squad" in model_name.lower():
                model = AutoModelForQuestionAnswering.from_pretrained(model_name)
                model_type = "qa"
            # Classification/Sentiment/NER
            elif any(
                x in description.lower()
                for x in ["sentiment", "sst", "ner", "classification"]
            ):
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                model_type = "classification"
            # Fill-mask
            elif "fill-mask" in description.lower() or "roberta" in model_name.lower():
                model = AutoModelForMaskedLM.from_pretrained(model_name)
                model_type = "masked-lm"
            # LLM models (GPT family, etc.)
            elif any(
                x in model_name.lower()
                for x in [
                    "gpt",
                    "pythia",
                    "bloom",
                    "llama",
                    "opt",
                    "codegen",
                    "redpajama",
                ]
            ):
                model = AutoModelForCausalLM.from_pretrained(model_name)
                model_type = "llm"
            # Default to base model
            else:
                model = AutoModel.from_pretrained(model_name)
                model_type = "base"

            print("✓")
            success_count += 1
            print(f"    Status: Successfully downloaded! ({model_type})")

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
