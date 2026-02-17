#!/usr/bin/env python3
"""
Model Browser and Tester for fishstick
Browse and test all downloaded models interactively
"""

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    pipeline,
)
import torch

# Available models catalog
MODELS = {
    "Text Generation": {
        "gpt2": "GPT-2 Base (124M) - General purpose",
        "gpt2-medium": "GPT-2 Medium (345M) - Better quality",
        "distilgpt2": "DistilGPT-2 (82M) - Fast, good quality",
        "EleutherAI/gpt-neo-125M": "GPT-Neo (125M) - Open source GPT",
        "EleutherAI/pythia-160m": "Pythia (160M) - EleutherAI model",
        "gpt2-xl": "GPT-2 XL (1.5B) - Best quality, SLOW on CPU",
        "sshleifer/tiny-gpt2": "Tiny GPT-2 - For testing",
    },
    "Understanding (BERT)": {
        "bert-base-uncased": "BERT Base Uncased (110M)",
        "bert-base-cased": "BERT Base Cased (110M)",
        "distilbert-base-uncased": "DistilBERT (66M) - Fast",
        "roberta-base": "RoBERTa Base (125M)",
        "albert-base-v2": "ALBERT Base (12M) - Lightweight",
        "distilroberta-base": "DistilRoBERTa (82M)",
        "prajjwal1/bert-tiny": "BERT Tiny (4M) - Ultra fast",
        "prajjwal1/bert-mini": "BERT Mini (11M) - Fast",
    },
    "Classification": {
        "distilbert-base-uncased-finetuned-sst-2-english": "Sentiment Analysis",
        "dslim/bert-base-NER": "Named Entity Recognition",
        "dslim/distilbert-NER": "NER (fast version)",
    },
    "Question Answering": {
        "distilbert-base-cased-distilled-squad": "QA - DistilBERT Squad",
        "bert-large-uncased-whole-word-masking-finetuned-squad": "QA - BERT Large",
    },
    "Embeddings": {
        "sentence-transformers/all-MiniLM-L6-v2": "MiniLM (22M) - Best for similarity",
        "sentence-transformers/all-distilroberta-v1": "DistilRoBERTa (82M)",
    },
    "Code Models": {
        "microsoft/CodeGPT-small-py": "CodeGPT Python",
        "microsoft/codebert-base": "CodeBERT (125M)",
        "Salesforce/codet5-small": "CodeT5 (60M)",
    },
    "Multilingual": {
        "bert-base-multilingual-cased": "mBERT (179M)",
        "distilbert-base-multilingual-cased": "DistilBERT Multilingual (135M)",
    },
    "Summarization": {
        "sshleifer/distilbart-cnn-12-6": "DistilBART CNN (66M)",
        "facebook/bart-large-cnn": "BART Large CNN (406M)",
    },
}


def test_text_generation(model_name="gpt2"):
    """Test text generation model"""
    print(f"\n{'=' * 60}")
    print(f"TESTING: {model_name}")
    print("=" * 60)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    prompts = [
        "The future of AI is",
        "In machine learning,",
        "The fishstick framework",
    ]

    print("\nGenerating text...\n")
    for prompt in prompts:
        print(f"Prompt: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        continuation = generated[len(prompt) :].strip()

        print(f"→ {continuation}\n")

    print("✓ Test complete!\n")


def test_classification(model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    """Test classification model"""
    print(f"\n{'=' * 60}")
    print(f"TESTING: {model_name}")
    print("=" * 60)

    print("Loading model...")
    classifier = pipeline("sentiment-analysis", model=model_name)

    texts = [
        "This is absolutely amazing!",
        "I hate this so much.",
        "It's okay, nothing special.",
        "The fishstick framework is incredible!",
    ]

    print("\nClassifying texts:\n")
    for text in texts:
        result = classifier(text)[0]
        emoji = "😊" if result["label"] == "POSITIVE" else "😞"
        print(f'{emoji} "{text}"')
        print(f"   → {result['label']} ({result['score']:.1%})\n")

    print("✓ Test complete!\n")


def test_qa(model_name="distilbert-base-cased-distilled-squad"):
    """Test question answering model"""
    print(f"\n{'=' * 60}")
    print(f"TESTING: {model_name}")
    print("=" * 60)

    print("Loading model...")
    qa_pipeline = pipeline("question-answering", model=model_name)

    context = """
    The fishstick framework is a mathematically rigorous AI framework that 
    combines theoretical physics, formal mathematics, and advanced machine learning. 
    It implements 6 unified frameworks including Hamiltonian Neural Networks, 
    Sheaf-Optimized Attention, and Renormalization Group flows. The framework 
    is developed by NeuralBlitz.
    """

    questions = [
        "What is fishstick?",
        "How many frameworks does it implement?",
        "Who developed fishstick?",
    ]

    print(f"\nContext: {context.strip()}\n")
    print("Questions and Answers:\n")

    for question in questions:
        result = qa_pipeline(question=question, context=context)
        print(f"Q: {question}")
        print(f"A: {result['answer']} (confidence: {result['score']:.1%})\n")

    print("✓ Test complete!\n")


def test_fill_mask(model_name="distilroberta-base"):
    """Test fill-mask model"""
    print(f"\n{'=' * 60}")
    print(f"TESTING: {model_name}")
    print("=" * 60)

    print("Loading model...")
    fill_mask = pipeline("fill-mask", model=model_name)

    texts = [
        "The fishstick framework combines <mask> and machine learning.",
        "Machine learning is a subset of <mask>.",
        "The future of AI is <mask>.",
    ]

    print("\nFilling masks:\n")
    for text in texts:
        print(f"Input: {text}")
        results = fill_mask(text)
        print("Top predictions:")
        for i, result in enumerate(results[:3], 1):
            print(f"  {i}. {result['token_str']} ({result['score']:.1%})")
        print()

    print("✓ Test complete!\n")


def test_ner(model_name="dslim/distilbert-NER"):
    """Test named entity recognition"""
    print(f"\n{'=' * 60}")
    print(f"TESTING: {model_name}")
    print("=" * 60)

    print("Loading model...")
    ner = pipeline("ner", model=model_name, aggregation_strategy="simple")

    texts = [
        "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
        "The fishstick framework was developed by NeuralBlitz in 2026.",
        "Google and Microsoft are major tech companies based in the USA.",
    ]

    print("\nExtracting entities:\n")
    for text in texts:
        print(f"Text: {text}")
        entities = ner(text)
        if entities:
            print("Entities found:")
            for entity in entities:
                print(
                    f"  - {entity['word']} ({entity['entity_group']}) - {entity['score']:.1%}"
                )
        else:
            print("  No entities found")
        print()

    print("✓ Test complete!\n")


def list_all_models():
    """Display all available models"""
    print("\n" + "=" * 70)
    print("AVAILABLE MODELS IN fishstick")
    print("=" * 70)

    for category, models in MODELS.items():
        print(f"\n{category}:")
        print("-" * 60)
        for model_id, description in models.items():
            print(f"  • {description}")
            print(f"    ID: {model_id}")


def interactive_mode():
    """Interactive model testing"""
    while True:
        print("\n" + "=" * 60)
        print("MODEL TESTER - Choose an option:")
        print("=" * 60)
        print("1. List all available models")
        print("2. Test text generation (GPT-2)")
        print("3. Test text generation (DistilGPT-2)")
        print("4. Test text generation (GPT-2 Medium)")
        print("5. Test classification (Sentiment)")
        print("6. Test question answering")
        print("7. Test fill-mask")
        print("8. Test named entity recognition")
        print("9. Test specific model (advanced)")
        print("0. Exit")

        choice = input("\nEnter choice (0-9): ").strip()

        if choice == "0":
            print("\nGoodbye!")
            break
        elif choice == "1":
            list_all_models()
        elif choice == "2":
            test_text_generation("gpt2")
        elif choice == "3":
            test_text_generation("distilgpt2")
        elif choice == "4":
            test_text_generation("gpt2-medium")
        elif choice == "5":
            test_classification()
        elif choice == "6":
            test_qa()
        elif choice == "7":
            test_fill_mask()
        elif choice == "8":
            test_ner()
        elif choice == "9":
            model_id = input(
                "Enter model ID (e.g., 'gpt2', 'bert-base-uncased'): "
            ).strip()
            if model_id:
                test_text_generation(model_id)
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        list_all_models()
    elif len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick test of a few models
        print("Running quick tests...")
        test_text_generation("distilgpt2")
        test_classification()
        test_qa()
    else:
        interactive_mode()
