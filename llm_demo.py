#!/usr/bin/env python3
"""
Simple LLM Demo for fishstick Environment
Run this to see how LLMs work with transformers
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def simple_text_generation():
    """Generate text with GPT-2"""
    print("=" * 70)
    print("SIMPLE TEXT GENERATION DEMO")
    print("=" * 70)

    # Load small, fast model
    print("\nLoading DistilGPT-2 (82M parameters)...")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model.eval()

    print("Model loaded!\n")

    # Test prompts
    prompts = [
        "The fishstick framework is",
        "In the future, AI will",
        "The best way to learn machine learning is",
        "Once upon a time",
    ]

    for prompt in prompts:
        print(f"Prompt: {prompt}")

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=40,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        continuation = generated[len(prompt) :].strip()

        print(f"→ {continuation}")
        print("-" * 70)

    print("\nDemo complete!")


def sentiment_analysis():
    """Classify text sentiment"""
    print("\n" + "=" * 70)
    print("SENTIMENT ANALYSIS DEMO")
    print("=" * 70)

    from transformers import pipeline

    print("\nLoading sentiment classifier...")
    classifier = pipeline(
        "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    texts = [
        "I love this framework! It's amazing!",
        "This is the worst experience ever.",
        "The code works fine, nothing special.",
        "fishstick is the most innovative AI framework I've seen!",
    ]

    print("\nAnalyzing texts:\n")
    for text in texts:
        result = classifier(text)[0]
        emoji = "😊" if result["label"] == "POSITIVE" else "😞"
        print(f'{emoji} "{text}"')
        print(f"   → {result['label']} (confidence: {result['score']:.1%})\n")


def text_completion():
    """Complete partial sentences"""
    print("\n" + "=" * 70)
    print("TEXT COMPLETION DEMO")
    print("=" * 70)

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model.eval()

    # Partial code/text
    partials = [
        "def calculate_fibonacci(n):",
        "import torch\nimport numpy as np\n\n# Define model",
        "The theory of relativity states that",
    ]

    print("\nCompleting partial texts:\n")
    for partial in partials:
        print(f"Input:\n{partial}")

        inputs = tokenizer(partial, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        completed = tokenizer.decode(outputs[0], skip_special_tokens=True)
        new_text = completed[len(partial) :]

        print(f"\nCompletion:\n{new_text}")
        print("=" * 70)


def question_answering():
    """Answer questions from context"""
    print("\n" + "=" * 70)
    print("QUESTION ANSWERING DEMO")
    print("=" * 70)

    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    import torch.nn.functional as F

    print("\nLoading QA model...")
    model = AutoModelForQuestionAnswering.from_pretrained(
        "distilbert-base-cased-distilled-squad"
    )
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
    model.eval()

    # Context about fishstick
    context = """
    fishstick is a mathematically rigorous AI framework that combines theoretical physics, 
    formal mathematics, and advanced machine learning. It implements 6 unified frameworks 
    including Hamiltonian Neural Networks, Sheaf-Optimized Attention, and Renormalization 
    Group flows. The framework is developed by NeuralBlitz.
    """

    questions = [
        "What is fishstick?",
        "How many frameworks does it implement?",
        "Who developed fishstick?",
        "What components does it include?",
    ]

    print(f"\nContext: {context.strip()}\n")
    print("Questions and Answers:\n")

    for question in questions:
        inputs = tokenizer(question, context, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        # Get answer span
        answer_start = outputs.start_logits.argmax()
        answer_end = outputs.end_logits.argmax() + 1

        # Convert to tokens
        answer_tokens = inputs["input_ids"][0][answer_start:answer_end]
        answer = tokenizer.decode(answer_tokens)

        # Get confidence
        start_prob = F.softmax(outputs.start_logits, dim=1).max().item()
        end_prob = F.softmax(outputs.end_logits, dim=1).max().item()
        confidence = (start_prob + end_prob) / 2

        print(f"Q: {question}")
        print(f"A: {answer} (confidence: {confidence:.1%})\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LLM DEMONSTRATION FOR fishstick ENVIRONMENT")
    print("=" * 70)
    print("\nThis demo shows various LLM capabilities available in your environment.")
    print("Note: CPU-only, so using small models for reasonable speed.\n")

    try:
        # Run demos
        simple_text_generation()
        sentiment_analysis()
        text_completion()
        question_answering()

        print("\n" + "=" * 70)
        print("ALL DEMOS COMPLETE!")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("• You can generate text with models like GPT-2")
        print("• You can classify text (sentiment, topics, etc.)")
        print("• You can answer questions from context")
        print("• Everything runs on CPU (slow but works!)")
        print("\nSee LLM_GUIDE.md for more examples and details.")

    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure you have an internet connection to download models.")
