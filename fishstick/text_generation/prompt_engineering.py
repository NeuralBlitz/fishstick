"""
Prompt Engineering Utilities
============================

Tools for prompt engineering including templates, few-shot learning,
chain-of-thought prompting, and dynamic prompt construction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import re


@dataclass
class TemplateVariable:
    """Represents a variable in a prompt template."""

    name: str
    description: str = ""
    default: Any = None
    required: bool = True

    def validate(self, value: Any) -> bool:
        """Validate that the variable value is acceptable."""
        if value is None and self.required:
            return False
        return True


class PromptTemplate(ABC):
    """Abstract base class for prompt templates."""

    @abstractmethod
    def format(self, **kwargs) -> str:
        """Format the template with provided variables."""
        pass

    @abstractmethod
    def get_variables(self) -> list[TemplateVariable]:
        """Get list of template variables."""
        pass


@dataclass
class BasicPromptTemplate(PromptTemplate):
    """Basic prompt template with simple variable substitution."""

    template: str
    variables: list[TemplateVariable] = field(default_factory=list)

    def format(self, **kwargs) -> str:
        result = self.template

        for var in self.variables:
            if var.name in kwargs:
                value = kwargs[var.name]
            elif var.default is not None:
                value = var.default
            elif var.required:
                raise ValueError(f"Required variable '{var.name}' not provided")
            else:
                value = ""

            placeholder = f"{{{var.name}}}"
            result = result.replace(placeholder, str(value))

        return result

    def get_variables(self) -> list[TemplateVariable]:
        return self.variables


@dataclass
class FewShotTemplate(PromptTemplate):
    """Few-shot learning prompt template with examples."""

    instruction: str
    examples: list[dict[str, str]]
    example_template: str = "Input: {input}\nOutput: {output}"
    variables: list[TemplateVariable] = field(default_factory=list)
    separator: str = "\n\n"

    def format(self, **kwargs) -> str:
        parts = [self.instruction]

        for example in self.examples:
            example_str = self.example_template.format(**example)
            parts.append(example_str)

        if kwargs:
            final_example = self.example_template.format(
                **{k: v for k, v in kwargs.items() if k in ["input", "output"]}
            )
            if kwargs.get("input"):
                example_with_input = self.example_template.replace(
                    "{input}", str(kwargs["input"])
                ).replace("{output}", "")
                parts.append(example_with_input.strip())

        return self.separator.join(parts)

    def get_variables(self) -> list[TemplateVariable]:
        return self.variables

    def add_example(self, input_text: str, output_text: str) -> None:
        """Add an example to the few-shot template."""
        self.examples.append({"input": input_text, "output": output_text})

    def clear_examples(self) -> None:
        """Clear all examples."""
        self.examples.clear()


@dataclass
class ChainOfThoughtTemplate(PromptTemplate):
    """Chain-of-thought prompting template."""

    instruction: str
    reasoning_prefix: str = "Let's think step by step:"
    answer_prefix: str = "Therefore,"
    include_reasoning: bool = True
    variables: list[TemplateVariable] = field(default_factory=list)

    def format(self, **kwargs) -> str:
        parts = [self.instruction]

        if self.include_reasoning and kwargs.get("show_reasoning", True):
            parts.append(self.reasoning_prefix)
            if "reasoning" in kwargs:
                parts.append(kwargs["reasoning"])

        if "answer" in kwargs:
            parts.append(f"{self.answer_prefix} {kwargs['answer']}")

        return "\n".join(parts)

    def get_variables(self) -> list[TemplateVariable]:
        return self.variables


class PromptFormatter:
    """Utility class for formatting and manipulating prompts."""

    def __init__(self, system_prompt: Optional[str] = None):
        self.system_prompt = system_prompt or ""
        self.conversation_history: list[dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.conversation_history.append({"role": role, "content": content})

    def format_chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Format messages as a chat prompt."""
        messages = []

        if system_prompt or self.system_prompt:
            messages.append(
                {"role": "system", "content": system_prompt or self.system_prompt}
            )

        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_message})

        return self._render_messages(messages)

    def _render_messages(self, messages: list[dict[str, str]]) -> str:
        """Render messages as a formatted string."""
        rendered = []
        for msg in messages:
            role = msg["role"].upper()
            rendered.append(f"{role}: {msg['content']}")
        return "\n".join(rendered)

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()

    @staticmethod
    def extract_variables(template: str) -> list[str]:
        """Extract variable names from a template string."""
        pattern = r"\{(\w+)\}"
        return re.findall(pattern, template)


class PromptOptimizer:
    """Optimizes prompts for better generation results."""

    @staticmethod
    def add_prefix(
        prompt: str,
        prefix: str = "You are a helpful assistant.",
        separator: str = "\n\n",
    ) -> str:
        """Add a prefix instruction to a prompt."""
        return f"{prefix}{separator}{prompt}"

    @staticmethod
    def add_suffix(
        prompt: str,
        suffix: str,
        separator: str = "\n\n",
    ) -> str:
        """Add a suffix instruction to a prompt."""
        return f"{prompt}{separator}{suffix}"

    @staticmethod
    def make_explicit(prompt: str) -> str:
        """Make implicit instructions more explicit."""
        transformations = [
            (r"\bthink\b", "reason step by step"),
            (r"\banswer\b", "provide a clear and detailed answer"),
            (r"\bexplain\b", "explain in detail with reasoning"),
            (r"\bsummarize\b", "provide a concise summary"),
        ]

        result = prompt
        for pattern, replacement in transformations:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        return result

    @staticmethod
    def add_formatting_instructions(
        prompt: str,
        format_type: str = "plaintext",
    ) -> str:
        """Add instructions for output formatting."""
        format_instructions = {
            "json": "Format your response as valid JSON.",
            "markdown": "Use Markdown formatting for your response.",
            "bullet": "Use bullet points to organize your response.",
            "numbered": "Use numbered lists to organize your response.",
            "plaintext": "Use plain text without special formatting.",
        }

        instruction = format_instructions.get(
            format_type.lower(), format_instructions["plaintext"]
        )

        return f"{prompt}\n\n{instruction}"


def create_qa_template(
    instruction: str = "Answer the following question:",
    include_context: bool = True,
) -> FewShotTemplate:
    """Create a QA prompt template."""
    if include_context:
        template = BasicPromptTemplate(
            template="Context: {context}\n\nQuestion: {question}\n\nAnswer:",
            variables=[
                TemplateVariable("context", "Supporting context", required=False),
                TemplateVariable("question", "The question to answer", required=True),
            ],
        )
    else:
        template = FewShotTemplate(
            instruction=instruction,
            examples=[],
            example_template="Question: {question}\nAnswer: {answer}",
            variables=[
                TemplateVariable("question", "The question to answer"),
                TemplateVariable("answer", "The answer", required=False),
            ],
        )

    return template


def create_classification_template(
    classes: list[str],
    instruction: str = "Classify the following text:",
) -> FewShotTemplate:
    """Create a classification prompt template."""
    class_list = ", ".join(classes)
    template = FewShotTemplate(
        instruction=f"{instruction}\n\nAvailable classes: {class_list}",
        examples=[],
        example_template="Text: {text}\nClass: {label}",
        variables=[
            TemplateVariable("text", "Text to classify"),
            TemplateVariable("label", "Predicted class", required=False),
        ],
    )
    return template
