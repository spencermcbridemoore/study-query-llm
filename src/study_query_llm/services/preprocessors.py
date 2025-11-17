"""
Prompt Preprocessor - Utilities for cleaning and formatting prompts.

This module provides tools for preprocessing user prompts before sending
them to LLM providers. All preprocessing is optional and configurable.

Usage:
    from study_query_llm.services.preprocessors import PromptPreprocessor

    preprocessor = PromptPreprocessor()

    # Clean whitespace
    clean = preprocessor.clean_whitespace("hello   world\\n\\n")
    # Returns: "hello world"

    # Apply template
    templated = preprocessor.apply_template(
        "What is 2+2?",
        "You are a helpful math tutor. {user_input}"
    )
    # Returns: "You are a helpful math tutor. What is 2+2?"
"""

import re
from typing import Optional


class PromptPreprocessor:
    """
    Utilities for preprocessing prompts before sending to LLM providers.

    All methods are static and can be used independently. The InferenceService
    can optionally apply these automatically when preprocessing is enabled.
    """

    @staticmethod
    def clean_whitespace(prompt: str) -> str:
        """
        Normalize whitespace in a prompt.

        Removes extra spaces, tabs, and newlines, replacing them with single spaces.
        Useful for cleaning up user input that may have inconsistent formatting.

        Args:
            prompt: The prompt to clean

        Returns:
            Prompt with normalized whitespace

        Example:
            >>> PromptPreprocessor.clean_whitespace("hello   world\\n\\n")
            "hello world"
            >>> PromptPreprocessor.clean_whitespace("  spaces  everywhere  ")
            "spaces everywhere"
        """
        # Split on any whitespace and rejoin with single spaces
        # This also strips leading/trailing whitespace
        return " ".join(prompt.split())

    @staticmethod
    def apply_template(prompt: str, template: str) -> str:
        """
        Apply a template to wrap the user's prompt.

        Templates can include system prompts, instructions, or few-shot examples.
        Use {user_input} as a placeholder for the user's prompt.

        Args:
            prompt: The user's prompt
            template: Template string with {user_input} placeholder

        Returns:
            Formatted prompt with template applied

        Example:
            >>> template = "You are a helpful assistant. {user_input}"
            >>> PromptPreprocessor.apply_template("Hello!", template)
            "You are a helpful assistant. Hello!"

            >>> template = "Question: {user_input}\\nAnswer:"
            >>> PromptPreprocessor.apply_template("What is 2+2?", template)
            "Question: What is 2+2?\\nAnswer:"
        """
        return template.format(user_input=prompt)

    @staticmethod
    def truncate(prompt: str, max_chars: int = 10000) -> str:
        """
        Truncate a prompt to a maximum character length.

        Helps prevent exceeding model context limits and reduces unnecessary costs.
        Adds "..." to indicate truncation.

        Args:
            prompt: The prompt to truncate
            max_chars: Maximum number of characters (default: 10000)

        Returns:
            Truncated prompt (or original if under limit)

        Example:
            >>> long_text = "a" * 15000
            >>> result = PromptPreprocessor.truncate(long_text, max_chars=10000)
            >>> len(result)
            10003  # 10000 chars + "..."
            >>> result.endswith("...")
            True
        """
        if len(prompt) > max_chars:
            return prompt[:max_chars] + "..."
        return prompt

    @staticmethod
    def remove_pii(prompt: str) -> str:
        """
        Remove basic PII (Personally Identifiable Information) from a prompt.

        This is a simple regex-based approach that redacts:
        - Email addresses
        - US phone numbers (various formats)

        Note: This is NOT comprehensive PII detection. For production use with
        sensitive data, consider specialized PII detection services.

        Args:
            prompt: The prompt to sanitize

        Returns:
            Prompt with PII redacted

        Example:
            >>> text = "Contact me at john@example.com or 555-123-4567"
            >>> PromptPreprocessor.remove_pii(text)
            "Contact me at [EMAIL] or [PHONE]"

            >>> text = "My email is test@domain.co.uk"
            >>> PromptPreprocessor.remove_pii(text)
            "My email is [EMAIL]"
        """
        # Remove email addresses
        # Matches: user@domain.com, user.name@sub.domain.co.uk, etc.
        prompt = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '[EMAIL]',
            prompt
        )

        # Remove US phone numbers
        # Matches: 555-123-4567, 555.123.4567, 5551234567, (555) 123-4567
        prompt = re.sub(
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            '[PHONE]',
            prompt
        )
        prompt = re.sub(
            r'\(\d{3}\)\s*\d{3}[-.]?\d{4}\b',
            '[PHONE]',
            prompt
        )

        return prompt

    @staticmethod
    def strip_control_characters(prompt: str) -> str:
        """
        Remove non-printable control characters from a prompt.

        Removes characters like null bytes, bell characters, etc. that
        could cause issues with certain APIs or databases.

        Args:
            prompt: The prompt to clean

        Returns:
            Prompt without control characters

        Example:
            >>> text = "Hello\\x00World\\x07"  # null byte and bell character
            >>> PromptPreprocessor.strip_control_characters(text)
            "HelloWorld"
        """
        # Remove control characters (ASCII 0-31 except newline, tab, carriage return)
        # Keep: \\n (10), \\r (13), \\t (9)
        return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', prompt)

    @staticmethod
    def preprocess(
        prompt: str,
        clean_whitespace: bool = True,
        truncate_prompts: bool = True,
        max_length: int = 10000,
        remove_pii: bool = False,
        strip_control_chars: bool = False,
        template: Optional[str] = None,
    ) -> str:
        """
        Apply multiple preprocessing steps in sequence.

        This is a convenience method that applies multiple preprocessing
        operations in a sensible order. Individual methods can also be
        called directly for more control.

        Processing order:
        1. Strip control characters (if enabled)
        2. Clean whitespace (if enabled)
        3. Remove PII (if enabled)
        4. Truncate (if enabled)
        5. Apply template (if provided)

        Args:
            prompt: The prompt to preprocess
            clean_whitespace: Normalize whitespace (default: True)
            truncate_prompts: Truncate to max_length (default: True)
            max_length: Maximum prompt length (default: 10000)
            remove_pii: Redact emails and phone numbers (default: False)
            strip_control_chars: Remove control characters (default: False)
            template: Optional template to apply (default: None)

        Returns:
            Preprocessed prompt

        Example:
            >>> preprocessor = PromptPreprocessor()
            >>> prompt = "  Contact: test@example.com  "
            >>> result = preprocessor.preprocess(
            ...     prompt,
            ...     clean_whitespace=True,
            ...     remove_pii=True
            ... )
            >>> result
            "Contact: [EMAIL]"
        """
        # Order matters! Apply in sequence:

        # 1. Strip control characters first (if enabled)
        if strip_control_chars:
            prompt = PromptPreprocessor.strip_control_characters(prompt)

        # 2. Clean whitespace
        if clean_whitespace:
            prompt = PromptPreprocessor.clean_whitespace(prompt)

        # 3. Remove PII (if enabled)
        if remove_pii:
            prompt = PromptPreprocessor.remove_pii(prompt)

        # 4. Truncate (do this before template to avoid truncating template)
        if truncate_prompts:
            prompt = PromptPreprocessor.truncate(prompt, max_length)

        # 5. Apply template last (so template wraps the processed prompt)
        if template:
            prompt = PromptPreprocessor.apply_template(prompt, template)

        return prompt
