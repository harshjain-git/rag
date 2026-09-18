"""
Text Cleaner Module for Conservative Preprocessing.
Cleans raw extracted page text without removing structural content or research tables.
"""

import re


def clean_text(text: str) -> str:
    """
    Applies conservative text cleaning to raw extracted PDF text.
    
    Cleaning rules:
    1. Remove non-printable control characters.
    2. Fix hyphenated words split across lines (e.g., 'evalua-\\ntion' -> 'evaluation').
    3. Replace multiple horizontal spaces with a single space.
    4. Normalize excessive newlines (max 2 consecutive newlines).
    5. Strip leading and trailing whitespace.
    """
    if not text:
        return ""

    # 1. Remove control characters (except standard newlines and tabs)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

    # 2. Fix hyphenation at line breaks (word-\nword -> wordword)
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

    # 3. Replace multiple horizontal spaces/tabs on a line with a single space
    text = re.sub(r'[ \t]+', ' ', text)

    # 4. Normalize multiple consecutive newlines to maximum of two (\n\n)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 5. Clean up spaces at line starts/ends
    lines = [line.strip() for line in text.split('\n')]
    cleaned_text = '\n'.join(lines).strip()

    return cleaned_text
