"""Shared text preprocessing for model training and inference."""

import re
import string


def clean_text(text: str) -> str:
    """Normalize SMS text using the same rules at training and inference time."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()
