import re
import string

def normalize_title(text: str) -> str:
    """
    Normalize a journal title by:
    - converting to lowercase
    - removing punctuation
    - normalizing whitespace

    Returns empty string if input is None or not a string.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    return text