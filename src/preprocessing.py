import html
import re

import pandas as pd


def normalize_numbers(text: str) -> str:
    """
    Normalize abbreviated numeric forms while preserving their value.

    Examples:
        1,000 -> 1000
        10k -> 10000
        2.5m -> 2500000
    """

    # 1,000 -> 1000
    text = re.sub(r"(?<=\d),(?=\d)", "", text)

    def replace_number(match):
        value = float(match.group(1))
        suffix = match.group(2).lower()

        multiplier = {
            "k": 1_000,
            "m": 1_000_000,
            "b": 1_000_000_000,
        }[suffix]

        normalized = value * multiplier

        if normalized.is_integer():
            return str(int(normalized))

        return str(normalized)

    text = re.sub(
        r"\b(\d+(?:\.\d+)?)\s*([kmb])\b",
        replace_number,
        text,
        flags=re.IGNORECASE,
    )

    return text


def prepare_text_for_tfidf(text) -> str:
    """
    Apply basic text normalization before spaCy preprocessing for TF-IDF.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # HTML entities: &amp; -> &
    text = html.unescape(text)

    # HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # 10k -> 10000, 2.5m -> 2500000
    text = normalize_numbers(text)

    # Line breaks and tabs
    text = re.sub(r"[\n\r\t]+", " ", text)

    # Multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def prepare_text_for_features(text) -> str:
    """
    Apply minimal normalization for manual feature engineering.

    Stop words, punctuation and numbers are preserved.
    """

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # HTML entities: &amp; -> &
    text = html.unescape(text)

    # Remove HTML tags but preserve surrounding text
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize line breaks and tabs
    text = re.sub(r"[\n\r\t]+", " ", text)

    # Normalize multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def process_spacy_doc(doc) -> str:
    """
    Remove spaces and punctuation, preserve stop words,
    and lemmatize tokens.
    """

    tokens = []

    for token in doc:
        if token.is_space or token.is_punct:
            continue

        lemma = token.lemma_.lower().strip()

        if lemma:
            tokens.append(lemma)

    return " ".join(tokens)


def collect_unique_questions(*dataframes) -> pd.Series:
    """
    Collect unique questions from question1 and question2 columns.
    """

    questions = []

    for dataframe in dataframes:
        questions.append(dataframe["question1"])
        questions.append(dataframe["question2"])

    return (
        pd.concat(questions, ignore_index=True)
        .fillna("")
        .astype(str)
        .drop_duplicates()
        .reset_index(drop=True)
    )