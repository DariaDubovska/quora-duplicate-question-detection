import re

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")
STOP_WORDS = set(ENGLISH_STOP_WORDS)

def safe_ratio(numerator: np.ndarray, denominator: np.ndarray):
    """
    Divide two arrays safely.

    Returns 0 where the denominator is 0.
    """
    numerator = np.asarray(
        numerator,
        dtype=np.float32,
    )
    denominator = np.asarray(
        denominator,
        dtype=np.float32,
    )

    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator != 0,
    )

def create_basic_pair_features(
    data: pd.DataFrame,
    question1_col: str = "question1_features",
    question2_col: str = "question2_features",
):
    """
    Create symmetric lexical and length features
    for pairs of questions.
    """
    q1 = (
        data[question1_col]
        .fillna("")
        .astype(str)
    )

    q2 = (
        data[question2_col]
        .fillna("")
        .astype(str)
    )

    # Character counts
    q1_char_count = q1.str.len().to_numpy()
    q2_char_count = q2.str.len().to_numpy()

    min_char_count = np.minimum(
        q1_char_count,
        q2_char_count,
    )

    max_char_count = np.maximum(
        q1_char_count,
        q2_char_count,
    )

    # Tokenized questions
    q1_tokens = q1.str.split()
    q2_tokens = q2.str.split()

    q1_word_count = q1_tokens.str.len().to_numpy()
    q2_word_count = q2_tokens.str.len().to_numpy()

    min_word_count = np.minimum(
        q1_word_count,
        q2_word_count,
    )

    max_word_count = np.maximum(
        q1_word_count,
        q2_word_count,
    )

    # Unique word sets
    q1_word_sets = [
        set(tokens)
        for tokens in q1_tokens
    ]

    q2_word_sets = [
        set(tokens)
        for tokens in q2_tokens
    ]

    common_word_count = np.fromiter(
        (
            len(words1 & words2)
            for words1, words2
            in zip(q1_word_sets, q2_word_sets)
        ),
        dtype=np.int32,
        count=len(data),
    )

    word_union_count = np.fromiter(
        (
            len(words1 | words2)
            for words1, words2
            in zip(q1_word_sets, q2_word_sets)
        ),
        dtype=np.int32,
        count=len(data),
    )

    same_first_word = np.fromiter(
        (
            int(
                bool(tokens1)
                and bool(tokens2)
                and tokens1[0] == tokens2[0]
            )
            for tokens1, tokens2
            in zip(q1_tokens, q2_tokens)
        ),
        dtype=np.int8,
        count=len(data),
    )

    same_last_word = np.fromiter(
        (
            int(
                bool(tokens1)
                and bool(tokens2)
                and tokens1[-1] == tokens2[-1]
            )
            for tokens1, tokens2
            in zip(q1_tokens, q2_tokens)
        ),
        dtype=np.int8,
        count=len(data),
    )

    features = pd.DataFrame(
        {
            "min_char_count": min_char_count,
            "max_char_count": max_char_count,
            "char_count_diff": (
                max_char_count - min_char_count
            ),
            "char_count_ratio": safe_ratio(
                min_char_count,
                max_char_count,
            ),
            "min_word_count": min_word_count,
            "max_word_count": max_word_count,
            "word_count_diff": (
                max_word_count - min_word_count
            ),
            "word_count_ratio": safe_ratio(
                min_word_count,
                max_word_count,
            ),
            "common_word_count": common_word_count,
            "word_union_count": word_union_count,
            "word_jaccard": safe_ratio(
                common_word_count,
                word_union_count,
            ),
            "same_first_word": same_first_word,
            "same_last_word": same_last_word,
            "exact_text_match": (
                q1 == q2
            ).astype(np.int8).to_numpy(),
        },
        index=data.index,
    )

    return features

def tokenize_content_words(text: str):
    """
    Convert text to lowercase tokens and remove stop words.

    This preprocessing is used only for content-word overlap features.
    It does not modify the main TF-IDF text representation.
    """
    tokens = TOKEN_PATTERN.findall(
        str(text).lower()
    )

    return [
        token
        for token in tokens
        if token not in STOP_WORDS
    ]

def build_word_weights(
    data: pd.DataFrame,
    question1_col: str = "question1_features",
    question2_col: str = "question2_features",
):
    """
    Build inverse-frequency word weights using training data only.
    """
    word_counts = Counter()

    for column in [
        question1_col,
        question2_col,
    ]:
        for text in data[column].fillna(""):
            word_counts.update(
                tokenize_content_words(text)
            )

    word_weights = {}
    for word, count in word_counts.items():
        if count < 2:
            weight = 0.0
        else:
            weight = 1.0 / (count + 5000.0)

        word_weights[word] = weight

    return word_weights

def create_content_overlap_features(
    data: pd.DataFrame,
    word_weights: dict[str, float],
    question1_col: str = "question1_features",
    question2_col: str = "question2_features",
):
    """
    Create unweighted and frequency-weighted overlap features
    using content words without stop words.
    """
    q1 = (
        data[question1_col]
        .fillna("")
        .astype(str)
        .tolist()
    )

    q2 = (
        data[question2_col]
        .fillna("")
        .astype(str)
        .tolist()
    )

    content_word_match = np.zeros(
        len(data),
        dtype=np.float32,
    )

    weighted_content_word_match = np.zeros(
        len(data),
        dtype=np.float32,
    )

    for index, (text1, text2) in enumerate(
        zip(q1, q2)
    ):
        words1 = set(
            tokenize_content_words(text1)
        )

        words2 = set(
            tokenize_content_words(text2)
        )

        if not words1 or not words2:
            continue

        shared_words = words1 & words2

        # Similarity based on the number of shared content words
        content_word_match[index] = (
            2.0 * len(shared_words)
            / (len(words1) + len(words2))
        )

        # Rare shared words receive more weight
        shared_weight = 2.0 * sum(
            word_weights.get(word, 0.0)
            for word in shared_words
        )

        total_weight = (
            sum(
                word_weights.get(word, 0.0)
                for word in words1
            )
            +
            sum(
                word_weights.get(word, 0.0)
                for word in words2
            )
        )

        if total_weight > 0:
            weighted_content_word_match[index] = (
                shared_weight / total_weight
            )

    return pd.DataFrame(
        {
            "content_word_match":
                content_word_match,
            "weighted_content_word_match":
                weighted_content_word_match,
        },
        index=data.index,
    )
