import math
import re
from collections import Counter

import numpy as np


TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{1,}")
STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "because",
    "been",
    "between",
    "but",
    "can",
    "from",
    "has",
    "have",
    "into",
    "not",
    "that",
    "the",
    "their",
    "then",
    "there",
    "this",
    "with",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "will",
    "your",
}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text) if token.lower() not in STOPWORDS]


class LocalTfidfVectorizer:
    """Compact local vectorizer for PDF chunk retrieval."""

    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}
        self.idf: np.ndarray | None = None

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        tokenized = [tokenize(text) for text in texts]
        document_frequency: Counter[str] = Counter()

        for tokens in tokenized:
            document_frequency.update(set(tokens))

        terms = sorted(document_frequency)
        self.vocabulary = {term: index for index, term in enumerate(terms)}
        total_docs = max(len(texts), 1)
        self.idf = np.array(
            [math.log((1 + total_docs) / (1 + document_frequency[term])) + 1 for term in terms],
            dtype=np.float32,
        )
        return self._transform_tokenized(tokenized)

    def transform(self, texts: list[str]) -> np.ndarray:
        return self._transform_tokenized([tokenize(text) for text in texts])

    def _transform_tokenized(self, tokenized_texts: list[list[str]]) -> np.ndarray:
        if self.idf is None:
            raise ValueError("Vectorizer must be fitted before transform.")

        matrix = np.zeros((len(tokenized_texts), len(self.vocabulary)), dtype=np.float32)
        for row, tokens in enumerate(tokenized_texts):
            counts = Counter(token for token in tokens if token in self.vocabulary)
            if not counts:
                continue
            max_count = max(counts.values())
            for token, count in counts.items():
                column = self.vocabulary[token]
                matrix[row, column] = (count / max_count) * self.idf[column]

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.clip(norms, 1e-8, None)
