import pickle
from pathlib import Path
from typing import Any


class DocumentStore:
    """Small disk-backed index store for demo-friendly persistence."""

    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict[str, Any]] = {}

    def save(self, doc_id: str, payload: dict[str, Any]) -> None:
        path = self._path(doc_id)
        with path.open("wb") as file:
            pickle.dump(payload, file)
        self._cache[doc_id] = payload

    def load(self, doc_id: str) -> dict[str, Any]:
        cached = self._cache.get(doc_id)
        if cached is not None:
            return cached

        path = self._path(doc_id)
        if not path.exists():
            raise FileNotFoundError(doc_id)
        with path.open("rb") as file:
            payload = pickle.load(file)

        self._cache[doc_id] = payload
        return payload

    def _path(self, doc_id: str) -> Path:
        safe_id = "".join(char for char in doc_id if char.isalnum() or char in {"-", "_"})
        return self.index_dir / f"{safe_id}.pkl"
