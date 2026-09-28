from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .analyzer import tokenize
from .document import Document


class Segment:
    def __init__(self, documents: list[Document], generation: int):
        self.generation = generation
        self.documents: dict[str, Document] = {
            document.doc_id: document for document in sorted(documents, key=lambda item: item.doc_id)
        }
        if len(self.documents) != len(documents):
            raise ValueError("duplicate document id in committed segment")
        self.postings: dict[str, dict[str, dict[str, list[int]]]] = {}
        self.lengths: dict[str, dict[str, int]] = {}
        for document in self.documents.values():
            for field, value in document.fields.items():
                tokens = tokenize(value)
                self.lengths.setdefault(document.doc_id, {})[field] = len(tokens)
                for token in tokens:
                    field_postings = self.postings.setdefault(field, {}).setdefault(token.text, {})
                    field_postings.setdefault(document.doc_id, []).append(token.position)

    @property
    def doc_count(self) -> int:
        return len(self.documents)

    def to_payload(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "documents": [
                {
                    "doc_id": document.doc_id,
                    "version": document.version,
                    "fields": document.fields,
                }
                for document in self.documents.values()
            ],
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Segment":
        documents = [
            Document(
                item["doc_id"],
                item["fields"],
                item["version"],
            )
            for item in payload["documents"]
        ]
        return cls(documents, payload["generation"])

    def write(self, directory: str | os.PathLike[str]) -> Path:
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"segment-{self.generation:012d}.json"
        temporary_path = path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(self.to_payload(), ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
        return path


class SegmentReader:
    @staticmethod
    def read(path: str | os.PathLike[str]) -> Segment:
        with Path(path).open("r", encoding="utf-8") as handle:
            return Segment.from_payload(json.load(handle))
