from __future__ import annotations

import json
import os
from pathlib import Path
from types import TracebackType
from typing import Mapping

from .document import Document
from .searcher import Hit, Ranker, Searcher
from .segment import Segment, SegmentReader
from .query import Query


MANIFEST_NAME = "manifest.json"


class LocalIndex:
    def __init__(self, directory: str | os.PathLike[str]):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.segment = self._load_latest_segment()

    def writer(self) -> "IndexWriter":
        return IndexWriter(self)

    def get_document(self, doc_id: str) -> Document | None:
        return self.segment.documents.get(doc_id)

    def all_documents(self) -> dict[str, Document]:
        return dict(self.segment.documents)

    def search(
        self,
        query: Query,
        ranker: Ranker | None = None,
        reverse: bool = True,
    ) -> list[Hit]:
        return Searcher(self.segment).search(query, ranker, reverse)

    def reload(self) -> None:
        self.segment = self._load_latest_segment()

    def _load_latest_segment(self) -> Segment:
        manifest_path = self.directory / MANIFEST_NAME
        if not manifest_path.exists():
            return Segment([], 0)
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        segment = SegmentReader.read(self.directory / manifest["active_segment"])
        if segment.generation != manifest["generation"]:
            raise ValueError("manifest generation does not match active segment")
        return segment


class IndexWriter:
    def __init__(self, index: LocalIndex):
        self.index = index
        self._upserts: dict[str, Document] = dict(index.segment.documents)
        self._deletions: set[str] = set()
        self._closed = False

    def upsert_document(
        self,
        doc_id: str,
        fields: Mapping[str, str],
        version: int | None = None,
    ) -> None:
        self._ensure_open()
        self._deletions.discard(doc_id)
        current = self._upserts.get(doc_id)
        next_version = version if version is not None else (current.version + 1 if current else 1)
        self._upserts[doc_id] = Document.from_mapping(doc_id, fields, next_version)

    def delete_document(self, doc_id: str) -> None:
        self._ensure_open()
        self._upserts.pop(doc_id, None)
        self._deletions.add(doc_id)

    def commit(self) -> int:
        self._ensure_open()
        next_generation = self.index.segment.generation + 1
        segment = Segment(list(self._upserts.values()), next_generation)
        segment_file = segment.write(self.index.directory)
        manifest = {
            "active_segment": segment_file.name,
            "generation": next_generation,
        }
        manifest_path = self.index.directory / MANIFEST_NAME
        temporary_path = manifest_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(manifest, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary_path, manifest_path)
        self.index.segment = segment
        self._deletions.clear()
        return next_generation

    def rollback(self) -> None:
        self._ensure_open()
        self._upserts.clear()
        self._deletions.clear()
        self._closed = True

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "IndexWriter":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is None and not self._closed:
            self.commit()
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("writer is closed")
