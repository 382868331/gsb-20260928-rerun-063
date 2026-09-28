import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from .query import Clause, Query
from .ranking import BM25Ranker, Ranker
from .segment import (
    Segment,
    StoredDocument,
    atomic_write_json,
    build_segment,
    write_segment,
)

MANIFEST_NAME = "manifest.json"


class SearchEngine:
    def __init__(self, directory: str | os.PathLike[str] = ".search_index", ranker: Optional[Ranker] = None) -> None:
        self.directory = Path(directory)
        self.segments_dir = self.directory / "segments"
        self.manifest_path = self.directory / MANIFEST_NAME
        self.ranker = ranker or BM25Ranker()
        self._staged_updates: Dict[str, Dict[str, object]] = {}
        self._staged_deletes: set[str] = set()
        self.segments: List[Segment] = []
        self.segment_sequences: Dict[str, int] = {}
        self.deletes: Dict[str, int] = {}
        self.sequence = 0
        self._load()

    def upsert(self, doc_id: str, fields: Dict[str, object]) -> None:
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError("Document id must be a non-empty string")
        if not isinstance(fields, dict):
            raise ValueError("Document fields must be an object")
        self._staged_updates[doc_id] = dict(fields)
        self._staged_deletes.discard(doc_id)

    def delete(self, doc_id: str) -> None:
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError("Document id must be a non-empty string")
        self._staged_updates.pop(doc_id, None)
        self._staged_deletes.add(doc_id)

    def commit(self) -> bool:
        if not self._staged_updates and not self._staged_deletes:
            return False

        next_sequence = self.sequence + 1
        segment_id = f"seg-{next_sequence:012d}"
        segment_path: Optional[str] = None
        documents = [StoredDocument(doc_id, fields) for doc_id, fields in self._staged_updates.items()]

        if documents:
            payload = build_segment(segment_id, documents)
            payload["sequence"] = next_sequence
            segment_path = os.path.relpath(write_segment(str(self.segments_dir), segment_id, payload), self.directory)

        manifest = {
            "sequence": next_sequence,
            "segments": [
                {"id": segment.segment_id, "sequence": self.segment_sequences[segment.segment_id], "path": os.path.relpath(segment.path, self.directory)}
                for segment in self.segments
            ],
            "deletes": dict(self.deletes),
        }
        if segment_path is not None:
            manifest["segments"].append({"id": segment_id, "sequence": next_sequence, "path": segment_path})
        for doc_id in self._staged_deletes:
            manifest["deletes"][doc_id] = next_sequence

        atomic_write_json(str(self.manifest_path), manifest)
        self._load()
        self._staged_updates.clear()
        self._staged_deletes.clear()
        return True

    def get(self, doc_id: str) -> Optional[Dict[str, object]]:
        doc, _ = self._documents_view()
        doc = doc.get(doc_id)
        return None if doc is None else dict(doc.fields)

    def search(self, query_text: str, limit: Optional[int] = None) -> List[Dict[str, object]]:
        query = query_text if isinstance(query_text, Query) else self._parse(query_text)
        visible, newest = self._documents_view()
        matching_ids = self._matching_ids(query, newest)
        matches = [visible[doc_id] for doc_id in matching_ids if doc_id in visible]
        scores = self.ranker.score(matches, query)
        ordered = sorted(matches, key=lambda doc: (-scores.get(doc.id, 0.0), doc.id))
        if limit is not None:
            ordered = ordered[:limit]
        return [{"id": doc.id, "score": scores.get(doc.id, 0.0), "fields": dict(doc.fields)} for doc in ordered]

    def all_documents(self) -> List[Dict[str, object]]:
        return [
            {"id": doc.id, "fields": dict(doc.fields)}
            for doc in sorted(self._documents_view()[0].values(), key=lambda item: item.id)
        ]

    def _parse(self, text: str) -> Query:
        from .query import parse_query

        return parse_query(text)

    def _documents_view(self) -> tuple[Dict[str, StoredDocument], Dict[str, int]]:
        visible: Dict[str, StoredDocument] = {}
        newest: Dict[str, int] = {}
        for segment in sorted(self.segments, key=lambda item: self.segment_sequences[item.segment_id]):
            sequence = self.segment_sequences[segment.segment_id]
            for doc_id, doc in segment.documents.items():
                if sequence > self.deletes.get(doc_id, -1):
                    visible[doc_id] = doc
                    newest[doc_id] = sequence
                elif newest.get(doc_id, -1) <= self.deletes.get(doc_id, -1):
                    visible.pop(doc_id, None)
        return visible, newest

    def _matching_ids(self, query: Query, newest: Dict[str, int]) -> set[str]:
        candidates = set(newest)
        for clause in query.clauses:
            candidates &= self._clause_matches(clause, newest)
        return candidates

    def _clause_matches(self, clause: Clause, newest: Dict[str, int]) -> set[str]:
        matches: set[str] = set()
        for segment in self.segments:
            sequence = self.segment_sequences[segment.segment_id]
            fields = [clause.field] if clause.field is not None else list(segment.postings)
            for field in fields:
                term_postings = segment.postings.get(field, {})
                first_posting = term_postings.get(clause.terms[0], {})
                for doc_id, positions in first_posting.items():
                    if newest.get(doc_id) != sequence:
                        continue
                    if not clause.phrase:
                        if all(term in term_postings and doc_id in term_postings[term] for term in clause.terms):
                            matches.add(doc_id)
                    elif self._positions_form_phrase(term_postings, doc_id, clause.terms):
                        matches.add(doc_id)
        return matches

    def _positions_form_phrase(
        self, term_postings: Dict[str, Dict[str, List[int]]], doc_id: str, terms: List[str]
    ) -> bool:
        starts = term_postings.get(terms[0], {}).get(doc_id, [])
        return any(
            all(
                start + offset in term_postings.get(term, {}).get(doc_id, [])
                for offset, term in enumerate(terms)
            )
            for start in starts
        )

    def _load(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.segments_dir.mkdir(exist_ok=True)
        if not self.manifest_path.exists():
            self.segments = []
            self.segment_sequences = {}
            self.deletes = {}
            self.sequence = 0
            return

        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.sequence = int(manifest["sequence"])
        self.segments = []
        self.segment_sequences = {}
        for entry in manifest.get("segments", []):
            path = self.directory / entry["path"]
            segment = Segment.load(entry["id"], str(path))
            self.segments.append(segment)
            self.segment_sequences[entry["id"]] = int(entry["sequence"])
        self.deletes = {doc_id: int(sequence) for doc_id, sequence in manifest.get("deletes", {}).items()}
