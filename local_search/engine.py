import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .analysis import SynonymRules, field_tokens
from .matcher import ClauseHit, match_query
from .query import Query, parse_query
from .ranking import BM25Ranker, Ranker
from .segment import (
    Segment,
    StoredDocument,
    atomic_write_json,
    build_segment,
    write_segment,
)

MANIFEST_NAME = "manifest.json"
RULES_NAME = "rules.json"
HISTORY_DIRNAME = "history"


@dataclass(frozen=True)
class Snapshot:
    """Read-only view bound to one published manifest and rules version."""

    sequence: int
    documents: Dict[str, StoredDocument]
    rules: SynonymRules
    ranker: Ranker

    @property
    def rules_version(self) -> int:
        return self.rules.version

    def get(self, doc_id: str) -> Optional[Dict[str, object]]:
        doc = self.documents.get(doc_id)
        return None if doc is None else dict(doc.fields)

    def all_documents(self) -> List[Dict[str, object]]:
        return [
            {"id": doc.id, "fields": dict(doc.fields)}
            for doc in sorted(self.documents.values(), key=lambda item: item.id)
        ]

    def search(
        self,
        query_text: "str | Query",
        limit: Optional[int] = None,
        *,
        explain: bool = True,
        expand: Optional[bool] = None,
    ) -> List[Dict[str, object]]:
        query = query_text if isinstance(query_text, Query) else parse_query(query_text)
        if expand is not None:
            clauses = tuple(replace(clause, expand=expand or clause.expand) for clause in query.clauses)
            query = Query(list(clauses), expand or query.expand)

        matches: List[StoredDocument] = []
        hit_by_id: Dict[str, List[ClauseHit]] = {}
        for doc in sorted(self.documents.values(), key=lambda item: item.id):
            tokenized = field_tokens(doc.fields)
            hits = match_query(doc.fields, tokenized, query, self.rules)
            if hits is not None:
                matches.append(doc)
                hit_by_id[doc.id] = hits

        scores = self.ranker.score(matches, query)
        explanation = {}
        if explain and hasattr(self.ranker, "explain"):
            explanation = self.ranker.explain(matches, query, hit_by_id)
        ordered = sorted(matches, key=lambda doc: (-scores.get(doc.id, 0.0), doc.id))
        if limit is not None:
            ordered = ordered[:limit]
        return [self._result(doc, scores.get(doc.id, 0.0), hit_by_id[doc.id], explanation.get(doc.id), explain) for doc in ordered]

    def _result(
        self,
        doc: StoredDocument,
        score: float,
        hits: Sequence[ClauseHit],
        score_detail: Optional[object],
        explain: bool,
    ) -> Dict[str, object]:
        result: Dict[str, object] = {"id": doc.id, "score": score, "fields": dict(doc.fields)}
        if explain:
            result["explanation"] = {
                "rules_version": self.rules.version,
                "clauses": [
                    {
                        "raw": hit.raw,
                        "field": hit.matched_field,
                        "phrase": hit.phrase,
                        "expanded": hit.expanded,
                        "terms": [
                            {
                                "term": term.term,
                                "match": term.kind,
                                "matched": term.matched_token,
                                "positions": list(term.positions),
                            }
                            for term in hit.terms
                        ],
                    }
                    for hit in hits
                ],
                "score": score_detail,
            }
        return result

class RulesVersionError(RuntimeError):
    pass

class SearchEngine:
    def __init__(self, directory: str | os.PathLike[str] = ".search_index", ranker: Optional[Ranker] = None) -> None:
        self.directory = Path(directory)
        self.segments_dir = self.directory / "segments"
        self.history_dir = self.directory / HISTORY_DIRNAME
        self.manifest_path = self.directory / MANIFEST_NAME
        self.rules_path = self.directory / RULES_NAME
        self.ranker = ranker or BM25Ranker()
        self._staged_updates: Dict[str, Dict[str, object]] = {}
        self._staged_deletes: set[str] = set()
        self._staged_synonyms: Optional[Dict[str, List[str]]] = None
        self.segments: List[Segment] = []
        self.segment_sequences: Dict[str, int] = {}
        self.deletes: Dict[str, int] = {}
        self.sequence = 0
        self.rules_version = 0
        self.rules = SynonymRules.empty(0)
        self._saved_rules_versions: Dict[int, SynonymRules] = {}
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

    def set_synonyms(self, mapping: Dict[str, Sequence[str]]) -> int:
        """Stage a new immutable synonym rules version for the next commit."""
        if not isinstance(mapping, dict):
            raise ValueError("Synonym rules must be an object")
        normalized: Dict[str, List[str]] = {}
        for term, alternatives in mapping.items():
            if not isinstance(alternatives, (list, tuple)):
                raise ValueError("Synonym alternatives must be a list")
            normalized[str(term)] = [str(item) for item in alternatives]
        self._staged_synonyms = normalized
        return self.rules_version + 1

    def synonyms(self) -> Dict[str, List[str]]:
        return self.rules.to_mapping()

    def commit(self) -> bool:
        if not self._staged_updates and not self._staged_deletes and self._staged_synonyms is None:
            return False

        next_sequence = self.sequence + 1
        segment_id = f"seg-{next_sequence:012d}"
        segment_path: Optional[str] = None
        documents = [StoredDocument(doc_id, fields) for doc_id, fields in self._staged_updates.items()]

        if documents:
            payload = build_segment(segment_id, documents)
            payload["sequence"] = next_sequence
            segment_path = os.path.relpath(write_segment(str(self.segments_dir), segment_id, payload), self.directory)

        next_rules = self.rules
        if self._staged_synonyms is not None:
            next_rules = SynonymRules.from_mapping(self._staged_synonyms, version=self.rules_version + 1)

        manifest = {
            "sequence": next_sequence,
            "rules_version": next_rules.version,
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

        self._archive_current_manifest()
        atomic_write_json(str(self.manifest_path), manifest)
        if next_rules.version != self.rules_version:
            self._write_rules(next_rules)
        self._load()
        self._staged_updates.clear()
        self._staged_deletes.clear()
        self._staged_synonyms = None
        return True

    def compact(self) -> bool:
        """Merge visible documents into one new segment atomically.

        Old segment files and manifest history are retained, so published
        snapshots remain queryable; an interrupted compaction never publishes
        a partial manifest.
        """
        visible, _ = self._documents_view()
        next_sequence = self.sequence + 1
        segment_id = f"seg-{next_sequence:012d}"
        documents = [StoredDocument(doc_id, stored.fields) for doc_id, stored in visible.items()]
        payload = build_segment(segment_id, documents)
        payload["sequence"] = next_sequence
        segment_relpath = os.path.relpath(write_segment(str(self.segments_dir), segment_id, payload), self.directory)
        manifest = {
            "sequence": next_sequence,
            "rules_version": self.rules_version,
            "segments": [{"id": segment_id, "sequence": next_sequence, "path": segment_relpath}],
            "deletes": dict(self.deletes),
        }
        self._archive_current_manifest()
        atomic_write_json(str(self.manifest_path), manifest)
        self._load()
        return True

    def get(self, doc_id: str) -> Optional[Dict[str, object]]:
        return self.snapshot().get(doc_id)

    def search(
        self,
        query_text: "str | Query",
        limit: Optional[int] = None,
        *,
        explain: bool = True,
        expand: Optional[bool] = None,
    ) -> List[Dict[str, object]]:
        return self.snapshot().search(query_text, limit, explain=explain, expand=expand)

    def snapshot(self, sequence: Optional[int] = None) -> Snapshot:
        """Return an immutable query snapshot for the current or a past version."""
        if sequence is None or sequence == self.sequence:
            visible, _ = self._documents_view()
            return Snapshot(self.sequence, visible, self.rules, self.ranker)
        if sequence < 0 or sequence > self.sequence:
            raise KeyError(f"Unknown snapshot version {sequence}")
        path = self.history_dir / f"manifest-{sequence:012d}.json"
        return self._snapshot_from_manifest(self._read_manifest_file(path), path)

    def snapshots(self) -> List[int]:
        versions: List[int] = []
        if self.manifest_path.exists():
            versions.append(self.sequence)
        if self.history_dir.exists():
            for entry in self.history_dir.glob("manifest-*.json"):
                stem = entry.stem[len("manifest-") :]
                if stem.isdigit():
                    versions.append(int(stem))
        return sorted(set(versions))

    def all_documents(self) -> List[Dict[str, object]]:
        return [
            {"id": doc.id, "fields": dict(doc.fields)}
            for doc in sorted(self._documents_view()[0].values(), key=lambda item: item.id)
        ]

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

    def _archive_current_manifest(self) -> None:
        if not self.manifest_path.exists() or self.sequence == 0:
            return
        target = self.history_dir / f"manifest-{self.sequence:012d}.json"
        if target.exists():
            return
        self.history_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(str(target), self._read_manifest_file(self.manifest_path))

    def _read_manifest_file(self, path: Path) -> Dict[str, object]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_rules(self, rules: SynonymRules) -> None:
        payload = {"version": rules.version, "synonyms": rules.to_mapping()}
        atomic_write_json(str(self.rules_path), payload)

    def _load_rules(self, version: int) -> SynonymRules:
        if version == 0:
            return SynonymRules.empty(0)
        cached = self._saved_rules_versions.get(version)
        if cached is not None:
            return cached
        if self.rules_path.exists():
            with self.rules_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if int(payload.get("version", 0)) == version:
                rules = SynonymRules.from_mapping(payload.get("synonyms", {}), version=version)
                self._saved_rules_versions[version] = rules
                return rules
        raise RulesVersionError(f"Rules version {version} is missing from {self.rules_path}")

    def _snapshot_from_manifest(self, manifest: Dict[str, object], base_path: Path) -> Snapshot:
        sequence = int(manifest["sequence"])
        segments: List[Segment] = []
        segment_sequences: Dict[str, int] = {}
        for entry in manifest.get("segments", []):
            segment_path = Path(entry["path"])
            if not segment_path.is_absolute():
                segment_path = self.directory / segment_path
            segment = Segment.load(entry["id"], str(segment_path))
            segments.append(segment)
            segment_sequences[entry["id"]] = int(entry["sequence"])
        deletes = {doc_id: int(value) for doc_id, value in manifest.get("deletes", {}).items()}
        visible, _ = self._view(segments, segment_sequences, deletes)
        rules_version = int(manifest.get("rules_version", 0))
        return Snapshot(sequence, visible, self._load_rules(rules_version), self.ranker)

    @staticmethod
    def _view(
        segments: Sequence[Segment],
        segment_sequences: Dict[str, int],
        deletes: Dict[str, int],
    ) -> tuple[Dict[str, StoredDocument], Dict[str, int]]:
        visible: Dict[str, StoredDocument] = {}
        newest: Dict[str, int] = {}
        for segment in sorted(segments, key=lambda item: segment_sequences[item.segment_id]):
            sequence = segment_sequences[segment.segment_id]
            for doc_id, doc in segment.documents.items():
                if sequence > deletes.get(doc_id, -1):
                    visible[doc_id] = doc
                    newest[doc_id] = sequence
                elif newest.get(doc_id, -1) <= deletes.get(doc_id, -1):
                    visible.pop(doc_id, None)
        return visible, newest

    def _load(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.segments_dir.mkdir(exist_ok=True)
        self.history_dir.mkdir(exist_ok=True)
        if not self.manifest_path.exists():
            self.segments = []
            self.segment_sequences = {}
            self.deletes = {}
            self.sequence = 0
            self.rules_version = 0
            self.rules = SynonymRules.empty(0)
            return

        manifest = self._read_manifest_file(self.manifest_path)
        self.sequence = int(manifest["sequence"])
        self.segments = []
        self.segment_sequences = {}
        for entry in manifest.get("segments", []):
            path = self.directory / entry["path"]
            segment = Segment.load(entry["id"], str(path))
            self.segments.append(segment)
            self.segment_sequences[entry["id"]] = int(entry["sequence"])
        self.deletes = {doc_id: int(sequence) for doc_id, sequence in manifest.get("deletes", {}).items()}
        self.rules_version = int(manifest.get("rules_version", 0))
        self.rules = self._load_rules(self.rules_version)
