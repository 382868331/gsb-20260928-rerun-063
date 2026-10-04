import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from .query import Query
from .ranking import BM25Ranker, Ranker
from .segment import (
    Segment,
    StoredDocument,
    atomic_write_json,
    build_segment,
    write_segment,
)
from .synonyms import EMPTY_RULES, SynonymRules
from .view import SearchView

MANIFEST_NAME = "manifest.json"
RULES_NAME = "synonyms.json"
SNAPSHOTS_DIRNAME = "snapshots"
RESEARCH_DIRNAME = "research"


class SearchEngine:
    def __init__(self, directory: str | os.PathLike[str] = ".search_index", ranker: Optional[Ranker] = None) -> None:
        self.directory = Path(directory)
        self.segments_dir = self.directory / "segments"
        self.snapshots_dir = self.directory / SNAPSHOTS_DIRNAME
        self.research_dir = self.directory / RESEARCH_DIRNAME
        self.manifest_path = self.directory / MANIFEST_NAME
        self.rules_path = self.directory / RULES_NAME
        self.ranker = ranker or BM25Ranker()
        self._staged_updates: Dict[str, Dict[str, object]] = {}
        self._staged_deletes: set[str] = set()
        self.segments: List[Segment] = []
        self.segment_sequences: Dict[str, int] = {}
        self.deletes: Dict[str, int] = {}
        self.sequence = 0
        self.rules: SynonymRules = EMPTY_RULES
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

    def set_synonym_rules(self, rules: SynonymRules) -> None:
        """Publish a new complete synonym rule version.

        New queries immediately use the new version.  Query objects that
        already carry an older complete version keep using it, so in-flight
        queries are unaffected.
        """
        if not isinstance(rules, SynonymRules):
            raise ValueError("rules must be a SynonymRules instance")
        if rules.version <= self.rules.version:
            raise ValueError(
                f"Synonym rule version must increase (current version is {self.rules.version})"
            )
        atomic_write_json(str(self.rules_path), rules.to_payload())
        self.rules = rules

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
            written_path = write_segment(str(self.segments_dir), segment_id, payload)
            segment_path = os.path.relpath(written_path, self.directory)

        manifest = self._manifest_payload(next_sequence, segment_id, segment_path)
        atomic_write_json(str(self.manifest_path), manifest)
        self._load_manifest()
        self._staged_updates.clear()
        self._staged_deletes.clear()
        return True

    def publish_snapshot(self, name: str) -> Dict[str, object]:
        """Publish an immutable, queryable snapshot of the current state.

        The snapshot records the complete synonym rule version in force at
        publication time.  Later document changes, rule updates or index
        compaction never alter a published snapshot.
        """
        if not name or any(char in name for char in (os.sep, "/", "\\")):
            raise ValueError("Snapshot name must be a non-empty path-safe string")
        snapshot = {
            "name": name,
            "sequence": self.sequence,
            "rules": self.rules.to_payload(),
            "segments": [
                {
                    "id": segment.segment_id,
                    "sequence": self.segment_sequences[segment.segment_id],
                    "path": os.path.relpath(segment.path, self.directory),
                }
                for segment in sorted(self.segments, key=lambda item: self.segment_sequences[item.segment_id])
            ],
            "deletes": dict(self.deletes),
        }
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(str(self.snapshots_dir / f"{name}.json"), snapshot)
        return snapshot

    def snapshot_view(self, name: str) -> SearchView:
        with (self.snapshots_dir / f"{name}.json").open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return self._view_from_payload(payload)

    def search_snapshot(self, name: str, query_text: str, limit: Optional[int] = None) -> List[Dict[str, object]]:
        from .query import parse_query

        view = self.snapshot_view(name)
        query = query_text if isinstance(query_text, Query) else parse_query(query_text)
        return self._run_query(view, query, limit)

    def compact(self) -> bool:
        """Rewrite all visible live documents into one fresh segment.

        The replacement segment and manifest are published with atomic
        renames.  An interruption before that publication leaves the
        previously published manifest, segments and snapshots intact.
        """
        view = self.live_view()
        visible = view.documents
        if not visible and not self.segments:
            return False

        next_sequence = self.sequence + 1
        segment_id = f"seg-{next_sequence:012d}"
        documents = [visible[doc_id] for doc_id in sorted(visible)]
        segment_path: Optional[str] = None
        if documents:
            payload = build_segment(segment_id, documents)
            payload["sequence"] = next_sequence
            written_path = write_segment(str(self.segments_dir), segment_id, payload)
            segment_path = os.path.relpath(written_path, self.directory)

        manifest = self._manifest_payload(next_sequence, segment_id, segment_path, include_old=False)
        atomic_write_json(str(self.manifest_path), manifest)
        self._load_manifest()
        return True

    def get(self, doc_id: str) -> Optional[Dict[str, object]]:
        doc = self.live_view().documents.get(doc_id)
        return None if doc is None else dict(doc.fields)

    def live_view(self) -> SearchView:
        return SearchView.live(self.segments, self.segment_sequences, self.deletes, self.rules)

    def search(self, query_text: str, limit: Optional[int] = None) -> List[Dict[str, object]]:
        from .query import parse_query

        view = self.live_view()
        query = query_text if isinstance(query_text, Query) else parse_query(query_text)
        return self._run_query(view, query, limit)

    def all_documents(self) -> List[Dict[str, object]]:
        return [
            {"id": doc.id, "fields": dict(doc.fields)}
            for doc in sorted(self.live_view().documents.values(), key=lambda item: item.id)
        ]

    # ----------------------------------------------------------- research sets

    def create_research_set(
        self,
        name: str,
        queries: "Dict[str, str] | List[tuple[str, str]]",
        *,
        mode: str = "any",
        budget: Optional[int] = None,
        snapshot: Optional[str] = None,
    ):
        """Confirm and save a multi-query research set against current state.

        ``queries`` maps a stable label to its query text (or is a sequence
        of ``(label, query)`` pairs).  ``mode`` is ``"any"`` (a document
        satisfying any query joins) or ``"all"`` (it must hit every query).
        The query texts, the complete synonym rule version and every
        document body are pinned into the saved set.
        """
        from .research import QuerySpec, ResearchSet

        if not name or any(char in name for char in (os.sep, "/", "\\")):
            raise ValueError("Research set name must be a non-empty path-safe string")
        if isinstance(queries, dict):
            pairs = list(queries.items())
        else:
            pairs = [(label, raw) for label, raw in queries]
        specs = [QuerySpec(str(label), str(raw)) for label, raw in pairs]
        view = self.snapshot_view(snapshot) if snapshot is not None else self.live_view()
        path = self.research_dir / f"{name}.json"
        research = ResearchSet(
            name,
            specs,
            mode,
            view.documents,
            view.rules,
            self.ranker,
            budget=budget,
            path=path,
        )
        research.save()
        return research

    def open_research_set(self, name: str):
        """Reload a saved research set; its pinned order is unchanged."""
        from .research import ResearchSet

        with (self.research_dir / f"{name}.json").open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return ResearchSet.from_payload(payload, self.research_dir / f"{name}.json", self.ranker)

    def revise_research_set(
        self,
        name: str,
        queries: "Dict[str, str] | List[tuple[str, str]] | None" = None,
        *,
        mode: Optional[str] = None,
        budget: Optional[int] = None,
        new_name: Optional[str] = None,
    ) -> dict:
        """Build a revised set from revised queries and compare with the old one.

        Returns the comparison report (entered/left/moved documents with
        reasons, and which manual decisions were inherited or dropped).  The
        new set is saved under ``new_name`` (defaulting to ``name``).
        """
        from .research import QuerySpec, ResearchSet, compare_research_sets

        old = self.open_research_set(name)
        pairs = (
            list(queries.items()) if isinstance(queries, dict)
            else [(label, raw) for label, raw in queries] if queries is not None
            else [(spec.label, spec.raw) for spec in old.specs]
        )
        specs = [QuerySpec(str(label), str(raw)) for label, raw in pairs]
        view = self.live_view()
        target_name = new_name or name
        path = self.research_dir / f"{target_name}.json"
        new = ResearchSet(
            target_name,
            specs,
            mode or old.mode,
            view.documents,
            self.rules,
            self.ranker,
            budget=old.budget if budget is None else budget,
            path=path,
        )
        report = compare_research_sets(old, new)
        new.save()
        return report

    def derive_research_set(
        self,
        name: str,
        new_name: str,
        *,
        budget: Optional[int] = None,
    ) -> dict:
        """Derive a new research set from a frozen one against the live index.

        The queries stay the same; documents that still exist keep their
        reviewers' evidence-based decisions and read positions, while
        changed or deleted records are reported explicitly in the returned
        comparison (dropped_decisions) instead of being silently lost.
        """
        return self.revise_research_set(name, None, budget=budget, new_name=new_name)

    def _run_query(self, view: SearchView, query: Query, limit: Optional[int]) -> List[Dict[str, object]]:
        query = query.with_rules(view.rules)
        matches = view.run(query)
        scored = self.ranker.score(matches, query)
        ordered_ids = sorted(
            matches,
            key=lambda doc_id: (-float(scored.get(doc_id, {}).get("score", 0.0)), doc_id),
        )
        if limit is not None:
            ordered_ids = ordered_ids[:limit]
        results: List[Dict[str, object]] = []
        for doc_id in ordered_ids:
            document = matches[doc_id].document
            detail = scored.get(doc_id, {})
            score = float(detail.get("score", 0.0))
            results.append(
                {
                    "id": document.id,
                    "score": score,
                    "fields": dict(document.fields),
                    "explanation": matches[doc_id].explanation(
                        score,
                        detail.get("term_contributions", {}),
                        query.rules.version if query.rules is not None else None,
                        SearchView.excluded_rawhs(query),
                    ),
                }
            )
        return results

    def _manifest_payload(
        self,
        next_sequence: int,
        new_segment_id: Optional[str],
        new_segment_path: Optional[str],
        include_old: bool = True,
    ) -> Dict[str, object]:
        segments: List[Dict[str, object]] = []
        if include_old:
            segments = [
                {
                    "id": segment.segment_id,
                    "sequence": self.segment_sequences[segment.segment_id],
                    "path": os.path.relpath(segment.path, self.directory),
                }
                for segment in self.segments
            ]
        if new_segment_id is not None and new_segment_path is not None:
            segments.append({"id": new_segment_id, "sequence": next_sequence, "path": new_segment_path})
        deletes = dict(self.deletes)
        for doc_id in self._staged_deletes:
            deletes[doc_id] = next_sequence
        return {"sequence": next_sequence, "segments": segments, "deletes": deletes}

    def _view_from_payload(self, payload: Dict[str, object]) -> SearchView:
        segments: List[Segment] = []
        sequences: Dict[str, int] = {}
        for entry in payload.get("segments", []):
            path = self.directory / entry["path"]
            segment = Segment.load(entry["id"], str(path))
            segments.append(segment)
            sequences[entry["id"]] = int(entry["sequence"])
        deletes = {doc_id: int(sequence) for doc_id, sequence in payload.get("deletes", {}).items()}
        rules = SynonymRules.from_payload(payload["rules"])
        return SearchView.live(segments, sequences, deletes, rules)

    def _load(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.segments_dir.mkdir(exist_ok=True)
        self.snapshots_dir.mkdir(exist_ok=True)
        self.research_dir.mkdir(exist_ok=True)
        self.rules = self._load_rules()
        self._load_manifest()

    def _load_rules(self) -> SynonymRules:
        if not self.rules_path.exists():
            return EMPTY_RULES
        with self.rules_path.open("r", encoding="utf-8") as handle:
            return SynonymRules.from_payload(json.load(handle))

    def _load_manifest(self) -> None:
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
