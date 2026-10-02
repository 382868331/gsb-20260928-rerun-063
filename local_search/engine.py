import json
import os
import uuid
from base64 import urlsafe_b64decode, urlsafe_b64encode
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .query import Query
from .ranking import BM25Ranker, Ranker, ranker_descriptor, ranker_from_descriptor, ranker_matches_descriptor
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
RUNS_DIRNAME = "runs"


def _encode_cursor(run_id: str, offset: int) -> str:
    payload = json.dumps({"run_id": run_id, "offset": offset}, separators=(",", ":")).encode("utf-8")
    return urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_cursor(cursor: str) -> Tuple[str, int]:
    padded = cursor + "=" * (-len(cursor) % 4)
    payload = json.loads(urlsafe_b64decode(padded.encode("ascii")))
    return str(payload["run_id"]), int(payload["offset"])


class SearchEngine:
    def __init__(self, directory: str | os.PathLike[str] = ".search_index", ranker: Optional[Ranker] = None) -> None:
        self.directory = Path(directory)
        self.segments_dir = self.directory / "segments"
        self.snapshots_dir = self.directory / SNAPSHOTS_DIRNAME
        self.runs_dir = self.directory / RUNS_DIRNAME
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

    def start_search(
        self,
        query_text: str | Query,
        *,
        page_size: int = 10,
        budget: Optional[int] = None,
        snapshot: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, object]:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if budget is not None and budget <= 0:
            raise ValueError("budget must be positive when provided")

        run_id = run_id or uuid.uuid4().hex
        snapshot_name = snapshot or f"run-{run_id}"
        if snapshot is None:
            self.publish_snapshot(snapshot_name)
        view = self.snapshot_view(snapshot_name)
        from .query import parse_query

        query = query_text if isinstance(query_text, Query) else parse_query(query_text)
        bound_query = query.with_rules(view.rules)
        candidate_queue = sorted(view.documents)
        run = {
            "id": run_id,
            "snapshot": snapshot_name,
            "query": bound_query.to_payload(),
            "ranking": ranker_descriptor(self.ranker),
            "page_size": page_size,
            "budget": budget,
            "sequence": view.entries[-1].sequence if view.entries else 0,
            "candidate_queue": candidate_queue,
            "pending_results": [],
            "confirmed": [],
            "exhausted": False,
        }
        self._save_run(run)
        return self.fetch_page(run_id, budget=budget)

    def fetch_page(
        self,
        run_id: str,
        cursor: Optional[str] = None,
        *,
        budget: Optional[int] = None,
    ) -> Dict[str, object]:
        run = self._load_run(run_id)
        expected_offset = len(run["confirmed"])
        if cursor is not None:
            cursor_run_id, cursor_offset = _decode_cursor(cursor)
            if cursor_run_id != run_id:
                raise ValueError("Cursor does not belong to this search run")
            if cursor_offset != expected_offset:
                raise ValueError("Cursor is not the next page of this search run")

        page_size = int(run["page_size"])
        effective_budget = run.get("budget") if budget is None else budget
        if effective_budget is not None and int(effective_budget) <= 0:
            raise ValueError("budget must be positive when provided")
        if not run["exhausted"] and run["candidate_queue"]:
            self._materialize_more(run, None if effective_budget is None else int(effective_budget))

        view = self.snapshot_view(str(run["snapshot"]))
        pending_results = list(run["pending_results"])
        pending_results.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
        offset = len(run["confirmed"])
        page_results = pending_results[:page_size]
        for index, result in enumerate(page_results, start=1):
            result["rank"] = offset + index
        page_ids = {str(item["id"]) for item in page_results}
        run["pending_results"] = [item for item in run["pending_results"] if str(item["id"]) not in page_ids]
        run["confirmed"].extend(page_results)
        run["exhausted"] = not run["candidate_queue"] and not run["pending_results"]
        self._save_run(run)

        next_cursor = None if run["exhausted"] else _encode_cursor(run_id, offset + len(page_results))
        return {
            "run_id": run_id,
            "cursor": None if offset == 0 else _encode_cursor(run_id, offset),
            "next_cursor": next_cursor,
            "page": {"number": offset // page_size + 1, "offset": offset, "size": len(page_results)},
            "perspective": self._perspective_payload(run, view),
            "ranking": run["ranking"],
            "results": page_results,
            "exhausted": bool(run["exhausted"]),
            "complete": bool(run["exhausted"]),
            "budget": effective_budget,
            "unexamined_candidates": len(run["candidate_queue"]),
        }

    def search_page(
        self,
        query_text: str | Query,
        *,
        page_size: int = 10,
        budget: Optional[int] = None,
        snapshot: Optional[str] = None,
    ) -> Dict[str, object]:
        return self.start_search(query_text, page_size=page_size, budget=budget, snapshot=snapshot)

    def explain_rank_changes(self, previous_run_id: str, current_run_id: str) -> Dict[str, object]:
        old_run = self._load_run(previous_run_id)
        new_run = self._load_run(current_run_id)
        old_view = self.snapshot_view(str(old_run["snapshot"]))
        new_view = self.snapshot_view(str(new_run["snapshot"]))
        old_query = Query.from_payload(old_run["query"]).with_rules(old_view.rules)
        new_query = Query.from_payload(new_run["query"]).with_rules(new_view.rules)
        old_results = {str(item["id"]): item for item in self._build_results(old_view, old_query, self._ranker_for_run(old_run))}
        new_results = {str(item["id"]): item for item in self._build_results(new_view, new_query, self._ranker_for_run(new_run))}
        old_order = sorted(old_results, key=lambda doc_id: (-float(old_results[doc_id]["score"]), doc_id))
        new_order = sorted(new_results, key=lambda doc_id: (-float(new_results[doc_id]["score"]), doc_id))
        old_positions = {doc_id: index + 1 for index, doc_id in enumerate(old_order)}
        new_positions = {doc_id: index + 1 for index, doc_id in enumerate(new_order)}
        changes = []
        for doc_id in sorted(set(old_results) | set(new_results)):
            old = old_results.get(doc_id)
            new = new_results.get(doc_id)
            if old is None or new is None:
                reason = "match_condition_added" if old is None else "match_condition_removed"
            elif self._condition_signature(old) != self._condition_signature(new):
                reason = "match_condition_changed"
            elif float(old["score"]) != float(new["score"]):
                reason = "score_changed"
            elif old_positions[doc_id] != new_positions[doc_id]:
                reason = "relative_position_changed"
            else:
                reason = "unchanged"
            changes.append(
                {
                    "id": doc_id,
                    "reason": reason,
                    "old_position": old_positions.get(doc_id),
                    "new_position": new_positions.get(doc_id),
                    "old_score": None if old is None else old["score"],
                    "new_score": None if new is None else new["score"],
                    "score_delta": None if old is None or new is None else float(new["score"]) - float(old["score"]),
                    "old_evidence": None if old is None else self._evidence(old, old_run),
                    "new_evidence": None if new is None else self._evidence(new, new_run),
                }
            )
        return {
            "previous_run_id": previous_run_id,
            "current_run_id": current_run_id,
            "changes": changes,
        }

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

    def _run_query(self, view: SearchView, query: Query, limit: Optional[int]) -> List[Dict[str, object]]:
        query = query.with_rules(view.rules)
        results = self._build_results(view, query, self.ranker)
        results.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
        return results[:limit] if limit is not None else results

    def _build_results(
        self,
        view: SearchView,
        query: Query,
        ranker: Ranker,
        candidate_ids: Optional[set[str]] = None,
    ) -> List[Dict[str, object]]:
        query = query.with_rules(view.rules)
        matches = view.run(query, candidate_ids)
        scored = ranker.score(matches, query)
        results: List[Dict[str, object]] = []
        for doc_id, document_match in matches.items():
            document = document_match.document
            detail = scored.get(doc_id, {})
            score = float(detail.get("score", 0.0))
            results.append(
                {
                    "id": document.id,
                    "score": score,
                    "document_version": view.newest.get(doc_id),
                    "fields": dict(document.fields),
                    "explanation": document_match.explanation(
                        score,
                        detail.get("term_contributions", {}),
                        query.rules.version if query.rules is not None else None,
                        SearchView.excluded_rawhs(query),
                    ),
                }
            )
        return results

    def _materialize_more(self, run: Dict[str, Any], budget: Optional[int]) -> None:
        view = self.snapshot_view(str(run["snapshot"]))
        query = Query.from_payload(run["query"]).with_rules(view.rules)
        count = len(run["candidate_queue"]) if budget is None else min(budget, len(run["candidate_queue"]))
        chosen = run["candidate_queue"][:count]
        del run["candidate_queue"][:count]
        if not chosen:
            return
        ranker = self._ranker_for_run(run)
        candidate_set = {str(item["id"]) for item in run["pending_results"]} | set(chosen)
        all_results = self._build_results(view, query, ranker, candidate_set)
        by_id = {str(item["id"]): item for item in all_results}
        run["pending_results"] = [by_id[doc_id] for doc_id in sorted(by_id)]

    def _ranker_for_run(self, run: Dict[str, Any]) -> Ranker:
        descriptor = run.get("ranking", {})
        if descriptor.get("persistent") is False and ranker_matches_descriptor(self.ranker, descriptor):
            return self.ranker
        return ranker_from_descriptor(descriptor)

    def _perspective_payload(self, run: Dict[str, Any], view: SearchView) -> Dict[str, object]:
        return {
            "snapshot": run["snapshot"],
            "sequence": run["sequence"],
            "synonym_version": view.rules.version,
            "synonym_rules": view.rules.to_payload(),
            "document_versions": dict(view.newest),
        }

    @staticmethod
    def _condition_signature(result: Dict[str, object]) -> str:
        explanation = result.get("explanation", {})
        clauses = explanation.get("clauses", []) if isinstance(explanation, dict) else []
        return json.dumps(clauses, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @staticmethod
    def _evidence(result: Dict[str, object], run: Dict[str, Any]) -> Dict[str, object]:
        explanation = result.get("explanation", {})
        return {
            "snapshot": run["snapshot"],
            "document_version": result.get("document_version"),
            "synonym_version": explanation.get("synonym_version") if isinstance(explanation, dict) else None,
            "explanation": explanation,
        }

    def _run_path(self, run_id: str) -> Path:
        if not run_id or any(char in run_id for char in (os.sep, "/", "\\")):
            raise ValueError("Invalid run id")
        return self.runs_dir / f"{run_id}.json"

    def _save_run(self, run: Dict[str, Any]) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(str(self._run_path(str(run["id"]))), run)

    def _load_run(self, run_id: str) -> Dict[str, Any]:
        with self._run_path(run_id).open("r", encoding="utf-8") as handle:
            return json.load(handle)

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
        self.runs_dir.mkdir(exist_ok=True)
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
