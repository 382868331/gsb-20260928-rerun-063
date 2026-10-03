"""Multi-query research sets with per-query evidence and stable review order.

A research set gathers the candidate documents produced by several queries
behind one reviewing perspective.  Each query keeps its own score and
positional evidence; scores from different queries are never summed into a
single relevance number.  The set pins the exact query texts, synonym rule
version and document bodies used at confirmation time, so confirmed pages
keep a stable order even when the live corpus changes later.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .pager import RankingOutcome, run_ranked
from .query import parse_query
from .ranking import BM25Ranker, ConstantScoreRanker, Ranker
from .segment import StoredDocument, atomic_write_json
from .synonyms import SynonymRules
from .view import DocumentMatch, SearchView

MODE_ANY = "any"
MODE_ALL = "all"

INCLUDED = "included"
EXCLUDED = "excluded"
PENDING_REVIEW = "pending"


@dataclass(frozen=True)
class QuerySpec:
    label: str
    raw: str


@dataclass
class Decision:
    action: str  # INCLUDED or EXCLUDED
    reason: str
    basis: Dict[str, dict] = field(default_factory=dict)
    rules_version: Optional[int] = None
    inherited: bool = False


@dataclass
class CandidateEvidence:
    """What one query contributed for one document."""

    score: Optional[float]
    rank: Optional[int]  # 1-based rank inside that query's own ranking
    signature: dict


def evidence_signature(match: DocumentMatch) -> dict:
    """Content-sensitive basis used to decide whether a decision still holds.

    The signature captures the concrete words, fields and positions that the
    query matched, not just the document name.
    """
    return {
        "clauses": [
            {
                "raw": hit.clause.raw,
                "field": hit.clause.field,
                "matched_field": hit.field_name,
                "phrase": hit.clause.phrase,
                "expand": hit.clause.expand,
                "slop": hit.clause.slop,
                "terms": [
                    {"term": term.term, "matched_as": term.matched_as, "position": term.position}
                    for term in hit.terms
                ],
            }
            for hit in match.clause_hits
        ]
    }


def _ranker_name(ranker: Ranker) -> str:
    if isinstance(ranker, ConstantScoreRanker):
        return "constant"
    if isinstance(ranker, BM25Ranker):
        return "bm25"
    return ranker.__class__.__name__


def _ranker_by_name(name: str, fallback: Ranker) -> Ranker:
    if name == "constant":
        return ConstantScoreRanker()
    if name == "bm25":
        return BM25Ranker()
    return fallback


class ResearchSet:
    def __init__(
        self,
        name: str,
        specs: Sequence[QuerySpec],
        mode: str,
        documents: Dict[str, StoredDocument],
        rules: SynonymRules,
        ranker: Ranker,
        budget: Optional[int] = None,
        path: Optional[Path] = None,
    ) -> None:
        if mode not in (MODE_ANY, MODE_ALL):
            raise ValueError("mode must be 'any' or 'all'")
        if not specs:
            raise ValueError("A research set needs at least one query")
        labels = [spec.label for spec in specs]
        if len(set(labels)) != len(labels):
            raise ValueError("Query labels must be unique")
        self.name = name
        self.specs = list(specs)
        self.mode = mode
        self.documents = dict(documents)
        self.rules = rules
        self.ranker = ranker
        self.ranker_name = _ranker_name(ranker)
        self.budget = budget
        self.path = path
        self.decisions: Dict[str, Decision] = {}
        self.read_ids: List[str] = []
        self.evidence: Dict[str, Dict[str, CandidateEvidence]] = {}
        self.outcomes: Dict[str, RankingOutcome] = {}
        self.exhausted = False
        self._recompute()

    def _bound_queries(self) -> List[Tuple[QuerySpec, object]]:
        return [(spec, parse_query(spec.raw).with_rules(self.rules)) for spec in self.specs]

    def _recompute(self) -> None:
        """Re-run every pinned query; matching is always fully evaluated.

        The scan budget only limits which matches receive a per-query score,
        mirroring single-query budgeted scanning.
        """
        view = SearchView.pinned(self.documents, self.rules)
        self.outcomes = {}
        self.evidence = {}
        all_exhausted = True
        for spec, query in self._bound_queries():
            outcome = run_ranked(view, query, self.ranker, self.budget)
            self.outcomes[spec.label] = outcome
            all_exhausted = all_exhausted and outcome.exhausted
            for position, entry in enumerate(outcome.entries):
                self._record_evidence(spec.label, entry.doc_id, position + 1, entry.score, outcome)
            # Matches the budget could not score stay visible as unscored
            # pending candidates, so nobody is implicitly judged as absent.
            for doc_id in outcome.skipped_unscored:
                self._record_evidence(spec.label, doc_id, None, None, outcome)
        self.exhausted = all_exhausted

    def _record_evidence(self, label: str, doc_id: str, rank, score, outcome) -> None:
        signature = evidence_signature(outcome.matches[doc_id])
        self.evidence.setdefault(doc_id, {})[label] = CandidateEvidence(
            score=score,
            rank=rank,
            signature=signature,
        )

    def _membership(self, doc_id: str) -> Tuple[bool, bool]:
        """Return (is_candidate, fully_evaluated) for one document."""
        hit_labels = self.evidence.get(doc_id, {})
        if not hit_labels:
            return False, True
        fully_scored = all(item.rank is not None for item in hit_labels.values())
        if self.mode == MODE_ANY:
            return True, fully_scored
        required = {spec.label for spec in self.specs}
        hits_required = required.issubset(hit_labels)
        return hits_required, hits_required and fully_scored

    def _combined_key(self, doc_id: str) -> Tuple:
        """Deterministic cross-query order that never sums query scores.

        Best per-query rank first, then how many queries hit, then the full
        rank vector, then the stable document identity.
        """
        per_query = self.evidence.get(doc_id, {})
        ranks = tuple(
            per_query[spec.label].rank if spec.label in per_query else None
            for spec in self.specs
        )
        scored_ranks = [rank for rank in ranks if rank is not None]
        best = min(scored_ranks) if scored_ranks else 10**12
        big = 10**12
        vector = tuple(rank if rank is not None else big for rank in ranks)
        return (best, -len(scored_ranks), vector, doc_id)

    def _ordered_candidate_ids(self) -> Tuple[List[str], List[str]]:
        """Return (evaluated_order, pending_unscored) document ids."""
        evaluated: List[str] = []
        pending: List[str] = []
        candidate_pool = set(self.evidence) | {
            doc_id
            for doc_id, decision in self.decisions.items()
            if decision.action == INCLUDED
        }
        for doc_id in sorted(candidate_pool):
            is_candidate, fully = self._membership(doc_id)
            decision = self.decisions.get(doc_id)
            if not is_candidate and decision is None:
                continue
            if decision is not None and decision.action == EXCLUDED:
                continue
            if not is_candidate and decision.action != INCLUDED:
                continue
            (evaluated if fully else pending).append(doc_id)
        evaluated.sort(key=self._combined_key)
        pending.sort(key=self._combined_key)
        return evaluated, pending

    def reading_order(self) -> List[str]:
        """Full reading list: frozen already-read prefix, then current order.

        Confirmed documents keep the order in which they were first read;
        every not-yet-read document follows in the current deterministic
        cross-query order.  Continuation therefore never drops or duplicates
        a document, and a fresh exhaustive read of the same perspective is
        exactly this order.
        """
        evaluated, pending = self._ordered_candidate_ids()
        current = evaluated + pending
        read = [doc_id for doc_id in self.read_ids if doc_id in current]
        seen = set(read)
        return read + [doc_id for doc_id in current if doc_id not in seen]

    def page(self, offset: int = 0, page_size: Optional[int] = None) -> dict:
        ordered = self.reading_order()
        end = len(ordered) if page_size is None else min(offset + page_size, len(ordered))
        ids = ordered[offset:end]
        next_offset = end if end < len(ordered) else None
        _, pending = self._ordered_candidate_ids()
        return {
            "results": [self.document_payload(doc_id) for doc_id in ids],
            "offset": offset,
            "next_offset": next_offset,
            "exhausted": self.exhausted,
            "complete": self.exhausted and next_offset is None,
            "total_candidates": len(ordered),
            "pending_unscored": len(pending),
        }

    def read_page(self, page_size: Optional[int] = None) -> dict:
        """Read the next unreviewed slice and mark it reviewed."""
        ordered = self.reading_order()
        offset = len([doc_id for doc_id in self.read_ids if doc_id in ordered])
        payload = self.page(offset, page_size)
        for result in payload["results"]:
            if result["id"] not in self.read_ids:
                self.read_ids.append(result["id"])
        payload["reviewed_count"] = len(self.read_ids)
        return payload

    def unreviewed(self) -> List[str]:
        ordered = self.reading_order()
        return [doc_id for doc_id in ordered if doc_id not in self.read_ids]

    def review_status(self) -> dict:
        evaluated, pending = self._ordered_candidate_ids()
        current = set(evaluated + pending)
        return {
            "exhausted": self.exhausted,
            "candidate_count": len(current),
            "reviewed": [doc_id for doc_id in self.read_ids if doc_id in current],
            "unreviewed": self.unreviewed(),
            "excluded": sorted(
                doc_id for doc_id, decision in self.decisions.items() if decision.action == EXCLUDED
            ),
            "manual_included": sorted(
                doc_id for doc_id, decision in self.decisions.items() if decision.action == INCLUDED
            ),
            "pending_unscored": sorted(pending),
        }

    def decide(self, doc_id: str, action: str, reason: str) -> Decision:
        if action not in (INCLUDED, EXCLUDED):
            raise ValueError("decision action must be 'included' or 'excluded'")
        if not reason or not reason.strip():
            raise ValueError("A manual decision needs an explicit reason")
        if doc_id not in self.documents:
            raise KeyError(doc_id)
        is_candidate, _ = self._membership(doc_id)
        basis = {
            label: evidence.signature
            for label, evidence in self.evidence.get(doc_id, {}).items()
        }
        if not is_candidate and action != INCLUDED:
            raise ValueError("Only an inclusion exception can be recorded for a non-matching document")
        decision = Decision(action, reason.strip(), basis, self.rules.version)
        self.decisions[doc_id] = decision
        return decision

    def document_payload(self, doc_id: str) -> dict:
        stored = self.documents[doc_id]
        per_query: Dict[str, dict] = {}
        for spec in self.specs:
            outcome = self.outcomes[spec.label]
            evidence = self.evidence.get(doc_id, {}).get(spec.label)
            if evidence is None:
                per_query[spec.label] = {"matched": False, "score": None, "rank": None}
                continue
            per_query[spec.label] = {
                "matched": True,
                "score": evidence.score,
                "rank": evidence.rank,
                "explanation": outcome.matches[doc_id].explanation(
                    evidence.score if evidence.score is not None else 0.0,
                    outcome.scores.get(doc_id, {}).get("term_contributions", {}),
                    self.rules.version,
                    SearchView.excluded_rawhs(parse_query(spec.raw).with_rules(self.rules)),
                ),
            }
        decision = self.decisions.get(doc_id)
        return {
            "id": doc_id,
            "fields": dict(stored.fields),
            "matched_queries": sorted(self.evidence.get(doc_id, {})),
            "per_query": per_query,
            "review": decision.action if decision is not None else PENDING_REVIEW,
            "review_reason": decision.reason if decision is not None else None,
            "reviewed": doc_id in self.read_ids,
        }

    def continue_scan(self, budget: Optional[int] = None) -> None:
        """Resume scoring with a larger (or unlimited) budget on the same view."""
        if budget is not None and self.budget is not None and budget < self.budget:
            raise ValueError("Continuation can only raise the scan budget")
        self.budget = budget
        self._recompute()

    def to_payload(self) -> dict:
        return {
            "name": self.name,
            "mode": self.mode,
            "budget": self.budget,
            "ranker": self.ranker_name,
            "rules": self.rules.to_payload(),
            "queries": [{"label": spec.label, "raw": spec.raw} for spec in self.specs],
            "documents": {doc_id: dict(doc.fields) for doc_id, doc in self.documents.items()},
            "read_ids": list(self.read_ids),
            "decisions": {
                doc_id: {
                    "action": decision.action,
                    "reason": decision.reason,
                    "basis": decision.basis,
                    "rules_version": decision.rules_version,
                    "inherited": decision.inherited,
                }
                for doc_id, decision in self.decisions.items()
            },
        }

    def save(self) -> None:
        if self.path is None:
            raise ValueError("This research set is not attached to a storage path")
        atomic_write_json(str(self.path), self.to_payload())

    @classmethod
    def from_payload(cls, payload: dict, path: Optional[Path], ranker_fallback: Ranker) -> "ResearchSet":
        rules = SynonymRules.from_payload(payload["rules"])
        documents = {
            doc_id: StoredDocument(doc_id, fields)
            for doc_id, fields in payload["documents"].items()
        }
        specs = [QuerySpec(entry["label"], entry["raw"]) for entry in payload["queries"]]
        ranker = _ranker_by_name(payload.get("ranker", "bm25"), ranker_fallback)
        research = cls(
            payload["name"],
            specs,
            payload["mode"],
            documents,
            rules,
            ranker,
            budget=payload.get("budget"),
            path=path,
        )
        research.read_ids = list(payload.get("read_ids", []))
        for doc_id, entry in payload.get("decisions", {}).items():
            research.decisions[doc_id] = Decision(
                action=entry["action"],
                reason=entry["reason"],
                basis=entry.get("basis", {}),
                rules_version=entry.get("rules_version"),
                inherited=bool(entry.get("inherited", False)),
            )
        return research
# --------------------------------------------------------------------- revision

DOCUMENT_ENTERED = "entered"
DOCUMENT_LEFT = "left"
RANK_MOVED = "rank_moved"
RANK_STABLE = "stable"

REASON_DOCUMENT_CHANGED = "document_changed"
REASON_QUERY_CHANGED = "query_meaning_changed"
REASON_RELATIVE_ORDER = "relative_order_from_other_candidates"
REASON_BUDGET = "previously_unexhausted"


def _rank_table(research: ResearchSet) -> Dict[str, int]:
    ordered = research.reading_order()
    return {doc_id: index + 1 for index, doc_id in enumerate(ordered)}


def compare_research_sets(old: ResearchSet, new: ResearchSet) -> dict:
    """Compare two runs of the same reviewing perspective.

    Every enter/leave/move carries an attributed reason: the document body
    changed, the query meaning (text or synonym version) changed, or only the
    relative order changed because other candidates joined/left.  Manual
    decisions from the old set are inherited only while their evidence basis
    still matches the new per-query evidence.
    """
    old_rank = _rank_table(old)
    new_rank = _rank_table(new)
    old_ids = set(old_rank)
    new_ids = set(new_rank)
    old_queries = {spec.label: (spec.raw, old.rules.version) for spec in old.specs}
    new_queries = {spec.label: (spec.raw, new.rules.version) for spec in new.specs}
    query_meaning_changed = old_queries != new_queries

    changes: List[dict] = []
    for doc_id in sorted(new_ids - old_ids):
        changes.append({
            "id": doc_id,
            "change": DOCUMENT_ENTERED,
            "old_rank": None,
            "new_rank": new_rank[doc_id],
            "reasons": _enter_reasons(old, new, doc_id, query_meaning_changed),
        })
    for doc_id in sorted(old_ids - new_ids):
        changes.append({
            "id": doc_id,
            "change": DOCUMENT_LEFT,
            "old_rank": old_rank[doc_id],
            "new_rank": None,
            "reasons": _leave_reasons(old, new, doc_id, query_meaning_changed),
        })
    for doc_id in sorted(old_ids & new_ids):
        delta = old_rank[doc_id] - new_rank[doc_id]
        change = RANK_STABLE if delta == 0 else RANK_MOVED
        changes.append({
            "id": doc_id,
            "change": change,
            "old_rank": old_rank[doc_id],
            "new_rank": new_rank[doc_id],
            "reasons": [] if delta == 0 else _move_reasons(old, new, doc_id, query_meaning_changed),
        })

    inherited, dropped = _inherit_decisions(old, new)
    return {
        "query_meaning_changed": query_meaning_changed,
        "old_exhausted": old.exhausted,
        "new_exhausted": new.exhausted,
        "changes": changes,
        "inherited_decisions": inherited,
        "dropped_decisions": dropped,
        "old_order": [doc_id for doc_id, _ in sorted(old_rank.items(), key=lambda item: item[1])],
        "new_order": [doc_id for doc_id, _ in sorted(new_rank.items(), key=lambda item: item[1])],
    }


def _doc_body(research: ResearchSet, doc_id: str):
    stored = research.documents.get(doc_id)
    return None if stored is None else dict(stored.fields)


def _enter_reasons(old: ResearchSet, new: ResearchSet, doc_id: str, query_changed: bool) -> List[str]:
    reasons: List[str] = []
    if doc_id not in old.documents:
        reasons.append(REASON_DOCUMENT_CHANGED)
    elif query_changed:
        reasons.append(REASON_QUERY_CHANGED)
    if _doc_body(old, doc_id) != _doc_body(new, doc_id) and REASON_DOCUMENT_CHANGED not in reasons:
        reasons.append(REASON_DOCUMENT_CHANGED)
    if not old.exhausted:
        reasons.append(REASON_BUDGET)
    return reasons or [REASON_RELATIVE_ORDER]


def _leave_reasons(old: ResearchSet, new: ResearchSet, doc_id: str, query_changed: bool) -> List[str]:
    reasons: List[str] = []
    if doc_id not in new.documents:
        reasons.append(REASON_DOCUMENT_CHANGED)
    elif _doc_body(old, doc_id) != _doc_body(new, doc_id):
        reasons.append(REASON_DOCUMENT_CHANGED)
    if query_changed:
        reasons.append(REASON_QUERY_CHANGED)
    return reasons or [REASON_RELATIVE_ORDER]


def _move_reasons(old: ResearchSet, new: ResearchSet, doc_id: str, query_changed: bool) -> List[str]:
    reasons: List[str] = []
    if _doc_body(old, doc_id) != _doc_body(new, doc_id):
        reasons.append(REASON_DOCUMENT_CHANGED)
    if query_changed:
        reasons.append(REASON_QUERY_CHANGED)
    if old.evidence.get(doc_id, {}).keys() != new.evidence.get(doc_id, {}).keys() and not query_changed:
        reasons.append(REASON_DOCUMENT_CHANGED)
    if not reasons:
        reasons.append(REASON_RELATIVE_ORDER)
    return reasons


def _decision_basis_still_matches(old: ResearchSet, new: ResearchSet, doc_id: str) -> bool:
    decision = old.decisions[doc_id]
    if not decision.basis:
        # Exception for a document that never matched: identical body required.
        return _doc_body(old, doc_id) == _doc_body(new, doc_id)
    new_evidence = new.evidence.get(doc_id, {})
    for label, old_sig in decision.basis.items():
        current = new_evidence.get(label)
        if current is None or current.signature != old_sig:
            return False
    return True


def _inherit_decisions(old: ResearchSet, new: ResearchSet) -> Tuple[List[dict], List[dict]]:
    inherited: List[dict] = []
    dropped: List[dict] = []
    for doc_id, decision in old.decisions.items():
        if doc_id not in new.documents:
            dropped.append({"id": doc_id, "action": decision.action, "reason": "document_absent"})
            continue
        if _decision_basis_still_matches(old, new, doc_id):
            new_evidence = new.evidence.get(doc_id, {})
            basis = {
                label: new_evidence[label].signature
                for label in decision.basis
                if label in new_evidence
            }
            inherited_decision = Decision(
                action=decision.action,
                reason=decision.reason,
                basis=basis or decision.basis,
                rules_version=new.rules.version,
                inherited=True,
            )
            new.decisions[doc_id] = inherited_decision
            inherited.append({"id": doc_id, "action": decision.action})
        else:
            dropped.append({"id": doc_id, "action": decision.action, "reason": "evidence_basis_changed"})
    return inherited, dropped
