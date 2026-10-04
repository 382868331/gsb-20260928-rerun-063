"""Multi-query research sets with per-query evidence and stable review order.

A research set gathers the candidate documents produced by several queries
behind one reviewing perspective.  Each query keeps its own score and
positional evidence; scores from different queries are never summed into a
single relevance number.  The set pins the exact query texts, synonym rule
version and document bodies used at confirmation time, so confirmed pages
keep a stable order even when the live corpus changes later.

Several independent reviewer perspectives can read and judge the same
frozen set.  Each reviewer keeps its own read positions and decisions;
one reviewer's paging or deciding never touches another reviewer's
unreviewed state.  Merging the perspectives distinguishes consensus,
unreviewed documents and genuine disagreements.  A disagreement is never
silently overwritten by the last write: resolving it records a new
resolution decision that cites the original conflicting judgements.
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

DEFAULT_REVIEWER = "main"

CONSENSUS_INCLUDED = "consensus_included"
CONSENSUS_EXCLUDED = "consensus_excluded"
MERGE_UNREVIEWED = "unreviewed"
MERGE_CONFLICT = "conflict"
MERGE_RESOLVED = "resolved"


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
    resolves: Optional[List[dict]] = None  # conflicting decisions this resolution cites


@dataclass
class ReviewerState:
    """One reviewer's private reading position and judgements."""

    read_ids: List[str] = field(default_factory=list)
    decisions: Dict[str, Decision] = field(default_factory=dict)


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
        self.reviewers: Dict[str, ReviewerState] = {DEFAULT_REVIEWER: ReviewerState()}
        self.resolutions: Dict[str, Decision] = {}
        self.evidence: Dict[str, Dict[str, CandidateEvidence]] = {}
        self.outcomes: Dict[str, RankingOutcome] = {}
        self.exhausted = False
        self._recompute()

    # ------------------------------------------------------- reviewer state

    @property
    def read_ids(self) -> List[str]:
        """Read positions of the default reviewer (backwards compatible)."""
        return self.reviewers[DEFAULT_REVIEWER].read_ids

    @property
    def decisions(self) -> Dict[str, Decision]:
        """Decisions of the default reviewer (backwards compatible)."""
        return self.reviewers[DEFAULT_REVIEWER].decisions

    def reviewer_state(self, reviewer: str = DEFAULT_REVIEWER) -> ReviewerState:
        if not reviewer:
            raise ValueError("Reviewer name must be a non-empty string")
        return self.reviewers.setdefault(reviewer, ReviewerState())

    def reviewer_names(self) -> List[str]:
        return sorted(self.reviewers)

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

    def _ordered_candidate_ids(self, reviewer: str = DEFAULT_REVIEWER) -> Tuple[List[str], List[str]]:
        """Return (evaluated_order, pending_unscored) document ids."""
        decisions = self.reviewer_state(reviewer).decisions
        evaluated: List[str] = []
        pending: List[str] = []
        candidate_pool = set(self.evidence) | {
            doc_id
            for doc_id, decision in decisions.items()
            if decision.action == INCLUDED
        }
        for doc_id in sorted(candidate_pool):
            is_candidate, fully = self._membership(doc_id)
            decision = decisions.get(doc_id)
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

    def reading_order(self, reviewer: str = DEFAULT_REVIEWER) -> List[str]:
        """Full reading list: frozen already-read prefix, then current order.

        Confirmed documents keep the order in which they were first read;
        every not-yet-read document follows in the current deterministic
        cross-query order.  Continuation therefore never drops or duplicates
        a document, and a fresh exhaustive read of the same perspective is
        exactly this order.
        """
        state = self.reviewer_state(reviewer)
        evaluated, pending = self._ordered_candidate_ids(reviewer)
        current = evaluated + pending
        read = [doc_id for doc_id in state.read_ids if doc_id in current]
        seen = set(read)
        return read + [doc_id for doc_id in current if doc_id not in seen]

    def page(self, offset: int = 0, page_size: Optional[int] = None, reviewer: str = DEFAULT_REVIEWER) -> dict:
        ordered = self.reading_order(reviewer)
        end = len(ordered) if page_size is None else min(offset + page_size, len(ordered))
        ids = ordered[offset:end]
        next_offset = end if end < len(ordered) else None
        _, pending = self._ordered_candidate_ids(reviewer)
        return {
            "results": [self.document_payload(doc_id, reviewer) for doc_id in ids],
            "offset": offset,
            "next_offset": next_offset,
            "exhausted": self.exhausted,
            "complete": self.exhausted and next_offset is None,
            "total_candidates": len(ordered),
            "pending_unscored": len(pending),
        }

    def read_page(self, page_size: Optional[int] = None, reviewer: str = DEFAULT_REVIEWER) -> dict:
        """Read the next unreviewed slice and mark it reviewed for this reviewer."""
        state = self.reviewer_state(reviewer)
        ordered = self.reading_order(reviewer)
        offset = len([doc_id for doc_id in state.read_ids if doc_id in ordered])
        payload = self.page(offset, page_size, reviewer)
        for result in payload["results"]:
            if result["id"] not in state.read_ids:
                state.read_ids.append(result["id"])
        payload["reviewed_count"] = len(state.read_ids)
        payload["reviewer"] = reviewer
        return payload

    def unreviewed(self, reviewer: str = DEFAULT_REVIEWER) -> List[str]:
        state = self.reviewer_state(reviewer)
        ordered = self.reading_order(reviewer)
        return [doc_id for doc_id in ordered if doc_id not in state.read_ids]

    def review_status(self, reviewer: str = DEFAULT_REVIEWER) -> dict:
        state = self.reviewer_state(reviewer)
        evaluated, pending = self._ordered_candidate_ids(reviewer)
        current = set(evaluated + pending)
        return {
            "reviewer": reviewer,
            "exhausted": self.exhausted,
            "candidate_count": len(current),
            "reviewed": [doc_id for doc_id in state.read_ids if doc_id in current],
            "unreviewed": self.unreviewed(reviewer),
            "excluded": sorted(
                doc_id for doc_id, decision in state.decisions.items() if decision.action == EXCLUDED
            ),
            "manual_included": sorted(
                doc_id for doc_id, decision in state.decisions.items() if decision.action == INCLUDED
            ),
            "pending_unscored": sorted(pending),
        }

    def decide(self, doc_id: str, action: str, reason: str, reviewer: str = DEFAULT_REVIEWER) -> Decision:
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
        self.reviewer_state(reviewer).decisions[doc_id] = decision
        return decision

    def document_payload(self, doc_id: str, reviewer: str = DEFAULT_REVIEWER) -> dict:
        state = self.reviewer_state(reviewer)
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
        decision = state.decisions.get(doc_id)
        return {
            "id": doc_id,
            "fields": dict(stored.fields),
            "matched_queries": sorted(self.evidence.get(doc_id, {})),
            "per_query": per_query,
            "review": decision.action if decision is not None else PENDING_REVIEW,
            "review_reason": decision.reason if decision is not None else None,
            "reviewed": doc_id in state.read_ids,
        }

    def continue_scan(self, budget: Optional[int] = None) -> None:
        """Resume scoring with a larger (or unlimited) budget on the same view."""
        if budget is not None and self.budget is not None and budget < self.budget:
            raise ValueError("Continuation can only raise the scan budget")
        self.budget = budget
        self._recompute()

    # ------------------------------------------------------- merging

    def merge_review(self, reviewers: Optional[Sequence[str]] = None) -> dict:
        """Merge every reviewer's judgements without overwriting any of them.

        Each document is classified as a consensus (all deciding reviewers
        agree), still unreviewed (no reviewer decided), a genuine conflict
        (reviewers disagree), or resolved (a resolution decision exists).
        The original per-reviewer decisions are always reported alongside,
        so conflicting evidence is never lost.
        """
        names = list(reviewers) if reviewers is not None else self.reviewer_names()
        for name in names:
            if name not in self.reviewers:
                raise KeyError(f"Unknown reviewer: {name}")
        doc_ids = set(self.evidence) | set(self.resolutions)
        for name in names:
            doc_ids |= set(self.reviewers[name].decisions)

        documents: Dict[str, dict] = {}
        summary = {
            CONSENSUS_INCLUDED: 0,
            CONSENSUS_EXCLUDED: 0,
            MERGE_UNREVIEWED: 0,
            MERGE_CONFLICT: 0,
            MERGE_RESOLVED: 0,
        }
        for doc_id in sorted(doc_ids):
            per_reviewer = {
                name: {
                    "action": self.reviewers[name].decisions[doc_id].action,
                    "reason": self.reviewers[name].decisions[doc_id].reason,
                }
                for name in names
                if doc_id in self.reviewers[name].decisions
            }
            resolution = self.resolutions.get(doc_id)
            if resolution is not None:
                status = MERGE_RESOLVED
            elif not per_reviewer:
                status = MERGE_UNREVIEWED
            elif len({entry["action"] for entry in per_reviewer.values()}) == 1:
                action = next(iter(per_reviewer.values()))["action"]
                status = CONSENSUS_INCLUDED if action == INCLUDED else CONSENSUS_EXCLUDED
            else:
                status = MERGE_CONFLICT
            summary[status] += 1
            documents[doc_id] = {
                "status": status,
                "decisions": per_reviewer,
                "resolution": (
                    None
                    if resolution is None
                    else {
                        "action": resolution.action,
                        "reason": resolution.reason,
                        "resolves": resolution.resolves or [],
                    }
                ),
            }
        return {
            "reviewers": names,
            "documents": documents,
            "summary": summary,
            "conflicts": sorted(
                doc_id for doc_id, entry in documents.items() if entry["status"] == MERGE_CONFLICT
            ),
            "unreviewed": sorted(
                doc_id for doc_id, entry in documents.items() if entry["status"] == MERGE_UNREVIEWED
            ),
        }

    def resolve_conflict(
        self,
        doc_id: str,
        action: str,
        reason: str,
        reviewer: str = DEFAULT_REVIEWER,
    ) -> Decision:
        """Resolve a genuine conflict with a new decision citing the originals.

        The conflicting per-reviewer decisions stay untouched; the resolution
        is a separate decision that references them, so the disagreement
        remains auditable instead of being overwritten.
        """
        if action not in (INCLUDED, EXCLUDED):
            raise ValueError("resolution action must be 'included' or 'excluded'")
        if not reason or not reason.strip():
            raise ValueError("A conflict resolution needs an explicit reason")
        conflicting = [
            {"reviewer": name, "action": state.decisions[doc_id].action, "reason": state.decisions[doc_id].reason}
            for name, state in sorted(self.reviewers.items())
            if doc_id in state.decisions
        ]
        if len({entry["action"] for entry in conflicting}) < 2:
            raise ValueError(f"Document {doc_id} has no genuine conflict to resolve")
        basis = {
            label: evidence.signature
            for label, evidence in self.evidence.get(doc_id, {}).items()
        }
        resolution = Decision(
            action,
            reason.strip(),
            basis,
            self.rules.version,
            resolves=conflicting,
        )
        self.resolutions[doc_id] = resolution
        return resolution

    def to_payload(self) -> dict:
        return {
            "name": self.name,
            "mode": self.mode,
            "budget": self.budget,
            "ranker": self.ranker_name,
            "rules": self.rules.to_payload(),
            "queries": [{"label": spec.label, "raw": spec.raw} for spec in self.specs],
            "documents": {doc_id: dict(doc.fields) for doc_id, doc in self.documents.items()},
            "reviewers": {
                name: {
                    "read_ids": list(state.read_ids),
                    "decisions": {
                        doc_id: _decision_payload(decision)
                        for doc_id, decision in state.decisions.items()
                    },
                }
                for name, state in sorted(self.reviewers.items())
            },
            "resolutions": {
                doc_id: _decision_payload(decision)
                for doc_id, decision in self.resolutions.items()
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
        if "reviewers" in payload:
            research.reviewers = {}
            for name, state_payload in payload["reviewers"].items():
                research.reviewers[name] = ReviewerState(
                    read_ids=list(state_payload.get("read_ids", [])),
                    decisions={
                        doc_id: _decision_from_payload(entry)
                        for doc_id, entry in state_payload.get("decisions", {}).items()
                    },
                )
            if DEFAULT_REVIEWER not in research.reviewers:
                research.reviewers[DEFAULT_REVIEWER] = ReviewerState()
        else:
            # Legacy single-reviewer payload.
            research.reviewers = {
                DEFAULT_REVIEWER: ReviewerState(
                    read_ids=list(payload.get("read_ids", [])),
                    decisions={
                        doc_id: _decision_from_payload(entry)
                        for doc_id, entry in payload.get("decisions", {}).items()
                    },
                )
            }
        research.resolutions = {
            doc_id: _decision_from_payload(entry)
            for doc_id, entry in payload.get("resolutions", {}).items()
        }
        return research


def _decision_payload(decision: Decision) -> dict:
    return {
        "action": decision.action,
        "reason": decision.reason,
        "basis": decision.basis,
        "rules_version": decision.rules_version,
        "inherited": decision.inherited,
        "resolves": decision.resolves,
    }


def _decision_from_payload(entry: dict) -> Decision:
    return Decision(
        action=entry["action"],
        reason=entry["reason"],
        basis=entry.get("basis", {}),
        rules_version=entry.get("rules_version"),
        inherited=bool(entry.get("inherited", False)),
        resolves=entry.get("resolves"),
    )
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

    inherited, dropped = _inherit_reviewers(old, new)
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


def _decision_basis_still_matches(old: ResearchSet, new: ResearchSet, decision: Decision, doc_id: str) -> bool:
    if not decision.basis:
        # Exception for a document that never matched: identical body required.
        return _doc_body(old, doc_id) == _doc_body(new, doc_id)
    new_evidence = new.evidence.get(doc_id, {})
    for label, old_sig in decision.basis.items():
        current = new_evidence.get(label)
        if current is None or current.signature != old_sig:
            return False
    return True


def _inherit_one_decision(old: ResearchSet, new: ResearchSet, decision: Decision, doc_id: str) -> Decision:
    new_evidence = new.evidence.get(doc_id, {})
    basis = {
        label: new_evidence[label].signature
        for label in decision.basis
        if label in new_evidence
    }
    return Decision(
        action=decision.action,
        reason=decision.reason,
        basis=basis or decision.basis,
        rules_version=new.rules.version,
        inherited=True,
        resolves=decision.resolves,
    )


def _inherit_reviewers(old: ResearchSet, new: ResearchSet) -> Tuple[List[dict], List[dict]]:
    """Carry each reviewer's decisions, read positions and resolutions over.

    Every reviewer keeps their own perspective in the derived set.  A
    decision is inherited only while its recorded evidence basis still
    matches; documents whose content changed or that were deleted are
    reported explicitly instead of silently losing or copying judgements.
    Read positions survive only for documents still present, and never
    duplicate.
    """
    inherited: List[dict] = []
    dropped: List[dict] = []
    new_current = set(new.reading_order())
    for reviewer, state in old.reviewers.items():
        new_state = new.reviewer_state(reviewer)
        seen: set = set()
        for doc_id in state.read_ids:
            if doc_id in new_current and doc_id not in seen:
                new_state.read_ids.append(doc_id)
                seen.add(doc_id)
        for doc_id, decision in state.decisions.items():
            if doc_id not in new.documents:
                dropped.append({
                    "id": doc_id,
                    "reviewer": reviewer,
                    "action": decision.action,
                    "reason": "document_absent",
                })
                continue
            if _decision_basis_still_matches(old, new, decision, doc_id):
                new_state.decisions[doc_id] = _inherit_one_decision(old, new, decision, doc_id)
                inherited.append({"id": doc_id, "reviewer": reviewer, "action": decision.action})
            else:
                dropped.append({
                    "id": doc_id,
                    "reviewer": reviewer,
                    "action": decision.action,
                    "reason": "evidence_basis_changed",
                })
    for doc_id, resolution in old.resolutions.items():
        if doc_id not in new.documents:
            dropped.append({"id": doc_id, "reviewer": None, "action": resolution.action, "reason": "document_absent"})
            continue
        if _decision_basis_still_matches(old, new, resolution, doc_id):
            new.resolutions[doc_id] = _inherit_one_decision(old, new, resolution, doc_id)
            inherited.append({"id": doc_id, "reviewer": None, "action": resolution.action})
        else:
            dropped.append({
                "id": doc_id,
                "reviewer": None,
                "action": resolution.action,
                "reason": "evidence_basis_changed",
            })
    return inherited, dropped
