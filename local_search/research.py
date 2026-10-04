"""Multi-query research sets with per-query evidence and stable review order.

A research set gathers the candidate documents produced by several queries
behind one frozen reviewing view.  Each query keeps its own score and
positional evidence; scores from different queries are never summed into a
single relevance number.  The set pins the exact query texts, synonym rule
version and document bodies used at confirmation time, so confirmed pages
keep a stable order even when the live corpus changes later.

Several independent review perspectives can read the same frozen set.  Each
perspective keeps its own read position and its own reasoned decisions, so
one reviewer turning a page or recording a decision never changes another
reviewer's unreviewed queue.  Merging perspectives reports consensus, still
unreviewed documents and genuine disagreements separately; a disagreement is
never settled by overwriting, only by an explicit resolution decision that
references the original judgements.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

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

DEFAULT_PERSPECTIVE = "main"

# Merge outcome categories.
MERGE_CONSENSUS = "consensus"
MERGE_CONFLICT = "conflict"
MERGE_UNREVIEWED = "unreviewed"
MERGE_PARTIAL = "partial"
MERGE_RESOLVED = "resolved"
MERGE_PENDING_REVIEW = "pending_review"

ACTIVE = "active"
NEEDS_REVIEW = "needs_review"


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
    references: List[dict] = field(default_factory=list)
    decision_id: Optional[str] = None
    batch_id: Optional[str] = None
    status: str = ACTIVE
    withdrawal_reasons: List[dict] = field(default_factory=list)
    cited_decision_ids: List[str] = field(default_factory=list)


@dataclass
class WithdrawalBatch:
    batch_id: str
    perspective: str
    reason: str
    withdrawn_batch_id: Optional[str]
    reviews: List[dict] = field(default_factory=list)


@dataclass
class Perspective:
    """One reviewer's independent reading state on a frozen research set."""

    name: str
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


def _decision_payload(decision: Decision) -> dict:
    payload = {
        "action": decision.action,
        "reason": decision.reason,
        "basis": decision.basis,
        "rules_version": decision.rules_version,
        "inherited": decision.inherited,
        "references": list(decision.references),
    }
    if decision.decision_id is not None:
        payload["decision_id"] = decision.decision_id
    if decision.batch_id is not None:
        payload["batch_id"] = decision.batch_id
    if decision.status != ACTIVE:
        payload["status"] = decision.status
    if decision.withdrawal_reasons:
        payload["withdrawal_reasons"] = list(decision.withdrawal_reasons)
    if decision.cited_decision_ids:
        payload["cited_decision_ids"] = list(decision.cited_decision_ids)
    return payload


def _decision_from_payload(entry: dict) -> Decision:
    return Decision(
        action=entry["action"],
        reason=entry["reason"],
        basis=entry.get("basis", {}),
        rules_version=entry.get("rules_version"),
        inherited=bool(entry.get("inherited", False)),
        references=list(entry.get("references", [])),
        decision_id=entry.get("decision_id"),
        batch_id=entry.get("batch_id"),
        status=entry.get("status", ACTIVE),
        withdrawal_reasons=list(entry.get("withdrawal_reasons", [])),
        cited_decision_ids=list(entry.get("cited_decision_ids", [])),
    )


def _withdrawal_payload(withdrawal: WithdrawalBatch) -> dict:
    return {
        "withdrawal_id": withdrawal.batch_id,
        "batch_id": withdrawal.withdrawn_batch_id,
        "perspective": withdrawal.perspective,
        "reason": withdrawal.reason,
        "reviews": list(withdrawal.reviews),
    }


def _withdrawal_from_payload(entry: dict) -> WithdrawalBatch:
    return WithdrawalBatch(
        batch_id=entry.get("withdrawal_id") or entry["batch_id"],
        perspective=entry["perspective"],
        reason=entry["reason"],
        withdrawn_batch_id=entry.get("batch_id"),
        reviews=list(entry.get("reviews", [])),
    )


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
        self.perspectives: Dict[str, Perspective] = {
            DEFAULT_PERSPECTIVE: Perspective(DEFAULT_PERSPECTIVE)
        }
        self.resolutions: Dict[str, Decision] = {}
        self.resolution_history: Dict[str, List[Decision]] = {}
        self.withdrawals: List[WithdrawalBatch] = []
        self.evidence: Dict[str, Dict[str, CandidateEvidence]] = {}
        self.outcomes: Dict[str, RankingOutcome] = {}
        self.exhausted = False
        self._recompute()

    # ----------------------------------------------------- perspectives
    @property
    def decisions(self):
        # Decisions of the default perspective (backwards-compatible view).
        return self.perspectives[DEFAULT_PERSPECTIVE].decisions

    @property
    def read_ids(self):
        # Read positions of the default perspective.
        return self.perspectives[DEFAULT_PERSPECTIVE].read_ids

    @read_ids.setter
    def read_ids(self, value):
        self.perspectives[DEFAULT_PERSPECTIVE].read_ids = value

    def _persp(self, name):
        key = name or DEFAULT_PERSPECTIVE
        if key not in self.perspectives:
            raise KeyError('Unknown review perspective: ' + str(key))
        return self.perspectives[key]

    def add_perspective(self, name):
        # Create (or return) an independent review perspective on this set.
        if not name:
            raise ValueError('Perspective name must be non-empty')
        if name not in self.perspectives:
            self.perspectives[name] = Perspective(name)
        return self.perspectives[name]

    def perspective(self, name):
        # Return the review handle for one independent perspective.
        self.add_perspective(name)
        return PerspectiveView(self, name)

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

    def _ordered_candidate_ids(self, perspective: Optional[str] = None) -> Tuple[List[str], List[str]]:
        """Return (evaluated_order, pending_unscored) document ids."""
        view = self._persp(perspective)
        evaluated: List[str] = []
        pending: List[str] = []
        candidate_pool = set(self.evidence) | {
            doc_id
            for doc_id, decision in view.decisions.items()
            if decision.action == INCLUDED
        }
        for doc_id in sorted(candidate_pool):
            is_candidate, fully = self._membership(doc_id)
            decision = view.decisions.get(doc_id)
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

    def reading_order(self, perspective: Optional[str] = None) -> List[str]:
        """Full reading list: frozen already-read prefix, then current order.

        Confirmed documents keep the order in which they were first read;
        every not-yet-read document follows in the current deterministic
        cross-query order.  Continuation therefore never drops or duplicates
        a document, and a fresh exhaustive read of the same perspective is
        exactly this order.  Each perspective owns its read prefix, so one
        reviewer's progress never reshapes another reviewer's queue.
        """
        view = self._persp(perspective)
        evaluated, pending = self._ordered_candidate_ids(perspective)
        current = evaluated + pending
        read = [doc_id for doc_id in view.read_ids if doc_id in current]
        seen = set(read)
        return read + [doc_id for doc_id in current if doc_id not in seen]

    def page(
        self,
        offset: int = 0,
        page_size: Optional[int] = None,
        perspective: Optional[str] = None,
    ) -> dict:
        ordered = self.reading_order(perspective)
        end = len(ordered) if page_size is None else min(offset + page_size, len(ordered))
        ids = ordered[offset:end]
        next_offset = end if end < len(ordered) else None
        _, pending = self._ordered_candidate_ids(perspective)
        return {
            "results": [self.document_payload(doc_id, perspective) for doc_id in ids],
            "offset": offset,
            "next_offset": next_offset,
            "exhausted": self.exhausted,
            "complete": self.exhausted and next_offset is None,
            "total_candidates": len(ordered),
            "pending_unscored": len(pending),
        }

    def read_page(self, page_size: Optional[int] = None, perspective: Optional[str] = None) -> dict:
        """Read the next unreviewed slice and mark it reviewed."""
        view = self._persp(perspective)
        ordered = self.reading_order(perspective)
        offset = len([doc_id for doc_id in view.read_ids if doc_id in ordered])
        payload = self.page(offset, page_size, perspective)
        for result in payload["results"]:
            if result["id"] not in view.read_ids:
                view.read_ids.append(result["id"])
        payload["reviewed_count"] = len(view.read_ids)
        return payload

    def unreviewed(self, perspective: Optional[str] = None) -> List[str]:
        view = self._persp(perspective)
        ordered = self.reading_order(perspective)
        return [doc_id for doc_id in ordered if doc_id not in view.read_ids]

    def review_status(self, perspective: Optional[str] = None) -> dict:
        view = self._persp(perspective)
        evaluated, pending = self._ordered_candidate_ids(perspective)
        current = set(evaluated + pending)
        return {
            "exhausted": self.exhausted,
            "perspective": view.name,
            "candidate_count": len(current),
            "reviewed": [doc_id for doc_id in view.read_ids if doc_id in current],
            "unreviewed": self.unreviewed(perspective),
            "excluded": sorted(
                doc_id for doc_id, decision in view.decisions.items() if decision.action == EXCLUDED
            ),
            "manual_included": sorted(
                doc_id for doc_id, decision in view.decisions.items() if decision.action == INCLUDED
            ),
            "needs_re_resolution": sorted(
                doc_id
                for doc_id, decision in self.resolutions.items()
                if decision.status == NEEDS_REVIEW
            ),
            "pending_unscored": sorted(pending),
        }

    def decide(
        self,
        doc_id: str,
        action: str,
        reason: str,
        perspective: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> Decision:
        if action not in (INCLUDED, EXCLUDED):
            raise ValueError("decision action must be 'included' or 'excluded'")
        if not reason or not reason.strip():
            raise ValueError("A manual decision needs an explicit reason")
        if doc_id not in self.documents:
            raise KeyError(doc_id)
        view = self._persp(perspective)
        is_candidate, _ = self._membership(doc_id)
        basis = {
            label: evidence.signature
            for label, evidence in self.evidence.get(doc_id, {}).items()
        }
        if not is_candidate and action != INCLUDED:
            raise ValueError("Only an inclusion exception can be recorded for a non-matching document")
        decision = Decision(
            action,
            reason.strip(),
            basis,
            self.rules.version,
            decision_id=uuid4().hex,
            batch_id=batch_id,
        )
        view.decisions[doc_id] = decision
        return decision

    def decide_batch(
        self,
        reviews: Sequence[Tuple[str, str, str]],
        perspective: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> dict:
        """Record several opinions from one reviewer as one review batch."""
        if not reviews:
            raise ValueError("A review batch needs at least one review")
        review_tuples = [(str(doc_id), action, reason) for doc_id, action, reason in reviews]
        doc_ids = [doc_id for doc_id, _, _ in review_tuples]
        if len(doc_ids) != len(set(doc_ids)):
            raise ValueError("Review batch document ids must be unique")
        for doc_id, action, reason in review_tuples:
            if action not in (INCLUDED, EXCLUDED):
                raise ValueError("decision action must be 'included' or 'excluded'")
            if not reason or not reason.strip():
                raise ValueError("A manual decision needs an explicit reason")
            if doc_id not in self.documents:
                raise KeyError(doc_id)
            is_candidate, _ = self._membership(doc_id)
            if not is_candidate and action != INCLUDED:
                raise ValueError("Only an inclusion exception can be recorded for a non-matching document")
        batch_id = batch_id or uuid4().hex
        decisions = {
            doc_id: self.decide(doc_id, action, reason, perspective, batch_id=batch_id)
            for doc_id, action, reason in review_tuples
        }
        return {
            "batch_id": batch_id,
            "perspective": self._persp(perspective).name,
            "decisions": decisions,
        }

    def withdraw_reviews(
        self,
        reason: str,
        perspective: Optional[str] = None,
        *,
        batch_id: Optional[str] = None,
        doc_ids: Optional[Sequence[str]] = None,
        withdrawn_batch_id: Optional[str] = None,
    ) -> dict:
        """Withdraw one reviewer's current opinions in one audited batch."""
        if not reason or not reason.strip():
            raise ValueError("A withdrawal needs an explicit reason")
        review_batch_id = batch_id if batch_id is not None else withdrawn_batch_id
        if review_batch_id is None and not doc_ids:
            raise ValueError("A withdrawal needs a review batch id or document ids")
        view = self._persp(perspective)
        selected_ids = [str(doc_id) for doc_id in doc_ids] if doc_ids is not None else []
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("Withdrawal document ids must be unique")

        removable: List[Tuple[str, Decision]] = []
        for doc_id, decision in list(view.decisions.items()):
            if decision.status != ACTIVE:
                continue
            if review_batch_id is not None and decision.batch_id != review_batch_id:
                continue
            if selected_ids and doc_id not in selected_ids:
                continue
            removable.append((doc_id, decision))
        if not removable:
            raise KeyError("No active reviews match the withdrawal selection")

        matched_docs = {doc_id for doc_id, _ in removable}
        missing = sorted(set(selected_ids) - matched_docs)
        if missing:
            raise KeyError("Some selected reviews do not belong to this reviewer and batch: " + ", ".join(missing))

        withdrawal_id = uuid4().hex
        reviews = [
            {
                "id": doc_id,
                "action": decision.action,
                "reason": decision.reason,
                "decision_id": decision.decision_id,
                "review_batch_id": decision.batch_id,
            }
            for doc_id, decision in sorted(removable)
        ]
        withdrawal = WithdrawalBatch(
            batch_id=withdrawal_id,
            perspective=view.name,
            reason=reason.strip(),
            withdrawn_batch_id=review_batch_id,
            reviews=reviews,
        )
        self.withdrawals.append(withdrawal)
        for doc_id, _ in removable:
            del view.decisions[doc_id]
        affected_resolutions = self._mark_resolutions_for_review(
            view.name, matched_docs, withdrawal
        )
        return {
            "withdrawal_id": withdrawal_id,
            "batch_id": review_batch_id,
            "perspective": view.name,
            "withdrawn_ids": sorted(matched_docs),
            "resolutions_needing_review": sorted(affected_resolutions),
            "reason": reason.strip(),
        }

    def withdraw_batch(
        self,
        batch_id: str,
        reason: Optional[str] = None,
        perspective: Optional[str] = None,
        *,
        withdrawal_reason: Optional[str] = None,
        doc_ids: Optional[Sequence[str]] = None,
    ) -> dict:
        actual_reason = reason or withdrawal_reason
        if actual_reason is None:
            raise ValueError("A withdrawal needs an explicit reason")
        return self.withdraw_reviews(
            actual_reason,
            perspective,
            batch_id=batch_id,
            doc_ids=doc_ids,
        )

    def withdraw_review_batch(
        self,
        review_batch_id: str,
        reason: str,
        perspective: Optional[str] = None,
    ) -> dict:
        return self.withdraw_reviews(
            reason,
            perspective,
            batch_id=review_batch_id,
        )

    def batch_withdraw(
        self,
        batch_id: str,
        reason: str,
        perspective: Optional[str] = None,
    ) -> dict:
        return self.withdraw_batch(batch_id, reason, perspective)

    def _mark_resolutions_for_review(
        self, perspective: str, doc_ids: set, withdrawal: WithdrawalBatch
    ) -> set:
        affected = set()
        for doc_id in doc_ids:
            review = next(entry for entry in withdrawal.reviews if entry["id"] == doc_id)
            rulings = [self.resolutions.get(doc_id)] + list(self.resolution_history.get(doc_id, []))
            for resolution in rulings:
                if resolution is not None and self._resolution_cites_review(
                    resolution, perspective, review
                ):
                    resolution.status = NEEDS_REVIEW
                    resolution.withdrawal_reasons.append({
                        "perspective": perspective,
                        "reason": withdrawal.reason,
                        "withdrawal_id": withdrawal.batch_id,
                    })
                    affected.add(doc_id)
        return affected

    @staticmethod
    def _resolution_cites_review(resolution: Decision, perspective: str, review: dict) -> bool:
        review_id = review.get("decision_id")
        if review_id is not None:
            return review_id in resolution.cited_decision_ids
        if resolution.cited_decision_ids:
            return False
        if review_id in resolution.cited_decision_ids:
            return True
        return any(
            ref.get("perspective") == perspective
            and ref.get("action") == review.get("action")
            and ref.get("reason") == review.get("reason")
            for ref in resolution.references
        )

    # ----------------------------------------------------- merging
    def merge_perspectives(self, names: Optional[Sequence[str]] = None) -> dict:
        """Merge review perspectives without overwriting any judgement.

        Every candidate document is classified as consensus (all perspectives
        that reviewed it agree and none is still pending), conflict (real
        disagreement between recorded decisions), unreviewed (no perspective
        decided yet), partial (some decided, some pending) or resolved (an
        explicit resolution decision exists).  The per-perspective decisions
        themselves are never mutated by a merge.
        """
        if names is not None:
            selected = list(names)
        else:
            # Perspectives without any reading or decision activity are not
            # reviewers yet and would otherwise block every consensus.
            active = [
                name
                for name, view in self.perspectives.items()
                if view.read_ids or view.decisions
            ]
            selected = active or list(self.perspectives)
        views = [self._persp(name) for name in selected]
        pool = set(self.evidence) | set(self.resolutions)
        for view in views:
            pool |= set(view.decisions)
        documents: List[dict] = []
        buckets: Dict[str, List[str]] = {
            MERGE_CONSENSUS: [],
            MERGE_CONFLICT: [],
            MERGE_UNREVIEWED: [],
            MERGE_PARTIAL: [],
            MERGE_RESOLVED: [],
            MERGE_PENDING_REVIEW: [],
        }
        for doc_id in sorted(pool):
            decided = {
                view.name: {"action": view.decisions[doc_id].action, "reason": view.decisions[doc_id].reason}
                for view in views
                if doc_id in view.decisions
            }
            resolution = self.resolutions.get(doc_id)
            if resolution is not None and resolution.status == NEEDS_REVIEW:
                status = MERGE_PENDING_REVIEW
            elif resolution is not None:
                status = MERGE_RESOLVED
            elif not decided:
                status = MERGE_UNREVIEWED
            elif len({entry["action"] for entry in decided.values()}) > 1:
                status = MERGE_CONFLICT
            elif len(decided) == len(views):
                status = MERGE_CONSENSUS
            else:
                status = MERGE_PARTIAL
            buckets[status].append(doc_id)
            documents.append({
                "id": doc_id,
                "status": status,
                "decisions": decided,
                "reviewed_by": sorted(view.name for view in views if doc_id in view.read_ids),
                "resolution": (
                    {
                        "action": resolution.action,
                        "reason": resolution.reason,
                        "status": resolution.status,
                        "withdrawal_reasons": list(resolution.withdrawal_reasons),
                        "references": list(resolution.references),
                    }
                    if resolution is not None
                    else None
                ),
            })
        return {
            "perspectives": [view.name for view in views],
            "documents": documents,
            "consensus": buckets[MERGE_CONSENSUS],
            "conflicts": buckets[MERGE_CONFLICT],
            "unreviewed": buckets[MERGE_UNREVIEWED],
            "partial": buckets[MERGE_PARTIAL],
            "resolved": buckets[MERGE_RESOLVED],
            "pending_review": buckets[MERGE_PENDING_REVIEW],
        }

    def resolve(self, doc_id: str, action: str, reason: str) -> Decision:
        """Settle a genuine disagreement with a new referencing decision.

        The resolution never overwrites the conflicting per-perspective
        decisions; it is recorded separately and cites each of them, so the
        conflict evidence stays intact.
        """
        if action not in (INCLUDED, EXCLUDED):
            raise ValueError("resolution action must be 'included' or 'excluded'")
        if not reason or not reason.strip():
            raise ValueError("A resolution needs an explicit reason")
        if doc_id not in self.documents:
            raise KeyError(doc_id)
        decided = {
            name: view.decisions[doc_id]
            for name, view in self.perspectives.items()
            if doc_id in view.decisions
        }
        previous = self.resolutions.get(doc_id)
        real_disagreement = len({decision.action for decision in decided.values()}) >= 2
        if not real_disagreement and (previous is None or previous.status != NEEDS_REVIEW):
            raise ValueError("A resolution requires a real disagreement or a pending re-review")
        references = [
            {"perspective": name, "action": decision.action, "reason": decision.reason}
            for name, decision in sorted(decided.items())
        ]
        cited_decision_ids = [
            decision.decision_id
            for decision in decided.values()
            if decision.decision_id is not None
        ]
        basis = {
            label: evidence.signature
            for label, evidence in self.evidence.get(doc_id, {}).items()
        }
        if previous is not None:
            self.resolution_history.setdefault(doc_id, []).append(previous)
        resolution = Decision(
            action,
            reason.strip(),
            basis,
            self.rules.version,
            references=references,
            cited_decision_ids=cited_decision_ids,
        )
        self.resolutions[doc_id] = resolution
        return resolution

    def document_payload(self, doc_id: str, perspective: Optional[str] = None) -> dict:
        stored = self.documents[doc_id]
        view = self._persp(perspective)
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
        decision = view.decisions.get(doc_id)
        resolution = self.resolutions.get(doc_id)
        return {
            "id": doc_id,
            "fields": dict(stored.fields),
            "matched_queries": sorted(self.evidence.get(doc_id, {})),
            "per_query": per_query,
            "review": decision.action if decision is not None else PENDING_REVIEW,
            "review_reason": decision.reason if decision is not None else None,
            "reviewed": doc_id in view.read_ids,
            "resolution_status": resolution.status if resolution is not None else None,
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
                doc_id: _decision_payload(decision)
                for doc_id, decision in self.decisions.items()
            },
            "perspectives": {
                name: {
                    "read_ids": list(view.read_ids),
                    "decisions": {
                        doc_id: _decision_payload(decision)
                        for doc_id, decision in view.decisions.items()
                    },
                }
                for name, view in self.perspectives.items()
            },
            "resolutions": {
                doc_id: _decision_payload(decision)
                for doc_id, decision in self.resolutions.items()
            },
            "resolution_history": {
                doc_id: [_decision_payload(decision) for decision in decisions]
                for doc_id, decisions in self.resolution_history.items()
            },
            "withdrawals": [_withdrawal_payload(entry) for entry in self.withdrawals],
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
            research.decisions[doc_id] = _decision_from_payload(entry)
        for name, view_payload in payload.get("perspectives", {}).items():
            view = research.add_perspective(name)
            view.read_ids = list(view_payload.get("read_ids", []))
            for doc_id, entry in view_payload.get("decisions", {}).items():
                view.decisions[doc_id] = _decision_from_payload(entry)
        for doc_id, entry in payload.get("resolutions", {}).items():
            research.resolutions[doc_id] = _decision_from_payload(entry)
        for doc_id, entries in payload.get("resolution_history", {}).items():
            research.resolution_history[doc_id] = [
                _decision_from_payload(entry) for entry in entries
            ]
        research.withdrawals = [
            _withdrawal_from_payload(entry) for entry in payload.get("withdrawals", [])
        ]
        return research


class PerspectiveView:
    """Review handle bound to one independent perspective of a research set.

    All reading and deciding through this handle touches only that
    perspective's read positions and decisions; other perspectives on the
    same frozen set are unaffected.
    """

    def __init__(self, research: "ResearchSet", name: str) -> None:
        self.research = research
        self.name = name

    def reading_order(self) -> List[str]:
        return self.research.reading_order(self.name)

    def page(self, offset: int = 0, page_size: Optional[int] = None) -> dict:
        return self.research.page(offset, page_size, self.name)

    def read_page(self, page_size: Optional[int] = None) -> dict:
        return self.research.read_page(page_size, self.name)

    def unreviewed(self) -> List[str]:
        return self.research.unreviewed(self.name)

    def review_status(self) -> dict:
        return self.research.review_status(self.name)

    def decide(self, doc_id: str, action: str, reason: str) -> Decision:
        return self.research.decide(doc_id, action, reason, self.name)

    def decide_batch(self, reviews, batch_id: Optional[str] = None) -> dict:
        return self.research.decide_batch(reviews, self.name, batch_id=batch_id)

    def withdraw_reviews(
        self,
        reason: str,
        *,
        batch_id: Optional[str] = None,
        doc_ids=None,
        withdrawn_batch_id: Optional[str] = None,
        ) -> dict:
        return self.research.withdraw_reviews(
            reason,
            self.name,
            batch_id=batch_id,
            doc_ids=doc_ids,
            withdrawn_batch_id=withdrawn_batch_id,
        )

    def withdraw_batch(self, batch_id: str, reason: str) -> dict:
        return self.research.withdraw_batch(
            batch_id,
            reason,
            self.name,
        )

    def batch_withdraw(self, batch_id: str, reason: str) -> dict:
        return self.withdraw_batch(batch_id, reason)

    def withdraw_review_batch(self, review_batch_id: str, reason: str) -> dict:
        return self.research.withdraw_review_batch(review_batch_id, reason, self.name)

    def document_payload(self, doc_id: str) -> dict:
        return self.research.document_payload(doc_id, self.name)
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


def _decision_basis_still_matches(
    old: ResearchSet, new: ResearchSet, doc_id: str, decision: Decision
) -> bool:
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
    new.withdrawals = list(old.withdrawals)
    for perspective_name, old_view in old.perspectives.items():
        new_view = new.add_perspective(perspective_name)
        # Surviving documents keep their read position; deleted documents
        # drop out of the read prefix without duplicating any position.
        seen: set = set()
        new_view.read_ids = [
            doc_id
            for doc_id in old_view.read_ids
            if doc_id in new.documents and not (doc_id in seen or seen.add(doc_id))
        ]
        for doc_id, decision in old_view.decisions.items():
            if doc_id not in new.documents:
                dropped.append({
                    "id": doc_id,
                    "perspective": perspective_name,
                    "action": decision.action,
                    "reason": "document_absent",
                })
                continue
            if _decision_basis_still_matches(old, new, doc_id, decision):
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
                new_view.decisions[doc_id] = inherited_decision
                inherited.append({
                    "id": doc_id,
                    "perspective": perspective_name,
                    "action": decision.action,
                })
            else:
                dropped.append({
                    "id": doc_id,
                    "perspective": perspective_name,
                    "action": decision.action,
                    "reason": "evidence_basis_changed",
                })
    new.resolution_history = {
        doc_id: [
            Decision(
                action=decision.action,
                reason=decision.reason,
                basis=dict(decision.basis),
                rules_version=decision.rules_version,
                inherited=True,
                references=list(decision.references),
                decision_id=decision.decision_id,
                batch_id=decision.batch_id,
                status=decision.status,
                withdrawal_reasons=list(decision.withdrawal_reasons),
                cited_decision_ids=list(decision.cited_decision_ids),
            )
            for decision in decisions
            if doc_id in new.documents
        ]
        for doc_id, decisions in old.resolution_history.items()
    }
    new.resolution_history = {
        doc_id: decisions for doc_id, decisions in new.resolution_history.items() if decisions
    }
    # Active resolutions survive derivation only while every decision they
    # cite is itself still inherited by the same perspective.
    for doc_id, resolution in old.resolutions.items():
        if doc_id not in new.documents:
            dropped.append({
                "id": doc_id,
                "perspective": None,
                "action": resolution.action,
                "reason": "document_absent",
            })
            continue
        if resolution.status == NEEDS_REVIEW:
            new.resolutions[doc_id] = Decision(
                action=resolution.action,
                reason=resolution.reason,
                basis=dict(resolution.basis),
                rules_version=resolution.rules_version,
                inherited=True,
                references=list(resolution.references),
                decision_id=resolution.decision_id,
                batch_id=resolution.batch_id,
                status=NEEDS_REVIEW,
                withdrawal_reasons=list(resolution.withdrawal_reasons),
                cited_decision_ids=list(resolution.cited_decision_ids),
            )
            inherited.append({"id": doc_id, "perspective": None, "action": resolution.action})
            continue
        cited = {ref["perspective"] for ref in resolution.references}
        still_cited = all(
            doc_id in new.perspectives.get(name, Perspective(name)).decisions
            for name in cited
        )
        if not still_cited:
            dropped.append({
                "id": doc_id,
                "perspective": None,
                "action": resolution.action,
                "reason": "evidence_basis_changed",
            })
            continue
        if not _decision_basis_still_matches(old, new, doc_id, resolution):
            dropped.append({
                "id": doc_id,
                "perspective": None,
                "action": resolution.action,
                "reason": "evidence_basis_changed",
            })
            continue
        new_evidence = new.evidence.get(doc_id, {})
        basis = {
            label: new_evidence[label].signature
            for label in resolution.basis
            if label in new_evidence
        }
        new.resolutions[doc_id] = Decision(
            action=resolution.action,
            reason=resolution.reason,
            basis=basis or resolution.basis,
            rules_version=new.rules.version,
            inherited=True,
            references=list(resolution.references),
            cited_decision_ids=list(resolution.cited_decision_ids),
        )
        inherited.append({"id": doc_id, "perspective": None, "action": resolution.action})
    return inherited, dropped
