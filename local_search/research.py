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

from dataclasses import dataclass, field, replace
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

DEFAULT_PERSPECTIVE = "main"

# Merge outcome categories.
MERGE_CONSENSUS = "consensus"
MERGE_CONFLICT = "conflict"
MERGE_UNREVIEWED = "unreviewed"
MERGE_PARTIAL = "partial"
MERGE_RESOLVED = "resolved"
MERGE_RECHECK = "recheck"

DECISION_ACTIVE = "active"
DECISION_WITHDRAWN = "withdrawn"
RESOLUTION_PENDING_RECHECK = "pending_recheck"
REFERENCE_ACTIVE = "active"
REFERENCE_REMOVED = "removed"
PLAN_DRAFT = "draft"
PLAN_APPLIED = "applied"
PLAN_CANCELLED = "cancelled"


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
    status: str = DECISION_ACTIVE
    withdrawal_batches: List[str] = field(default_factory=list)
    opinion_id: Optional[str] = None


@dataclass
class WithdrawalBatch:
    """One researcher's atomic withdrawal of several own decisions."""

    batch_id: str
    perspective: str
    reason: str
    decisions: Dict[str, Decision] = field(default_factory=dict)
    removed_references: List[dict] = field(default_factory=list)


@dataclass
class WithdrawalPlan:
    """A persisted, reviewable proposal before opinions are withdrawn."""

    plan_id: str
    perspective: str
    reason: str
    opinion_ids: List[str]
    reference_ids: List[str]
    retain_resolutions: List[str]
    status: str = PLAN_DRAFT
    opinion_snapshots: Dict[str, dict] = field(default_factory=dict)
    reference_snapshots: Dict[str, dict] = field(default_factory=dict)
    preview: dict = field(default_factory=dict)


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
    return {
        "action": decision.action,
        "reason": decision.reason,
        "basis": decision.basis,
        "rules_version": decision.rules_version,
        "inherited": decision.inherited,
        "references": list(decision.references),
        "status": decision.status,
        "withdrawal_batches": list(decision.withdrawal_batches),
        "opinion_id": decision.opinion_id,
    }


def _decision_from_payload(entry: dict) -> Decision:
    return Decision(
        action=entry["action"],
        reason=entry["reason"],
        basis=entry.get("basis", {}),
        rules_version=entry.get("rules_version"),
        inherited=bool(entry.get("inherited", False)),
        references=list(entry.get("references", [])),
        status=entry.get("status", DECISION_ACTIVE),
        withdrawal_batches=list(entry.get("withdrawal_batches", [])),
        opinion_id=entry.get("opinion_id"),
    )


def _withdrawal_payload(batch: WithdrawalBatch) -> dict:
    return {
        "batch_id": batch.batch_id,
        "perspective": batch.perspective,
        "reason": batch.reason,
        "decisions": {
            doc_id: _decision_payload(decision)
            for doc_id, decision in batch.decisions.items()
        },
        "removed_references": list(batch.removed_references),
    }


def _withdrawal_from_payload(entry: dict) -> WithdrawalBatch:
    return WithdrawalBatch(
        batch_id=entry["batch_id"],
        perspective=entry["perspective"],
        reason=entry["reason"],
        decisions={
            doc_id: _decision_from_payload(payload)
            for doc_id, payload in entry.get("decisions", {}).items()
        },
        removed_references=list(entry.get("removed_references", [])),
    )


def _plan_payload(plan: WithdrawalPlan) -> dict:
    return {
        "plan_id": plan.plan_id,
        "perspective": plan.perspective,
        "reason": plan.reason,
        "opinion_ids": list(plan.opinion_ids),
        "reference_ids": list(plan.reference_ids),
        "retain_resolutions": list(plan.retain_resolutions),
        "status": plan.status,
        "opinion_snapshots": {key: dict(value) for key, value in plan.opinion_snapshots.items()},
        "reference_snapshots": {key: dict(value) for key, value in plan.reference_snapshots.items()},
        "preview": plan.preview,
    }


def _plan_from_payload(entry: dict) -> WithdrawalPlan:
    return WithdrawalPlan(
        plan_id=entry["plan_id"],
        perspective=entry["perspective"],
        reason=entry["reason"],
        opinion_ids=list(entry.get("opinion_ids", [])),
        reference_ids=list(entry.get("reference_ids", [])),
        retain_resolutions=list(entry.get("retain_resolutions", [])),
        status=entry.get("status", PLAN_DRAFT),
        opinion_snapshots={key: dict(value) for key, value in entry.get("opinion_snapshots", {}).items()},
        reference_snapshots={key: dict(value) for key, value in entry.get("reference_snapshots", {}).items()},
        preview=dict(entry.get("preview", {})),
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
        self.withdrawals: Dict[str, WithdrawalBatch] = {}
        self.withdrawal_plans: Dict[str, WithdrawalPlan] = {}
        self._reference_sequence = 0
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

    def _all_decisions(self):
        seen = set()
        def emit(decision):
            key = id(decision)
            if key not in seen:
                seen.add(key)
                return decision
            return None
        for view in self.perspectives.values():
            for decision in view.decisions.values():
                current = emit(decision)
                if current is not None:
                    yield current
        for decision in self.resolutions.values():
            current = emit(decision)
            if current is not None:
                yield current
        for history in self.resolution_history.values():
            for decision in history:
                current = emit(decision)
                if current is not None:
                    yield current
        for batch in self.withdrawals.values():
            for decision in batch.decisions.values():
                current = emit(decision)
                if current is not None:
                    yield current

    def _opinion_number(self, opinion_id: Optional[str]) -> int:
        if not opinion_id or not opinion_id.startswith("opinion-"):
            return 0
        suffix = opinion_id.removeprefix("opinion-")
        return int(suffix) if suffix.isdigit() else 0

    def _next_opinion_id(self) -> str:
        next_number = max((self._opinion_number(decision.opinion_id) for decision in self._all_decisions()), default=0) + 1
        return f"opinion-{next_number}"

    def find_opinion(self, opinion_id: str) -> Optional[Tuple[str, str, Decision]]:
        """Return (doc_id, perspective, decision) for a stable opinion id."""
        for perspective, view in self.perspectives.items():
            for doc_id, decision in view.decisions.items():
                if decision.opinion_id == opinion_id:
                    return doc_id, perspective, decision
        return None

    def _reference_key(self, doc_id: str, index: int) -> str:
        return f"{doc_id}#{index + 1}"

    def _iter_references(self):
        containers = []
        containers.extend(self.resolutions.items())
        for history_doc, history in self.resolution_history.items():
            containers.extend((history_doc, decision) for decision in history)
        for doc_id, decision in containers:
            for index, ref in enumerate(decision.references):
                ref.setdefault("doc_id", doc_id)
                yield doc_id, decision, index, self._reference_key(doc_id, index), ref

    def find_reference(self, reference_id: str) -> Optional[Tuple[str, Decision, int, dict]]:
        for doc_id, decision, index, current_id, ref in self._iter_references():
            if ref.get("reference_id") == reference_id or current_id == reference_id:
                if doc_id in self.resolutions and self.resolutions[doc_id] is decision:
                    return doc_id, decision, index, ref
        return None

    def _assign_legacy_identity(self) -> None:
        for view in self.perspectives.values():
            for doc_id in sorted(view.decisions):
                decision = view.decisions[doc_id]
                if decision.opinion_id is None:
                    decision.opinion_id = self._next_opinion_id()
        for doc_id in sorted(self.resolutions):
            decision = self.resolutions[doc_id]
            if decision.opinion_id is None:
                decision.opinion_id = self._next_opinion_id()
        for doc_id in sorted(self.resolution_history):
            for decision in self.resolution_history[doc_id]:
                if decision.opinion_id is None:
                    decision.opinion_id = self._next_opinion_id()
        seen_refs = set()
        for _, _, _, reference_id, ref in self._iter_references():
            ref.setdefault("reference_id", reference_id)
            ref.setdefault("status", REFERENCE_ACTIVE)
            ref.setdefault("doc_id", None)
            if not ref.get("opinion_id"):
                located = self.find_perspective_opinion(ref.get("perspective"), ref.get("doc_id"))
                ref["opinion_id"] = located.opinion_id if located is not None else None
            while ref["reference_id"] in seen_refs:
                ref["reference_id"] = ref["reference_id"] + ":legacy"
            seen_refs.add(ref["reference_id"])

    def find_perspective_opinion(self, perspective: Optional[str], doc_id: Optional[str]) -> Optional[Decision]:
        if perspective is None or doc_id is None:
            return None
        return self.perspectives.get(perspective, Perspective(perspective)).decisions.get(doc_id)

    def _next_reference_id(self) -> str:
        self._reference_sequence += 1
        return f"ref-{self._reference_sequence}"

    def _resolution_refs_after_proposal(
        self,
        resolution: Decision,
        withdrawn_opinions: set,
        removed_references: set,
    ) -> List[dict]:
        refs = []
        for ref in resolution.references:
            ref_id = ref.get("reference_id")
            opinion_id = ref.get("opinion_id")
            if ref.get("status", REFERENCE_ACTIVE) != REFERENCE_ACTIVE:
                continue
            if ref_id in removed_references or opinion_id in withdrawn_opinions:
                continue
            located = self.find_opinion(opinion_id) if opinion_id else None
            if located is not None and located[2].status == DECISION_ACTIVE:
                refs.append({"ref": ref, "decision": located[2], "perspective": located[1]})
        return refs

    def preview_withdrawal(
        self,
        perspective: Optional[str],
        reason: str,
        opinion_ids: Optional[Sequence[str]] = None,
        reference_ids: Optional[Sequence[str]] = None,
        retain_resolutions: Optional[Sequence[str]] = None,
    ) -> dict:
        view = self._persp(perspective)
        opinion_ids = set(opinion_ids or [])
        reference_ids = set(reference_ids or [])
        retain_resolutions = set(retain_resolutions or [])
        if not reason or not reason.strip():
            raise ValueError("A withdrawal needs an explicit reason")
        if not opinion_ids and not reference_ids:
            raise ValueError("A withdrawal batch needs at least one opinion or citation")
        for opinion_id in opinion_ids:
            located = self.find_opinion(opinion_id)
            if located is None:
                raise KeyError("Unknown opinion: " + opinion_id)
            doc_id, owner, decision = located
            if owner != view.name:
                raise ValueError("Opinion belongs to another perspective: " + opinion_id)
            if decision.status != DECISION_ACTIVE:
                raise ValueError("Already withdrawn: " + opinion_id)
        for reference_id in reference_ids:
            located = self.find_reference(reference_id)
            if located is None:
                raise KeyError("Unknown citation: " + reference_id)
            _, _, _, ref = located
            if ref.get("perspective") != view.name:
                raise ValueError("Citation belongs to another perspective: " + reference_id)
            if ref.get("status", REFERENCE_ACTIVE) != REFERENCE_ACTIVE:
                raise ValueError("Citation already withdrawn: " + reference_id)

        pending = []
        independent = []
        dependencies = []
        for doc_id, resolution in self.resolutions.items():
            active_refs = self._resolution_refs_after_proposal(resolution, opinion_ids, reference_ids)
            affected_refs = [
                ref
                for ref in resolution.references
                if ref.get("perspective") == view.name
                and ref.get("status", REFERENCE_ACTIVE) == REFERENCE_ACTIVE
                and (
                    ref.get("reference_id") in reference_ids
                    or ref.get("opinion_id") in opinion_ids
                )
            ]
            if not affected_refs:
                continue
            supporters = [
                entry
                for entry in active_refs
                if entry["decision"].action == resolution.action
            ]
            unique_supporter_ids = {entry["decision"].opinion_id for entry in supporters}
            supporter_perspectives = {entry["perspective"] for entry in supporters}
            record = {
                "doc_id": doc_id,
                "resolution_action": resolution.action,
                "withdrawn_citations": [ref.get("reference_id") for ref in affected_refs],
                "independent_support": [
                    {
                        "opinion_id": opinion_id,
                        "perspective": self.find_opinion(opinion_id)[1],
                        "action": self.find_opinion(opinion_id)[2].action,
                    }
                    for opinion_id in sorted(unique_supporter_ids)
                ],
            }
            dependencies.append(record)
            if supporter_perspectives - {view.name}:
                independent.append(record)
            else:
                pending.append(record)
        conflicts = [
            {
                "doc_id": doc_id,
                "resolution_action": self.resolutions[doc_id].action,
                "reason": "required_resolution_depends_only_on_withdrawn_support",
                "dependencies": [entry for entry in pending if entry["doc_id"] == doc_id],
            }
            for doc_id in sorted(retain_resolutions)
            if doc_id not in self.resolutions or any(entry["doc_id"] == doc_id for entry in pending)
        ]
        return {
            "feasible": not conflicts,
            "perspective": view.name,
            "reason": reason.strip(),
            "withdraw_opinions": sorted(opinion_ids),
            "remove_citations": sorted(reference_ids),
            "pending_recheck": pending,
            "kept_with_independent_support": independent,
            "dependencies": dependencies,
            "retain_resolutions": sorted(retain_resolutions),
            "conflicts": conflicts,
            "unchanged_opinions": sorted(
                decision.opinion_id
                for owner in self.perspectives.values()
                for decision in owner.decisions.values()
                if decision.status == DECISION_ACTIVE and decision.opinion_id not in opinion_ids
            ),
        }

    def plan_withdrawal(
        self,
        perspective: Optional[str],
        reason: str,
        opinion_ids: Optional[Sequence[str]] = None,
        reference_ids: Optional[Sequence[str]] = None,
        retain_resolutions: Optional[Sequence[str]] = None,
        *,
        doc_ids: Optional[Sequence[str]] = None,
        plan_id: Optional[str] = None,
    ) -> WithdrawalPlan:
        view = self._persp(perspective)
        opinion_ids = list(opinion_ids or [])
        if doc_ids:
            opinion_ids.extend(view.decisions[doc_id].opinion_id for doc_id in doc_ids)
        opinion_ids = list(dict.fromkeys(opinion_ids))
        reference_ids = list(dict.fromkeys(reference_ids or []))
        preview = self.preview_withdrawal(
            view.name,
            reason,
            opinion_ids,
            reference_ids,
            retain_resolutions,
        )
        if plan_id is None:
            sequence = max(
                (
                    int(pid.removeprefix("plan-"))
                    for pid in self.withdrawal_plans
                    if pid.startswith("plan-") and pid.removeprefix("plan-").isdigit()
                ),
                default=0,
            ) + 1
            plan_id = f"plan-{sequence}"
        if plan_id in self.withdrawal_plans:
            raise ValueError("Withdrawal plan id must be unique")
        opinion_snapshots = {
            opinion_id: {
                "status": self.find_opinion(opinion_id)[2].status,
                "reason": self.find_opinion(opinion_id)[2].reason,
            }
            for opinion_id in opinion_ids
        }
        reference_snapshots = {}
        for reference_id in reference_ids:
            _, _, _, ref = self.find_reference(reference_id)
            reference_snapshots[reference_id] = {key: ref.get(key) for key in ("status", "perspective", "opinion_id")}
        plan = WithdrawalPlan(
            plan_id,
            view.name,
            reason.strip(),
            opinion_ids,
            reference_ids,
            list(dict.fromkeys(retain_resolutions or [])),
            opinion_snapshots=opinion_snapshots,
            reference_snapshots=reference_snapshots,
            preview=preview,
        )
        self.withdrawal_plans[plan_id] = plan
        return plan

    def resume_withdrawal_plan(self, plan_id: str) -> dict:
        plan = self.withdrawal_plans.get(plan_id)
        if plan is None:
            raise KeyError(plan_id)
        if plan.status != PLAN_DRAFT:
            raise ValueError("Only draft plans can be resumed")
        warnings = []
        for opinion_id, snapshot in plan.opinion_snapshots.items():
            located = self.find_opinion(opinion_id)
            if located is None:
                warnings.append({"opinion_id": opinion_id, "change": "missing"})
                continue
            current = {"status": located[2].status, "reason": located[2].reason}
            if current["status"] != snapshot["status"]:
                warnings.append({"opinion_id": opinion_id, "change": "already_withdrawn"})
        for reference_id, snapshot in plan.reference_snapshots.items():
            located = self.find_reference(reference_id)
            if located is None:
                warnings.append({"reference_id": reference_id, "change": "missing"})
                continue
            current = {
                "status": located[3].get("status", REFERENCE_ACTIVE),
                "perspective": located[3].get("perspective"),
                "opinion_id": located[3].get("opinion_id"),
            }
            if current["status"] != snapshot.get("status", REFERENCE_ACTIVE):
                warnings.append({"reference_id": reference_id, "change": "already_removed"})
        preview = self.preview_withdrawal(
            plan.perspective,
            plan.reason,
            plan.opinion_ids,
            plan.reference_ids,
            plan.retain_resolutions,
        ) if not warnings else plan.preview
        return {"plan_id": plan_id, "status": plan.status, "warnings": warnings, "preview": preview}

    def apply_withdrawal_plan(self, plan_id: str) -> WithdrawalBatch:
        resume = self.resume_withdrawal_plan(plan_id)
        if resume["warnings"]:
            raise ValueError("Saved plan is stale; reopen and create a fresh plan")
        plan = self.withdrawal_plans[plan_id]
        if not plan.preview.get("feasible", False):
            raise ValueError("Plan conflicts with required retained resolutions")
        batch = self.withdraw_decisions(
            reason=plan.reason,
            perspective=plan.perspective,
            opinion_ids=plan.opinion_ids,
            reference_ids=plan.reference_ids,
            retain_resolutions=plan.retain_resolutions,
        )
        plan.status = PLAN_APPLIED
        plan.preview["batch_id"] = batch.batch_id
        return batch

    def cancel_withdrawal_plan(self, plan_id: str) -> None:
        plan = self.withdrawal_plans.get(plan_id)
        if plan is None:
            raise KeyError(plan_id)
        if plan.status != PLAN_DRAFT:
            raise ValueError("Only draft plans can be cancelled")
        plan.status = PLAN_CANCELLED

    def plan_minimal_withdrawal(
        self,
        perspective: Optional[str],
        reason: str,
        opinion_ids: Sequence[str],
        retain_resolutions: Sequence[str],
        *,
        plan_id: Optional[str] = None,
    ) -> WithdrawalPlan:
        view = self._persp(perspective)
        requested_opinion_ids = list(dict.fromkeys(opinion_ids))
        retained = set(retain_resolutions)
        preview = self.preview_withdrawal(
            view.name,
            reason,
            requested_opinion_ids,
            [],
            retained,
        )
        removed_reference_ids = []
        if retained:
            for conflict in preview["conflicts"]:
                doc_id = conflict["doc_id"]
                resolution = self.resolutions[doc_id]
                for ref in resolution.references:
                    if (
                        ref.get("perspective") == view.name
                        and ref.get("opinion_id") in requested_opinion_ids
                        and ref.get("status", REFERENCE_ACTIVE) == REFERENCE_ACTIVE
                    ):
                        removed_reference_ids.append(ref["reference_id"])
            for doc_id in retained:
                resolution = self.resolutions.get(doc_id)
                if resolution is None:
                    continue
                for ref in resolution.references:
                    if (
                        ref.get("perspective") == view.name
                        and ref.get("opinion_id") in requested_opinion_ids
                        and ref.get("status", REFERENCE_ACTIVE) == REFERENCE_ACTIVE
                    ):
                        removed_reference_ids.append(ref["reference_id"])
            remaining_opinion_ids = []
            for opinion_id in requested_opinion_ids:
                source_doc, _, _ = self.find_opinion(opinion_id)
                still_needed_elsewhere = any(
                    doc_id != source_doc
                    and doc_id in retained
                    and self.resolutions[doc_id].status == DECISION_ACTIVE
                    and any(
                        ref.get("perspective") == view.name
                        and ref.get("opinion_id") == opinion_id
                        and ref.get("status", REFERENCE_ACTIVE) == REFERENCE_ACTIVE
                        for ref in self.resolutions[doc_id].references
                    )
                    for doc_id in self.resolutions
                )
                if still_needed_elsewhere:
                    remaining_opinion_ids.append(opinion_id)
            preview = self.preview_withdrawal(
                view.name,
                reason,
                remaining_opinion_ids,
                removed_reference_ids,
                retained,
            )
        else:
            remaining_opinion_ids = requested_opinion_ids
        preview["requested_opinions"] = requested_opinion_ids
        preview["minimal_opinions"] = list(remaining_opinion_ids)
        preview["minimal_citations"] = list(dict.fromkeys(removed_reference_ids))
        if not preview["feasible"]:
            raise ValueError("No minimal withdrawal satisfies required retained resolutions")
        plan = self.plan_withdrawal(
            view.name,
            reason,
            remaining_opinion_ids,
            list(dict.fromkeys(removed_reference_ids)),
            retained,
            plan_id=plan_id,
        )
        plan.preview.update(preview)
        return plan

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
            if decision.status == DECISION_ACTIVE and decision.action == INCLUDED
        }
        for doc_id in sorted(candidate_pool):
            is_candidate, fully = self._membership(doc_id)
            decision = view.decisions.get(doc_id)
            if not is_candidate and decision is None:
                continue
            if decision is not None and decision.status == DECISION_ACTIVE and decision.action == EXCLUDED:
                continue
            if not is_candidate and (
                decision is None or decision.status != DECISION_ACTIVE or decision.action != INCLUDED
            ):
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
                doc_id
                for doc_id, decision in view.decisions.items()
                if decision.status == DECISION_ACTIVE and decision.action == EXCLUDED
            ),
            "manual_included": sorted(
                doc_id
                for doc_id, decision in view.decisions.items()
                if decision.status == DECISION_ACTIVE and decision.action == INCLUDED
            ),
            "withdrawn": sorted(
                doc_id
                for doc_id, decision in view.decisions.items()
                if decision.status == DECISION_WITHDRAWN
            ),
            "pending_unscored": sorted(pending),
            "recheck_resolutions": sorted(
                doc_id
                for doc_id, resolution in self.resolutions.items()
                if resolution.status == RESOLUTION_PENDING_RECHECK
            ),
        }

    def decide(
        self,
        doc_id: str,
        action: str,
        reason: str,
        perspective: Optional[str] = None,
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
            opinion_id=self._next_opinion_id(),
        )
        view.decisions[doc_id] = decision
        return decision

    def withdraw_decisions(
        self,
        doc_ids: Optional[Sequence[str]] = None,
        reason: Optional[str] = None,
        perspective: Optional[str] = None,
        *,
        opinion_ids: Optional[Sequence[str]] = None,
        reference_ids: Optional[Sequence[str]] = None,
        retain_resolutions: Optional[Sequence[str]] = None,
        batch_id: Optional[str] = None,
    ) -> WithdrawalBatch:
        """Apply a batch withdrawal by stable opinion and/or citation ids."""
        view = self._persp(perspective)
        opinion_ids = list(opinion_ids or [])
        reference_ids = list(reference_ids or [])
        retain_resolutions = set(retain_resolutions or [])
        if doc_ids:
            opinion_ids.extend(
                view.decisions[doc_id].opinion_id
                for doc_id in dict.fromkeys(doc_ids)
            )
        opinion_ids = list(dict.fromkeys(opinion_id for opinion_id in opinion_ids if opinion_id))
        reference_ids = list(dict.fromkeys(reference_ids))
        if not opinion_ids and not reference_ids:
            raise ValueError("A withdrawal batch needs at least one opinion or citation")
        if reason is None or not reason.strip():
            raise ValueError("A withdrawal needs an explicit reason")

        selected: Dict[str, Decision] = {}
        for opinion_id in opinion_ids:
            located = self.find_opinion(opinion_id)
            if located is None:
                raise KeyError("Unknown opinion: " + opinion_id)
            doc_id, owner, decision = located
            if owner != view.name:
                raise ValueError("Opinion belongs to another perspective: " + opinion_id)
            if decision.status != DECISION_ACTIVE:
                raise ValueError("Already withdrawn: " + opinion_id)
            selected[doc_id] = decision

        selected_references: List[Tuple[str, Decision, int, dict]] = []
        for reference_id in reference_ids:
            located_ref = self.find_reference(reference_id)
            if located_ref is None:
                raise KeyError("Unknown citation: " + reference_id)
            doc_id, resolution, index, ref = located_ref
            if ref.get("perspective") != view.name:
                raise ValueError("Citation belongs to another perspective: " + reference_id)
            if ref.get("status", REFERENCE_ACTIVE) != REFERENCE_ACTIVE:
                raise ValueError("Citation already withdrawn: " + reference_id)
            selected_references.append((doc_id, resolution, index, ref))

        proposed_preview = self.preview_withdrawal(
            view.name,
            reason.strip(),
            opinion_ids,
            reference_ids,
            retain_resolutions,
        )
        if retain_resolutions and not proposed_preview["feasible"]:
            raise ValueError(
                "Required confirmed resolutions would become pending: "
                + ", ".join(entry["doc_id"] for entry in proposed_preview["conflicts"])
            )

        already_withdrawn = [
            opinion_id
            for opinion_id in opinion_ids
            if self.find_opinion(opinion_id) is not None
            and self.find_opinion(opinion_id)[2].status != DECISION_ACTIVE
        ]
        if already_withdrawn:
            raise ValueError("Already withdrawn: " + ", ".join(already_withdrawn))

        if batch_id is None:
            sequence = max(
                (
                    int(bid.removeprefix("withdrawal-"))
                    for bid in self.withdrawals
                    if bid.startswith("withdrawal-") and bid.removeprefix("withdrawal-").isdigit()
                ),
                default=0,
            ) + 1
            batch_id = f"withdrawal-{sequence}"
        if not batch_id or batch_id in self.withdrawals:
            raise ValueError("Withdrawal batch id must be unique and non-empty")

        archived = {
            doc_id: replace(
                decision,
                status=DECISION_WITHDRAWN,
                withdrawal_batches=[*decision.withdrawal_batches, batch_id],
            )
            for doc_id, decision in selected.items()
        }
        removed_references = []
        for doc_id, resolution, index, ref in selected_references:
            removed = {
                **ref,
                "status": REFERENCE_REMOVED,
                "withdrawn_batch": batch_id,
                "withdrawn_reason": reason.strip(),
            }
            resolution.references[index] = removed
            removed_references.append(dict(removed))

        batch = WithdrawalBatch(batch_id, view.name, reason.strip(), archived, removed_references)
        self.withdrawals[batch_id] = batch
        for doc_id, decision in archived.items():
            view.decisions[doc_id] = decision
        for doc_id, resolution in list(self.resolutions.items()):
            cites_withdrawn = any(
                ref.get("perspective") == view.name
                and ref.get("opinion_id") in opinion_ids
                and ref.get("status", REFERENCE_ACTIVE) == REFERENCE_ACTIVE
                for ref in resolution.references
            )
            if not cites_withdrawn:
                continue
            if self._resolution_has_independent_support(resolution, batch_id, reason.strip(), doc_id):
                continue
            self.resolution_history.setdefault(doc_id, []).append(resolution)
            self.resolutions[doc_id] = replace(resolution, status=RESOLUTION_PENDING_RECHECK)
            for ref in self.resolutions[doc_id].references:
                if ref.get("perspective") == view.name and ref.get("status", REFERENCE_ACTIVE) == REFERENCE_ACTIVE:
                    source = self.find_opinion(ref.get("opinion_id")) if ref.get("opinion_id") else None
                    if source is not None and source[2].status == DECISION_WITHDRAWN:
                        ref["withdrawn_batch"] = batch_id
                        ref["withdrawn_reason"] = batch.reason
        return batch

    def _active_reference_opinions(self, resolution: Decision) -> List[Decision]:
        result = []
        for ref in resolution.references:
            if ref.get("status", REFERENCE_ACTIVE) != REFERENCE_ACTIVE:
                continue
            located = self.find_opinion(ref.get("opinion_id")) if ref.get("opinion_id") else None
            if located is not None and located[2].status == DECISION_ACTIVE:
                result.append(located[2])
        return result

    def _resolution_has_independent_support(self, resolution: Decision, batch_id: str, reason: str, doc_id: str) -> bool:
        supporting = []
        for ref in resolution.references:
            if ref.get("status", REFERENCE_ACTIVE) != REFERENCE_ACTIVE:
                continue
            located = self.find_opinion(ref.get("opinion_id")) if ref.get("opinion_id") else None
            if located is not None and located[2].status == DECISION_ACTIVE and located[2].action == resolution.action:
                supporting.append((ref, located))
        if not supporting:
            return False
        supporting_perspectives = {located[1] for _, located in supporting}
        batch_perspective = self.withdrawals[batch_id].perspective if batch_id in self.withdrawals else None
        return bool(supporting_perspectives - {batch_perspective})

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
                if view.read_ids or any(
                    decision.status == DECISION_ACTIVE for decision in view.decisions.values()
                )
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
            MERGE_RECHECK: [],
        }
        for batch in self.withdrawals.values():
            pool |= set(batch.decisions)
        for doc_id in sorted(pool):
            decided = {
                view.name: {
                    "action": view.decisions[doc_id].action,
                    "reason": view.decisions[doc_id].reason,
                }
                for view in views
                if doc_id in view.decisions and view.decisions[doc_id].status == DECISION_ACTIVE
            }
            resolution = self.resolutions.get(doc_id)
            if resolution is not None and resolution.status == RESOLUTION_PENDING_RECHECK:
                status = MERGE_RECHECK
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
                        "references": list(resolution.references),
                        "status": resolution.status,
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
            "recheck": buckets[MERGE_RECHECK],
            "resolution_history": {
                doc_id: [
                    {
                        "action": decision.action,
                        "reason": decision.reason,
                        "references": list(decision.references),
                        "status": decision.status,
                    }
                    for decision in history
                ]
                for doc_id, history in self.resolution_history.items()
            },
        }

    def resolve(
        self,
        doc_id: str,
        action: str,
        reason: str,
        support_opinion_ids: Optional[Sequence[str]] = None,
    ) -> Decision:
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
            if doc_id in view.decisions and view.decisions[doc_id].status == DECISION_ACTIVE
        }
        existing = self.resolutions.get(doc_id)
        if existing is not None and existing.status != RESOLUTION_PENDING_RECHECK:
            raise ValueError("An active resolution can only change after recheck")
        explicit_support_requested = support_opinion_ids is not None
        support_opinion_ids = list(support_opinion_ids or [])
        extra_supports = []
        for opinion_id in support_opinion_ids:
            located = self.find_opinion(opinion_id)
            if located is None:
                raise KeyError("Unknown support opinion: " + opinion_id)
            support_doc, perspective, support_decision = located
            if support_decision.status != DECISION_ACTIVE:
                raise ValueError("Support opinion is not active: " + opinion_id)
            extra_supports.append((support_doc, perspective, support_decision))
        if existing is None and len({decision.action for decision in decided.values()}) < 2 and not explicit_support_requested:
            raise ValueError("A resolution requires a real disagreement between perspectives")
        references = [
            {
                "reference_id": self._next_reference_id(),
                "doc_id": doc_id,
                "opinion_id": decision.opinion_id,
                "perspective": name,
                "action": decision.action,
                "reason": decision.reason,
                "status": REFERENCE_ACTIVE,
            }
            for name, decision in sorted(decided.items())
        ]
        referenced = {(ref["perspective"], ref.get("doc_id")) for ref in references}
        for support_doc, perspective, support_decision in extra_supports:
            if (perspective, support_doc) in referenced:
                continue
            references.append({
                "reference_id": self._next_reference_id(),
                "doc_id": support_doc,
                "opinion_id": support_decision.opinion_id,
                "perspective": perspective,
                "action": support_decision.action,
                "reason": support_decision.reason,
                "status": REFERENCE_ACTIVE,
            })
        basis = {
            label: evidence.signature
            for label, evidence in self.evidence.get(doc_id, {}).items()
        }
        resolution = Decision(
            action,
            reason.strip(),
            basis,
            self.rules.version,
            references=references,
            opinion_id=self._next_opinion_id(),
        )
        if existing is not None:
            self.resolution_history.setdefault(doc_id, []).append(existing)
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
        withdrawal = None
        if decision is not None and decision.status == DECISION_WITHDRAWN:
            batch_id = decision.withdrawal_batches[-1] if decision.withdrawal_batches else None
            batch = self.withdrawals.get(batch_id) if batch_id else None
            withdrawal = {
                "batch_id": batch_id,
                "reason": batch.reason if batch is not None else None,
            }
        return {
            "id": doc_id,
            "fields": dict(stored.fields),
            "matched_queries": sorted(self.evidence.get(doc_id, {})),
            "per_query": per_query,
            "review": (
                decision.action
                if decision is not None and decision.status == DECISION_ACTIVE
                else DECISION_WITHDRAWN if decision is not None
                else PENDING_REVIEW
            ),
            "review_reason": decision.reason if decision is not None else None,
            "review_opinion_id": decision.opinion_id if decision is not None else None,
            "withdrawal": withdrawal,
            "reviewed": doc_id in view.read_ids,
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
                doc_id: [_decision_payload(decision) for decision in history]
                for doc_id, history in self.resolution_history.items()
            },
            "withdrawals": {
                batch_id: _withdrawal_payload(batch)
                for batch_id, batch in self.withdrawals.items()
            },
            "withdrawal_plans": {
                plan_id: _plan_payload(plan)
                for plan_id, plan in self.withdrawal_plans.items()
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
        for batch_id, entry in payload.get("withdrawals", {}).items():
            research.withdrawals[batch_id] = _withdrawal_from_payload(entry)
        for plan_id, entry in payload.get("withdrawal_plans", {}).items():
            research.withdrawal_plans[plan_id] = _plan_from_payload(entry)
        research._assign_legacy_identity()
        max_ref = 0
        for _, _, _, _, ref in research._iter_references():
            ref_id = ref.get("reference_id", "")
            if ref_id.startswith("ref-"):
                suffix = ref_id.removeprefix("ref-")
                if suffix.isdigit():
                    max_ref = max(max_ref, int(suffix))
        research._reference_sequence = max_ref
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

    def withdraw_decisions(
        self,
        doc_ids: Optional[Sequence[str]] = None,
        reason: Optional[str] = None,
        *,
        opinion_ids: Optional[Sequence[str]] = None,
        reference_ids: Optional[Sequence[str]] = None,
        retain_resolutions: Optional[Sequence[str]] = None,
    ) -> WithdrawalBatch:
        return self.research.withdraw_decisions(
            doc_ids,
            reason,
            self.name,
            opinion_ids=opinion_ids,
            reference_ids=reference_ids,
            retain_resolutions=retain_resolutions,
        )

    def document_payload(self, doc_id: str) -> dict:
        return self.research.document_payload(doc_id, self.name)

    def preview_withdrawal(
        self,
        reason: str,
        opinion_ids: Optional[Sequence[str]] = None,
        reference_ids: Optional[Sequence[str]] = None,
        retain_resolutions: Optional[Sequence[str]] = None,
    ) -> dict:
        return self.research.preview_withdrawal(
            self.name, reason, opinion_ids, reference_ids, retain_resolutions
        )

    def plan_withdrawal(
        self,
        reason: str,
        opinion_ids: Optional[Sequence[str]] = None,
        reference_ids: Optional[Sequence[str]] = None,
        retain_resolutions: Optional[Sequence[str]] = None,
        *,
        plan_id: Optional[str] = None,
    ) -> WithdrawalPlan:
        return self.research.plan_withdrawal(
            self.name,
            reason,
            opinion_ids,
            reference_ids,
            retain_resolutions,
            plan_id=plan_id,
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
                inherited_decision = replace(
                    decision,
                    basis=basis or decision.basis,
                    rules_version=new.rules.version,
                    inherited=True,
                )
                new_view.decisions[doc_id] = inherited_decision
                if decision.status == DECISION_ACTIVE:
                    inherited.append({
                        "id": doc_id,
                        "perspective": perspective_name,
                        "action": decision.action,
                    })
            else:
                if decision.status == DECISION_ACTIVE:
                    dropped.append({
                        "id": doc_id,
                        "perspective": perspective_name,
                        "action": decision.action,
                        "reason": "evidence_basis_changed",
                    })
    for batch_id, batch in old.withdrawals.items():
        new.withdrawals[batch_id] = replace(
            batch,
            decisions={
                doc_id: replace(decision, inherited=True)
                for doc_id, decision in batch.decisions.items()
                if doc_id in new.documents and _decision_basis_still_matches(old, new, doc_id, decision)
            },
        )
    # Resolutions survive derivation only while every decision they cite is
    # itself still inherited by the same perspective.
    for doc_id, resolution in old.resolutions.items():
        if doc_id not in new.documents:
            dropped.append({
                "id": doc_id,
                "perspective": None,
                "action": resolution.action,
                "reason": "document_absent",
            })
            continue
        def cited_reference_is_available(ref: dict) -> bool:
            if ref.get("status", REFERENCE_ACTIVE) != REFERENCE_ACTIVE:
                return True
            opinion_id = ref.get("opinion_id")
            if opinion_id:
                located = new.find_opinion(opinion_id)
                if located is None:
                    return False
                support_doc, _, decision = located
                if support_doc not in new.documents:
                    return False
                return decision.status == DECISION_ACTIVE or (
                    decision.status == DECISION_WITHDRAWN
                    and any(
                        batch.perspective == located[1]
                        and support_doc in batch.decisions
                        and batch.decisions[support_doc].opinion_id == opinion_id
                        for batch in new.withdrawals.values()
                    )
                )
            decision = new.perspectives.get(ref.get("perspective"), Perspective(ref.get("perspective"))).decisions.get(doc_id)
            return decision is not None
        still_cited = all(cited_reference_is_available(ref) for ref in resolution.references)
        if not still_cited:
            dropped.append({
                "id": doc_id,
                "perspective": None,
                "action": resolution.action,
                "reason": "evidence_basis_changed",
            })
            continue
        cited_basis_changed = False
        for ref in resolution.references:
            opinion_id = ref.get("opinion_id")
            if not opinion_id:
                continue
            old_located = old.find_opinion(opinion_id)
            new_located = new.find_opinion(opinion_id)
            if old_located is None or new_located is None:
                cited_basis_changed = True
                break
            source_doc, _, old_decision = old_located
            new_doc, _, new_decision = new_located
            if source_doc != new_doc or not _decision_basis_still_matches(old, new, source_doc, old_decision):
                cited_basis_changed = True
                break
        if cited_basis_changed or not _decision_basis_still_matches(old, new, doc_id, resolution):
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
        inherited_resolution = replace(
            resolution,
            basis=basis or resolution.basis,
            rules_version=new.rules.version,
            inherited=True,
            references=list(resolution.references),
        )
        new.resolutions[doc_id] = inherited_resolution
        new.resolution_history[doc_id] = [
            replace(history, inherited=True) for history in old.resolution_history.get(doc_id, [])
        ]
        if inherited_resolution.status == DECISION_ACTIVE:
            inherited.append({"id": doc_id, "perspective": None, "action": resolution.action})
        else:
            inherited.append({
                "id": doc_id,
                "perspective": None,
                "action": resolution.action,
                "recheck": True,
            })
    return inherited, dropped
