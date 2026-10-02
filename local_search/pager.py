"""Stable ordering, budgeted scanning and pagination bookkeeping.

A page of results always orders documents by
``(rank score, stable document identity)``.  The identity tie-break makes
equal-score documents deterministic forever, so a confirmed page can never
shuffle when later pages are read or when other documents enter the corpus.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .query import Query
from .ranking import Ranker
from .view import DocumentMatch, SearchView


@dataclass(frozen=True)
class RankedEntry:
    doc_id: str
    score: float


@dataclass
class RankingOutcome:
    """The fully resolved ranking of one query execution.

    ``exhausted`` is False only when a scan budget stopped evaluation before
    every candidate document had been considered.
    """

    entries: List[RankedEntry]
    matches: Dict[str, DocumentMatch]
    scores: Dict[str, Dict[str, object]]
    exhausted: bool
    considered: int
    budget: Optional[int]
    skipped_unscored: List[str] = field(default_factory=list)

    def index(self) -> Dict[str, int]:
        return {entry.doc_id: position for position, entry in enumerate(self.entries)}


def order_matches(
    matches: Dict[str, DocumentMatch],
    scores: Dict[str, Dict[str, object]],
) -> List[RankedEntry]:
    """Stable ordering: descending score, then ascending document id."""
    entries = [
        RankedEntry(doc_id, float(detail.get("score", 0.0)))
        for doc_id, detail in scores.items()
        if doc_id in matches
    ]
    entries.sort(key=lambda entry: (-entry.score, entry.doc_id))
    return entries


def run_ranked(
    view: SearchView,
    query: Query,
    ranker: Ranker,
    budget: Optional[int] = None,
) -> RankingOutcome:
    """Execute ``query`` against an immutable view with a scan budget.

    ``budget`` caps the number of candidate documents whose scoring evidence
    is fully evaluated.  When the cap is reached the remaining documents are
    left unconsidered and the outcome is marked non-exhaustive; continuation
    appends them without touching already confirmed ranks.
    """
    bound_query = query.with_rules(view.rules)
    if budget is None or budget < 0:
        matches = view.run(bound_query)
        scored = ranker.score(matches, bound_query)
        return RankingOutcome(order_matches(matches, scored), matches, scored, True, len(matches), None)

    # Budgeted evaluation: scoring considers only the first ``budget``
    # candidate documents in the stable id order.  Matching is independent
    # of candidate iteration order, so this never reshapes prior pages.
    all_matches = view.run(bound_query)
    candidate_ids = sorted(all_matches)
    considered_ids = set(candidate_ids[:budget])
    considered_matches = {doc_id: all_matches[doc_id] for doc_id in candidate_ids[:budget]}
    scored = ranker.score(considered_matches, bound_query)
    entries = order_matches(considered_matches, scored)
    skipped = candidate_ids[budget:]
    return RankingOutcome(
        entries,
        all_matches,
        scored,
        exhausted=not skipped,
        considered=len(considered_matches),
        budget=budget,
        skipped_unscored=skipped,
    )


def build_result(
    view: SearchView,
    query: Query,
    entry: RankedEntry,
    outcome: RankingOutcome,
) -> Dict[str, object]:
    document = outcome.matches[entry.doc_id].document
    detail = outcome.scores.get(entry.doc_id, {})
    bound_query = query.with_rules(view.rules)
    return {
        "id": document.id,
        "score": entry.score,
        "fields": dict(document.fields),
        "explanation": outcome.matches[entry.doc_id].explanation(
            entry.score,
            detail.get("term_contributions", {}),
            bound_query.rules.version if bound_query.rules is not None else None,
            SearchView.excluded_rawhs(bound_query),
        ),
    }


def page_payload(
    view: SearchView,
    query: Query,
    outcome: RankingOutcome,
    window: slice,
    *,
    ranker_name: str,
    page_size: Optional[int],
    next_offset: Optional[int],
    exhausted: bool,
) -> Dict[str, object]:
    entries = outcome.entries[window]
    return {
        "results": [build_result(view, query, entry, outcome) for entry in entries],
        "ranking": ranker_name,
        "synonym_version": query.with_rules(view.rules).rules.version,
        "page_size": page_size,
        "next_offset": next_offset,
        "exhausted": exhausted,
        "complete": exhausted,
        "considered": outcome.considered,
        "budget": outcome.budget,
    }
