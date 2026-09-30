from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .analysis import SynonymRules
from .query import Clause, Query


@dataclass(frozen=True)
class TermHit:
    term: str
    kind: str  # "original" | "synonym"
    matched_token: str
    positions: Tuple[int, ...]


@dataclass
class ClauseHit:
    raw: str
    field: Optional[str]
    matched_field: str
    phrase: bool
    expanded: bool
    terms: List[TermHit] = field(default_factory=list)


def match_query(
    fields: Dict[str, object],
    tokenized: Dict[str, List[str]],
    query: Query,
    rules: SynonymRules,
) -> Optional[List[ClauseHit]]:
    hits: List[ClauseHit] = []
    for clause in query.clauses:
        hit = _match_clause(tokenized, clause, rules, expanded=clause.expand)
        if hit is None:
            return None
        hits.append(hit)
    return hits


def _match_clause(
    tokenized: Dict[str, List[str]],
    clause: Clause,
    rules: SynonymRules,
    expanded: bool,
) -> Optional[ClauseHit]:
    field_names = [clause.field] if clause.field is not None else sorted(tokenized)
    for field_name in field_names:
        tokens = tokenized.get(field_name, [])
        if clause.phrase:
            hit = _match_phrase(tokens, clause.terms, expanded, rules)
        else:
            hit = _match_and(tokens, clause.terms, expanded, rules)
        if hit is not None:
            return ClauseHit(clause.raw, clause.field, field_name, clause.phrase, expanded, hit)
    return None


def _match_phrase(
    tokens: Sequence[str],
    terms: Sequence[str],
    expanded: bool,
    rules: SynonymRules,
) -> Optional[List[TermHit]]:
    options = [
        [TermHit(term, "original", term, ())] if not expanded
        else [TermHit(term, "original", term, ())]
        + [
            TermHit(term, "synonym", alt, ())
            for alt in rules.single_token_alternatives(term)
        ]
        for term in terms
    ]
    for start in range(len(tokens) - len(terms) + 1):
        chosen: List[TermHit] = []
        for offset, choices in enumerate(options):
            actual = tokens[start + offset]
            choice = next((item for item in choices if item.matched_token == actual), None)
            if choice is None:
                break
            chosen.append(TermHit(choice.term, choice.kind, actual, (start + offset,)))
        else:
            return chosen
    return None


def _match_and(
    tokens: Sequence[str],
    terms: Sequence[str],
    expanded: bool,
    rules: SynonymRules,
) -> Optional[List[TermHit]]:
    hits: List[TermHit] = []
    for term in dict.fromkeys(terms):
        hit = _match_single_term(tokens, term, expanded, rules)
        if hit is None:
            return None
        hits.append(hit)
    return hits


def _match_single_term(
    tokens: Sequence[str],
    term: str,
    expanded: bool,
    rules: SynonymRules,
) -> Optional[TermHit]:
    positions = tuple(index for index, token in enumerate(tokens) if token == term)
    if positions:
        return TermHit(term, "original", term, positions)
    if expanded:
        for alternative in rules.alternatives(term):
            alt_positions: List[int] = []
            if len(alternative) == 1:
                alt_positions = [index for index, token in enumerate(tokens) if token == alternative[0]]
            else:
                for start in range(len(tokens) - len(alternative) + 1):
                    if tuple(tokens[start : start + len(alternative)]) == alternative:
                        alt_positions.append(start)
            if alt_positions:
                return TermHit(term, "synonym", " ".join(alternative), tuple(alt_positions))
    return None
