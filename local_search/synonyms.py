from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

from .analysis import tokenize


@dataclass(frozen=True)
class SynonymRules:
    """An immutable, versioned set of synonym groups.

    A term may occur in several groups, which models ambiguous synonyms:
    expansion returns every possible alternative and matching reports the
    concrete synonym that was found in the document.
    """

    version: int
    groups: Tuple[Tuple[str, ...], ...] = ()
    _index: Dict[str, Tuple[Tuple[str, ...], ...]] = field(compare=False, repr=False, default_factory=dict)

    @classmethod
    def create(cls, version: int, groups: Iterable[Iterable[str]]) -> "SynonymRules":
        normalized: List[Tuple[str, ...]] = []
        lookup: Dict[str, List[Tuple[str, ...]]] = {}
        for group in groups:
            words: List[str] = []
            for entry in group:
                tokens = tokenize(entry)
                if len(tokens) != 1:
                    raise ValueError("Synonym entries must be single tokens")
                word = tokens[0]
                if word not in words:
                    words.append(word)
            if len(words) < 2:
                continue
            normalized.append(tuple(words))
            for word in words:
                lookup.setdefault(word, []).append(tuple(words))
        index = {word: tuple(alternatives) for word, alternatives in lookup.items()}
        return cls(version=version, groups=tuple(normalized), _index=index)

    def alternatives(self, term: str) -> Tuple[str, ...]:
        """All synonym alternatives for a term (the term itself excluded)."""
        result: List[str] = []
        for group in self._index.get(term, ()):
            for word in group:
                if word != term and word not in result:
                    result.append(word)
        return tuple(result)

    def candidates(self, term: str) -> Tuple[str, ...]:
        alternatives = self.alternatives(term)
        return (term, *alternatives)

    def to_payload(self) -> Dict[str, object]:
        return {"version": self.version, "groups": [list(group) for group in self.groups]}

    @classmethod
    def from_payload(cls, payload: Dict[str, object]) -> "SynonymRules":
        return cls.create(int(payload["version"]), payload.get("groups", []))  # type: ignore[arg-type]


class SynonymRulesBuilder:
    """Accumulates synonym groups and publishes a new immutable version."""

    def __init__(self, rules: SynonymRules | None = None) -> None:
        self._groups: List[List[str]] = [list(group) for group in (rules.groups if rules else ())]

    def add_group(self, words: Iterable[str]) -> "SynonymRulesBuilder":
        self._groups.append(list(words))
        return self

    def publish(self, version: int) -> SynonymRules:
        return SynonymRules.create(version, self._groups)


EMPTY_RULES = SynonymRules(version=0)
