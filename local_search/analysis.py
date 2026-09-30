import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def tokenize(value: object) -> List[str]:
    return TOKEN_RE.findall(str(value).lower())


def field_tokens(fields: Dict[str, object]) -> Dict[str, List[str]]:
    indexed = {}
    for name, value in fields.items():
        if value is not None:
            indexed[name] = tokenize(value)
    return indexed


@dataclass(frozen=True)
class SynonymRules:
    """Immutable synonym table for one rules version.

    ``synonyms`` maps a canonical query term to alternative expansions; each
    expansion is a token sequence (single-token alternatives participate in
    strict phrase expansion, multi-token alternatives only in loose clauses).
    """

    version: int
    synonyms: Mapping[str, Tuple[Tuple[str, ...], ...]]

    @classmethod
    def empty(cls, version: int = 0) -> "SynonymRules":
        return cls(version, {})

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Sequence[str]], version: int = 0) -> "SynonymRules":
        normalized: Dict[str, Tuple[Tuple[str, ...], ...]] = {}
        for term, alternatives in (mapping or {}).items():
            key = " ".join(tokenize(term))
            if not key:
                continue
            expansions: List[Tuple[str, ...]] = []
            for alternative in alternatives:
                tokens = tuple(tokenize(alternative))
                if tokens and tokens not in expansions:
                    expansions.append(tokens)
            if expansions:
                normalized[key] = tuple(expansions)
        return cls(version, normalized)

    def to_mapping(self) -> Dict[str, List[str]]:
        return {term: [" ".join(words) for words in alternatives] for term, alternatives in self.synonyms.items()}

    def alternatives(self, term: str) -> Tuple[Tuple[str, ...], ...]:
        return self.synonyms.get(term, ())

    def single_token_alternatives(self, term: str) -> Tuple[str, ...]:
        return tuple(words[0] for words in self.alternatives(term) if len(words) == 1 and words[0] != term)

    def equivalents(self, term: str) -> frozenset[str]:
        """Tokens that can stand for ``term`` at one phrase position."""
        return frozenset((term, *self.single_token_alternatives(term)))

    def is_ambiguous(self, term: str) -> bool:
        """True when the term expands to more than one single-token sense."""
        return len(self.single_token_alternatives(term)) > 1
