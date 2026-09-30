from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from .analysis import tokenize

if TYPE_CHECKING:
    from .synonyms import SynonymRules


@dataclass(frozen=True)
class Clause:
    field: Optional[str]
    phrase: bool
    expand: bool
    negated: bool
    slop: Optional[int]
    terms: List[str]
    raw: str


@dataclass(frozen=True)
class Query:
    clauses: List[Clause]
    rules: Optional["SynonymRules"] = None

    def with_rules(self, current: "SynonymRules") -> "Query":
        """Bind the complete synonym version used while running.

        A query that already carries its own rules (an in-flight query
        parsed against an older version) keeps that complete version; a
        freshly parsed query adopts the engine's current version.
        """
        return self if self.rules is not None else Query(self.clauses, current)


def _read_quoted(text: str, start: int) -> tuple[str, int]:
    end = start + 1
    escaped = False
    output = []
    while end < len(text):
        char = text[end]
        if escaped:
            output.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return "".join(output), end + 1
        else:
            output.append(char)
        end += 1
    raise ValueError("Unterminated quoted phrase")


def parse_query(text: str) -> Query:
    """Parse a mixed must/phrase/near/exclude query.

    Supported clause forms (combinable with field prefixes):
      word       term that must be present (clauses are AND-combined)
      "a b"      strictly contiguous ordered phrase
      "a b"~3    ordered phrase allowing at most 3 extra in-between tokens
      ~clause    allow synonym expansion (prefix or directly after field)
      -clause    exclude documents matching the clause
    """
    clauses = []
    position = 0
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            break

        start = position
        negated = False
        if text[position] == "-":
            negated = True
            position += 1
        leading_tilde = False
        if position < len(text) and text[position] == "~":
            leading_tilde = True
            position += 1

        field_start = position
        field = None
        while position < len(text) and (text[position].isalnum() or text[position] == "_"):
            position += 1
        if position < len(text) and text[position] == ":" and position > field_start:
            field = text[field_start:position]
            position += 1
            if position < len(text) and text[position] == "-":
                negated = True
                position += 1
            if position < len(text) and text[position] == "~":
                leading_tilde = True
                position += 1
        else:
            position = start + int(negated) + int(leading_tilde)

        expand = leading_tilde
        if position < len(text) and text[position] in ("~", "-"):
            if text[position] == "-":
                negated = True
            else:
                expand = True
            position += 1

        if position < len(text) and text[position] == '"':
            phrase_text, position = _read_quoted(text, position)
            terms = tokenize(phrase_text)
            if not terms:
                raise ValueError("Empty phrase is not supported")
            slop: Optional[int] = 0
            if position < len(text) and text[position] == "~":
                digits_start = position + 1
                digits_end = digits_start
                while digits_end < len(text) and text[digits_end].isdigit():
                    digits_end += 1
                if digits_end > digits_start:
                    slop = int(text[digits_start:digits_end])
                    position = digits_end
                    if position < len(text) and text[position] == "~":
                        expand = True
                        position += 1
                else:
                    expand = True
                    position += 1
            raw = text[start:position]
            clauses.append(Clause(field, True, expand, negated, slop, terms, raw))
        else:
            word_start = position
            while position < len(text) and not text[position].isspace():
                position += 1
            word = text[word_start:position]
            if not word:
                raise ValueError("Malformed query")
            terms = tokenize(word)
            if not terms:
                position += 1
                continue
            raw = text[start:position]
            clauses.append(Clause(field, False, expand, negated, None, terms, raw))

    if not clauses:
        raise ValueError("Empty query")
    return Query(clauses)
