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
    exclude: bool
    slop: int
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
    clauses = []
    position = 0
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            break

        exclude = False
        if text[position] == "-":
            exclude = True
            position += 1

        expand = False
        if position < len(text) and text[position] == "~":
            expand = True
            position += 1

        start = position
        raw_start = position
        field_start = position
        field = None
        while position < len(text) and (text[position].isalnum() or text[position] in "_"):
            position += 1
        if position < len(text) and text[position] == ":" and position > field_start:
            field = text[field_start:position]
            position += 1
            token_start = position
        else:
            position = start
            token_start = start

        if position < len(text) and text[position] == "~":
            expand = True
            position += 1
            token_start = position

        slop = 0
        if position < len(text) and text[position] == '"':
            phrase_text, position = _read_quoted(text, position)
            terms = tokenize(phrase_text)
            if not terms:
                raise ValueError("Empty phrase is not supported")
            # Optional distance limit: "word word"~3 (0 means contiguous).
            if position < len(text) and text[position] == "~":
                number_start = position + 1
                number_end = number_start
                while number_end < len(text) and text[number_end].isdigit():
                    number_end += 1
                if number_end == number_start:
                    raise ValueError("Phrase distance must be a non-negative integer")
                slop = int(text[number_start:number_end])
                position = number_end
            raw = ("-" if exclude else "") + text[raw_start:position]
            clauses.append(Clause(field, True, expand, exclude, slop, terms, raw))
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
            raw = ("-" if exclude else "") + text[raw_start:position]
            clauses.append(Clause(field, False, expand, exclude, 0, terms, raw))

    if not clauses:
        raise ValueError("Empty query")
    return Query(clauses)
