from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .analyzer import terms


class Query(Protocol):
    def clause_key(self) -> tuple[str, str, tuple[str, ...]]: ...


@dataclass(frozen=True)
class TermQuery:
    field: str
    value: str

    def clause_key(self) -> tuple[str, str, tuple[str, ...]]:
        return ("term", self.field, tuple(terms(self.value)))


@dataclass(frozen=True)
class PhraseQuery:
    field: str
    phrase: str

    def clause_key(self) -> tuple[str, str, tuple[str, ...]]:
        return ("phrase", self.field, tuple(terms(self.phrase)))


@dataclass(frozen=True)
class And:
    children: tuple[Query, ...]

    def __init__(self, *children: Query) -> None:
        object.__setattr__(self, "children", children)

    def clause_key(self) -> tuple[str, str, tuple[str, ...]]:
        return ("and", "", tuple(child.clause_key()[2] for child in self.children))


@dataclass(frozen=True)
class Or:
    children: tuple[Query, ...]

    def __init__(self, *children: Query) -> None:
        object.__setattr__(self, "children", children)

    def clause_key(self) -> tuple[str, str, tuple[str, ...]]:
        return ("or", "", tuple(child.clause_key()[2] for child in self.children))


@dataclass(frozen=True)
class Not:
    child: Query

    def clause_key(self) -> tuple[str, str, tuple[str, ...]]:
        return ("not", "", self.child.clause_key()[2])


def parse_query(text: str) -> Query:
    tokens = _lex(text)
    parser = _Parser(tokens)
    query = parser.parse_expression()
    if parser.current is not None:
        raise ValueError(f"unexpected token {parser.current[1]!r}")
    return query


def _lex(text: str) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
        elif char == "(":
            result.append(("lparen", char))
            index += 1
        elif char == ")":
            result.append(("rparen", char))
            index += 1
        elif char == '"':
            end = text.find('"', index + 1)
            if end == -1:
                raise ValueError("unterminated quoted phrase")
            result.append(("word", text[index + 1:end]))
            index = end + 1
        else:
            start = index
            while index < len(text) and not text[index].isspace() and text[index] not in "()":
                if text[index] == '"':
                    end = text.find('"', index + 1)
                    if end == -1:
                        raise ValueError("unterminated quoted phrase")
                    index = end + 1
                else:
                    index += 1
            word = text[start:index]
            if word in {"AND", "OR", "NOT"}:
                result.append(("operator", word))
            else:
                result.append(("word", word))
    return result


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]):
        self.tokens = tokens
        self.position = 0

    @property
    def current(self) -> tuple[str, str] | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def parse_expression(self) -> Query:
        return self.parse_or()

    def parse_or(self) -> Query:
        left = self.parse_and()
        children = [left]
        while self.current == ("operator", "OR"):
            self.position += 1
            children.append(self.parse_and())
        return children[0] if len(children) == 1 else Or(*children)

    def parse_and(self) -> Query:
        left = self.parse_unary()
        children = [left]
        while True:
            if self.current is None or self.current == ("operator", "OR") or self.current[0] == "rparen":
                break
            if self.current == ("operator", "AND"):
                self.position += 1
            elif self.current == ("operator", "NOT") or self.current[0] in {"word", "lparen"}:
                pass
            else:
                raise ValueError(f"unexpected token {self.current[1]!r}")
            children.append(self.parse_unary())
        return children[0] if len(children) == 1 else And(*children)

    def parse_unary(self) -> Query:
        if self.current == ("operator", "NOT"):
            self.position += 1
            return Not(self.parse_unary())
        return self.parse_primary()

    def parse_primary(self) -> Query:
        token = self.current
        if token is None:
            raise ValueError("expected query term")
        kind, value = token
        if kind == "lparen":
            self.position += 1
            query = self.parse_expression()
            if self.current != ("rparen", ")"):
                raise ValueError("missing closing parenthesis")
            self.position += 1
            return query
        if kind != "word":
            raise ValueError(f"unexpected token {value!r}")
        self.position += 1
        if ":" not in value:
            raise ValueError("fielded queries require field:value")
        field, raw_value = value.split(":", 1)
        if not field:
            raise ValueError("missing field name")
        if len(raw_value) >= 2 and raw_value.startswith('"') and raw_value.endswith('"'):
            return PhraseQuery(field, raw_value[1:-1])
        if raw_value.startswith('"') or raw_value.endswith('"'):
            raise ValueError("unterminated quoted phrase")
        return TermQuery(field, raw_value)
