from __future__ import annotations

import re
from dataclasses import dataclass


TOKEN_PATTERN = re.compile(r"[0-9]+|\w+", re.UNICODE)


@dataclass(frozen=True)
class Token:
    text: str
    position: int


def tokenize(text: str) -> list[Token]:
    if not isinstance(text, str):
        raise TypeError("field values must be strings")
    return [
        Token(match.group(0).lower(), position)
        for position, match in enumerate(TOKEN_PATTERN.finditer(text))
    ]


def terms(text: str) -> list[str]:
    return [token.text for token in tokenize(text)]
