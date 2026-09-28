from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Document:
    doc_id: str
    fields: dict[str, str]
    version: int

    def __post_init__(self) -> None:
        if not isinstance(self.doc_id, str) or not self.doc_id:
            raise ValueError("doc_id must be a non-empty string")
        if not isinstance(self.version, int) or self.version < 1:
            raise ValueError("version must be a positive integer")
        normalized: dict[str, str] = {}
        for name, value in self.fields.items():
            if not isinstance(name, str) or not name:
                raise ValueError("field names must be non-empty strings")
            if not isinstance(value, str):
                raise TypeError(f"field {name!r} must be a string")
            normalized[name] = value
        object.__setattr__(self, "fields", normalized)

    @classmethod
    def from_mapping(
        cls, doc_id: str, fields: Mapping[str, str], version: int
    ) -> "Document":
        return cls(doc_id, dict(fields), version)
