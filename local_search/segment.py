import json
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional

from .analysis import field_tokens


@dataclass(frozen=True)
class StoredDocument:
    id: str
    fields: Dict[str, object]


@dataclass(frozen=True)
class Segment:
    segment_id: str
    path: str
    documents: Dict[str, StoredDocument]
    postings: Dict[str, Dict[str, Dict[str, List[int]]]]

    @classmethod
    def load(cls, segment_id: str, path: str) -> "Segment":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        documents = {
            doc_id: StoredDocument(doc_id, fields)
            for doc_id, fields in payload["documents"].items()
        }
        postings = payload["postings"]
        return cls(segment_id, path, documents, postings)


def build_segment(segment_id: str, documents: List[StoredDocument]) -> Dict[str, object]:
    docs = {}
    postings: Dict[str, Dict[str, Dict[str, List[int]]]] = {}
    for document in documents:
        docs[document.id] = document.fields
        for field, tokens in field_tokens(document.fields).items():
            field_postings = postings.setdefault(field, {})
            for position, term in enumerate(tokens):
                field_postings.setdefault(term, {}).setdefault(document.id, []).append(position)
    return {"id": segment_id, "documents": docs, "postings": postings}


def write_segment(directory: str, segment_id: str, payload: Dict[str, object]) -> str:
    os.makedirs(directory, exist_ok=True)
    final_path = os.path.join(directory, f"{segment_id}.json")
    fd, temporary_path = tempfile.mkstemp(prefix=f".{segment_id}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, final_path)
        _fsync_directory(directory)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise
    return final_path


def atomic_write_json(path: str, payload: Dict[str, object]) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(prefix=".manifest.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(directory)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def _fsync_directory(directory: str) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
