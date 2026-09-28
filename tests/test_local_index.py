from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from local_data_retrieval.analyzer import terms
from local_data_retrieval import (
    And,
    LocalIndex,
    Not,
    Or,
    PhraseQuery,
    TermQuery,
    parse_query,
)


DOCUMENTS = {
    "alpha": {
        "title": "local engine",
        "body": "build a local retrieval engine locally",
        "tag": "python",
    },
    "beta": {
        "title": "retrieval notes",
        "body": "engine local retrieval keeps word positions local",
        "tag": "python",
    },
    "gamma": {
        "title": "other notes",
        "body": "a local engine is not a retrieval engine here",
        "tag": "storage",
    },
    "repeated": {
        "title": "local local",
        "body": "local local engine",
        "tag": "local",
    },
}


def scan_term(field: str, value: str) -> set[str]:
    wanted = terms(value)
    if len(wanted) != 1:
        return set()
    wanted_term = wanted[0]
    return {
        doc_id
        for doc_id, fields in DOCUMENTS.items()
        if wanted_term in terms(fields.get(field, ""))
    }


def scan_phrase(field: str, phrase: str) -> set[str]:
    wanted = terms(phrase)
    if not wanted:
        return set()
    result = set()
    for doc_id, fields in DOCUMENTS.items():
        values = terms(fields.get(field, ""))
        if any(
            values[position:position + len(wanted)] == wanted
            for position in range(0, len(values) - len(wanted) + 1)
        ):
            result.add(doc_id)
    return result


class LocalIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_dir.name)
        self.index = LocalIndex(self.directory)
        with self.index.writer() as writer:
            for doc_id, fields in DOCUMENTS.items():
                writer.upsert_document(doc_id, fields)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def ids(self, query) -> set[str]:
        return {hit.doc_id for hit in self.index.search(query)}

    def test_term_and_fielded_phrase_match_direct_scan(self) -> None:
        term_matches = self.ids(TermQuery("body", "local"))
        self.assertEqual(term_matches, scan_term("body", "local"))

        phrase_matches = self.ids(PhraseQuery("body", "local retrieval"))
        self.assertEqual(phrase_matches, scan_phrase("body", "local retrieval"))
        self.assertEqual(phrase_matches, {"alpha", "beta"})

    def test_phrase_requires_contiguous_positions_in_one_field(self) -> None:
        self.assertEqual(
            self.ids(PhraseQuery("body", "retrieval engine")),
            scan_phrase("body", "retrieval engine"),
        )
        self.assertEqual(
            self.ids(PhraseQuery("body", "retrieval engine")),
            {"alpha", "gamma"},
        )
        self.assertEqual(
            self.ids(parse_query('title:local body:engine')),
            {"alpha", "repeated"},
        )
        self.assertEqual(self.ids(parse_query('title:"local engine"')), {"alpha"})
        self.assertEqual(self.ids(PhraseQuery("title", "engine local")), set())

    def test_repeated_terms_are_position_checked_without_field_mixing(self) -> None:
        self.assertEqual(self.ids(PhraseQuery("title", "local local")), {"repeated"})
        self.assertEqual(self.ids(PhraseQuery("body", "local local")), {"repeated"})
        self.assertEqual(
            self.ids(PhraseQuery("tag", "local local")),
            scan_phrase("tag", "local local"),
        )
        self.assertEqual(self.ids(PhraseQuery("tag", "local local")), set())

    def test_boolean_queries(self) -> None:
        query = And(
            TermQuery("tag", "python"),
            Or(TermQuery("title", "engine"), TermQuery("body", "positions")),
            Not(TermQuery("body", "build")),
        )
        self.assertEqual(self.ids(query), {"beta"})

        parsed = parse_query('tag:python AND (title:engine OR body:positions) NOT body:build')
        self.assertEqual(self.ids(parsed), {"beta"})

    def test_identity_remains_stable_when_ranking_changes(self) -> None:
        query = TermQuery("body", "engine")
        default_ids = [hit.doc_id for hit in self.index.search(query)]
        reversed_ids = [
            hit.doc_id
            for hit in self.index.search(
                query,
                ranker=lambda hit, segment: -ord(hit.doc_id[0]),
                reverse=True,
            )
        ]
        alphabetical_ids = [
            hit.doc_id
            for hit in self.index.search(
                query,
                ranker=lambda hit, segment: hit.doc_id,
                reverse=False,
            )
        ]
        self.assertNotEqual(default_ids, reversed_ids)
        self.assertEqual(sorted(default_ids), alphabetical_ids)

    def test_batched_add_update_delete_are_visible_only_after_commit(self) -> None:
        writer = self.index.writer()
        writer.upsert_document("delta", {"title": "new", "body": "delta arrives", "tag": "python"})
        writer.upsert_document(
            "alpha",
            {"title": "changed", "body": "no old words here", "tag": "python"},
        )
        writer.delete_document("beta")

        before_commit = self.ids(TermQuery("body", "delta"))
        self.assertEqual(before_commit, set())
        self.assertIsNotNone(self.index.get_document("beta"))

        writer.commit()
        writer.close()
        self.assertEqual(self.ids(TermQuery("body", "delta")), {"delta"})
        self.assertEqual(self.ids(TermQuery("body", "arrives")), {"delta"})
        self.assertIsNone(self.index.get_document("beta"))
        self.assertEqual(self.index.get_document("alpha").version, 2)
        self.assertEqual(self.ids(TermQuery("body", "build")), set())

    def test_rollback_leaves_previous_index_visible(self) -> None:
        with self.index.writer() as writer:
            writer.delete_document("alpha")
            writer.rollback()
        self.assertEqual(self.ids(TermQuery("body", "build")), {"alpha"})

    def test_failed_build_keeps_last_manifest_and_segment_usable(self) -> None:
        manifest_path = self.directory / "manifest.json"
        original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        segment_path = self.directory / "segment-000000000002.json"
        segment_path.mkdir()
        writer = self.index.writer()
        writer.upsert_document("delta", {"body": "valid"})
        with self.assertRaises(OSError):
            writer.commit()
        writer.close()

        self.assertEqual(
            json.loads(manifest_path.read_text(encoding="utf-8")),
            original_manifest,
        )
        self.assertEqual(self.ids(TermQuery("body", "positions")), {"beta"})

    def test_interrupted_segment_file_does_not_replace_active_index(self) -> None:
        next_generation = 2
        broken_segment = self.directory / f"segment-{next_generation:012d}.json.tmp"
        broken_segment.write_text("{not valid json", encoding="utf-8")

        reopened = LocalIndex(self.directory)
        self.assertEqual(reopened.segment.generation, 1)
        self.assertEqual(
            {hit.doc_id for hit in reopened.search(TermQuery("tag", "storage"))},
            {"gamma"},
        )

    def test_restart_sees_committed_updates_and_deletions(self) -> None:
        staged_writer = self.index.writer()
        staged_writer.upsert_document("delta", {"title": "delta", "body": "restart visible"})
        staged_writer.delete_document("gamma")
        reopened_before_commit = LocalIndex(self.directory)
        self.assertEqual(
            {hit.doc_id for hit in reopened_before_commit.search(TermQuery("body", "restart"))},
            set(),
        )
        staged_writer.commit()
        staged_writer.close()

        reopened_after_commit = LocalIndex(self.directory)
        self.assertEqual(
            {hit.doc_id for hit in reopened_after_commit.search(TermQuery("body", "restart"))},
            {"delta"},
        )
        self.assertIsNone(reopened_after_commit.get_document("gamma"))

    def test_phrase_query_repeated_word_frequency_is_independent_from_scoring(self) -> None:
        hits = self.index.search(TermQuery("body", "engine"))
        by_doc = {hit.doc_id: hit for hit in hits}
        gamma_token_count = Counter(terms(DOCUMENTS["gamma"]["body"]))["engine"]
        self.assertEqual(gamma_token_count, 2)
        self.assertGreater(by_doc["alpha"].score, by_doc["gamma"].score)


if __name__ == "__main__":
    unittest.main()
