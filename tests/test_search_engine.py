import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_search.engine import SearchEngine
from local_search.ranking import BM25Ranker, ConstantScoreRanker
from local_search.segment import write_segment
from tests.reference_scan import scan, scan_explained


DOCUMENTS = [
    {"id": "d1", "fields": {"title": "new york news", "body": "the city opens a new park"}},
    {"id": "d2", "fields": {"title": "york and new", "body": "different adjacent word order"}},
    {"id": "d3", "fields": {"title": "other", "body": "york appears after a new gap without adjacency"}},
    {"id": "d4", "fields": {"title": "repeat repeat repeat", "body": "repeat"}},
]

# york -> metropolis (single-token), new -> fresh/first (ambiguous),
# city -> municipal (single), park -> municipal garden (multi-token).
SYNONYMS = {
    "york": ["metropolis"],
    "new": ["fresh", "first"],
    "city": ["municipal"],
    "park": ["municipal garden"],
}


class SearchEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.engine = SearchEngine(self.root, ranker=ConstantScoreRanker())
        for document in DOCUMENTS:
            self.engine.upsert(document["id"], document["fields"])
        self.engine.commit()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def ids(self, query, *args, **kwargs):
        return [item["id"] for item in self.engine.search(query, *args, **kwargs)]

    def test_term_search_matches_direct_scan(self) -> None:
        for query in [
            "new",
            "york",
            "new york",
            "repeat",
            "park city",
            'title:"new york"',
            'body:"new york"',
            'title:"repeat repeat"',
            "title:news body:park",
            "title:park",
        ]:
            with self.subTest(query=query):
                self.assertEqual(self.ids(query), scan(DOCUMENTS, query))

    def test_phrase_uses_positions_within_one_field(self) -> None:
        self.assertEqual(["d1"], self.ids('title:"new york"'))
        self.assertEqual([], self.ids('body:"new york"'))
        self.assertEqual(["d4"], self.ids('title:"repeat repeat"'))

    def test_field_conditions_are_scoped_per_field(self) -> None:
        self.assertEqual(["d1"], self.ids("title:news body:park"))
        self.assertEqual([], self.ids("title:park"))

    def test_batch_commit_updates_and_deletes(self) -> None:
        self.engine.upsert("d1", {"title": "updated story", "body": "changed content"})
        self.engine.delete("d2")
        self.assertEqual("new york news", self.engine.get("d1")["title"])
        self.assertIsNotNone(self.engine.get("d2"))

        self.engine.commit()

        self.assertEqual("updated story", self.engine.get("d1")["title"])
        self.assertIsNone(self.engine.get("d2"))
        self.assertEqual(["d3"], self.ids("york"))
        self.assertEqual(["d1"], self.ids("content"))

    def test_interrupted_build_keeps_previous_index(self) -> None:
        original_write_segment = write_segment

        def fail_segment(directory, segment_id, payload):
            if segment_id.endswith("000002"):
                raise RuntimeError("simulated build interruption")
            return original_write_segment(directory, segment_id, payload)

        self.engine.upsert("d5", {"title": "uncommitted crash"})
        with self.assertRaises(RuntimeError):
            with patch("local_search.engine.write_segment", side_effect=fail_segment):
                self.engine.commit()

        recovered = SearchEngine(self.root, ranker=ConstantScoreRanker())
        self.assertIsNone(recovered.get("d5"))
        self.assertEqual({"d1", "d2", "d3", "d4"}, {doc["id"] for doc in recovered.all_documents()})

        self.engine.upsert("d5", {"title": "visible after restart commit"})
        self.engine.commit()
        restarted = SearchEngine(self.root)
        self.assertIsNotNone(restarted.get("d5"))
        self.assertEqual(["d5"], [item["id"] for item in restarted.search("visible restart")])

    # ---- new behavior ------------------------------------------------

    def test_strict_phrase_never_uses_synonyms_or_cross_field(self) -> None:
        self.engine.set_synonyms(SYNONYMS)
        self.engine.commit()
        self.engine.upsert("s1", {"title": "new metropolis news", "body": "elsewhere"})
        self.engine.upsert("s2", {"title": "fresh metropolis", "body": "text"})
        self.engine.upsert("cross", {"title": "new", "body": "york begins here"})
        self.engine.commit()
        # strict quoted phrase ignores synonyms: only the original words hit
        self.assertEqual(["d1"], self.ids('title:"new york"'))
        self.assertEqual([], self.ids('~title:"new york"'))  # prefix on strict form stays explicit
        # cross-field concatenation is never a phrase
        self.assertEqual(["d1"], self.ids('"new york"'))
        self.assertNotIn("cross", self.ids('"new york"'))

    def test_expanded_phrase_allows_single_token_synonyms_in_one_field(self) -> None:
        self.engine.set_synonyms(SYNONYMS)
        self.engine.commit()
        self.engine.upsert("s1", {"title": "new metropolis news", "body": "else"})
        self.engine.upsert("s2", {"title": "fresh metropolis today", "body": "else"})
        self.engine.upsert("cross", {"title": "new", "body": "metropolis starts"})
        self.engine.commit()
        self.assertEqual(["d1", "s1", "s2"], self.ids('title:~"new york"'))
        self.assertEqual(["d1", "s1", "s2"], self.ids('~"new york"'))
        self.assertNotIn("cross", self.ids('~"new york"'))

    def test_synonym_expansion_matches_direct_scan_and_explains(self) -> None:
        self.engine.set_synonyms(SYNONYMS)
        self.engine.commit()
        self.engine.upsert("s1", {"title": "fresh metropolis", "body": "municipal garden opens"})
        self.engine.upsert("s2", {"title": "first metropolis", "body": "unrelated text"})
        self.engine.commit()
        all_docs = DOCUMENTS + [
            {"id": "s1", "fields": {"title": "fresh metropolis", "body": "municipal garden opens"}},
            {"id": "s2", "fields": {"title": "first metropolis", "body": "unrelated text"}},
        ]

        # strict word does not expand
        self.assertEqual(self.ids("metropolis"), scan(all_docs, "metropolis"))
        # expanded loose clauses use synonyms incl. multi-token expansions
        for query in ["~york", "~new", "title:~fresh", '~"new york"']:
            with self.subTest(query=query):
                self.assertEqual(self.ids(query), scan(all_docs, query, SYNONYMS))
        # multi-token synonym "municipal garden" for park matches s1;
        # original "park" still matches d1; both terms must be in one field
        self.assertEqual(["d1", "s1"], self.ids("~city park"))
        self.assertEqual(["s1"], self.ids("municipal"))

        results = {item["id"]: item for item in self.engine.search("~york", explain=True)}
        s1 = results["s1"]["explanation"]["clauses"][0]["terms"][0]
        self.assertEqual("synonym", s1["match"])
        self.assertEqual("metropolis", s1["matched"])
        d1 = results["d1"]["explanation"]["clauses"][0]["terms"][0]
        self.assertEqual("original", d1["match"])
        self.assertEqual("york", d1["matched"])

        # ambiguous synonym expansion reports whichever sense actually hit
        explained = dict(scan_explained(all_docs, "~new", SYNONYMS))
        fresh_hit = explained["s1"][0].terms[0]
        first_hit = explained["s2"][0].terms[0]
        self.assertEqual(("synonym", "fresh"), (fresh_hit.kind, fresh_hit.matched_token))
        self.assertEqual(("synonym", "first"), (first_hit.kind, first_hit.matched_token))

    def test_repeated_terms_explained_positions(self) -> None:
        self.assertEqual(["d4"], self.ids("repeat"))
        result = self.engine.search('title:"repeat repeat"')[0]
        terms = result["explanation"]["clauses"][0]["terms"]
        self.assertEqual([0], terms[0]["positions"])
        self.assertEqual([1], terms[1]["positions"])

    def test_rules_version_pins_inflight_queries(self) -> None:
        before = self.engine.snapshot()
        self.assertEqual(before.rules_version, 0)
        self.engine.set_synonyms({"york": ["metropolis"]})
        self.engine.commit()
        after = self.engine.snapshot()
        self.assertEqual(after.rules_version, 1)
        self.engine.upsert("s1", {"title": "metropolis report", "body": ""})
        self.engine.commit()

        # new query uses new rules; old in-flight snapshot keeps version 0
        self.assertEqual(["d1", "d2", "d3", "s1"], self.ids("~york"))
        # before the rules existed expansion is a no-op (original word still works)
        self.assertEqual(["d1", "d2", "d3"], [item["id"] for item in before.search("~york")])
        # a true synonym-only term exists only in new rules and cannot match on the old snapshot
        self.assertEqual(["s1"], self.ids("~metropolis"))
        self.assertEqual([], [item["id"] for item in before.search("~metropolis")])
        self.assertEqual(0, before.search("york")[0]["explanation"]["rules_version"])

        restarted = SearchEngine(self.root, ranker=ConstantScoreRanker())
        self.assertEqual(1, restarted.snapshot().rules_version)
        self.assertEqual(0, restarted.snapshot(1).rules_version)

    def test_old_snapshots_still_queryable_after_updates_and_deletes(self) -> None:
        first = self.engine.snapshot()
        self.assertEqual(4, len(first.all_documents()))
        self.engine.upsert("d1", {"title": "overwritten", "body": "totally changed"})
        self.engine.delete("d2")
        self.engine.commit()

        current = self.engine.snapshot()
        self.assertIsNone(current.get("d2"))
        self.assertEqual("overwritten", current.get("d1")["title"])
        # d3 still contains york; d1 was overwritten and d2 deleted
        self.assertEqual(["d3"], [item["id"] for item in current.search("york")])

        old_docs = {item["id"]: item for item in first.all_documents()}
        self.assertIn("d2", old_docs)
        self.assertEqual("new york news", old_docs["d1"]["fields"]["title"])
        old_hits = [item["id"] for item in first.search("york")]
        self.assertEqual(["d1", "d2", "d3"], old_hits)
        self.assertEqual(first, self.engine.snapshot(1))
        self.assertIn(1, self.engine.snapshots())

    def test_ranking_changes_keep_document_identity(self) -> None:
        constant = [item["id"] for item in self.engine.search("new york repeat", explain=False)]
        bm25 = SearchEngine(self.root, ranker=BM25Ranker())
        scored = bm25.search("new york repeat")
        self.assertEqual(set(constant), {item["id"] for item in scored})
        for item in scored:
            detail = item["explanation"]["score"]
            self.assertAlmostEqual(
                detail["total"],
                sum(term["contribution"] for term in detail["terms"]),
                places=9,
            )
            kinds = {term["match"] for term in detail["terms"]}
            self.assertTrue(kinds <= {"original", "synonym"})
        # ties break by stable doc id, never by content address
        ids_constant = [item["id"] for item in self.engine.search("new")]
        self.assertEqual(sorted(ids_constant), ids_constant)

    def test_interrupted_compaction_keeps_published_snapshots(self) -> None:
        self.engine.set_synonyms(SYNONYMS)
        self.engine.commit()
        self.engine.upsert("d5", {"title": "fresh metropolis", "body": "park"})
        self.engine.delete("d2")
        self.engine.commit()
        pre_compact_seq = self.engine.sequence
        pre_snapshot = self.engine.snapshot(pre_compact_seq)
        pre_ids = {item["id"] for item in pre_snapshot.all_documents()}

        original = write_segment

        def fail(directory, segment_id, payload):
            raise RuntimeError("compaction interrupted")

        with self.assertRaises(RuntimeError):
            with patch("local_search.engine.write_segment", side_effect=fail):
                self.engine.compact()

        # manifest untouched; current and old snapshots still work
        self.assertEqual(pre_compact_seq, self.engine.sequence)
        self.assertEqual(pre_ids, {item["id"] for item in self.engine.all_documents()})
        self.assertEqual([item["id"] for item in pre_snapshot.search("york")], ["d1", "d3"])
        self.assertEqual(
            ["d1", "d3", "d5"],
            [item["id"] for item in self.engine.snapshot(pre_compact_seq).search("~york")],
        )

        restarted = SearchEngine(self.root, ranker=ConstantScoreRanker())
        self.assertEqual(pre_ids, {item["id"] for item in restarted.all_documents()})
        self.assertEqual(
            ["d1", "d2", "d3"],
            [item["id"] for item in restarted.snapshot(pre_compact_seq - 1).search("york")],
        )

        # a successful compaction keeps history queryable
        self.assertTrue(self.engine.compact())
        compacted = SearchEngine(self.root, ranker=ConstantScoreRanker())
        self.assertEqual(pre_ids, {item["id"] for item in compacted.all_documents()})
        self.assertEqual(
            ["d1", "d2", "d3"],
            [item["id"] for item in compacted.snapshot(1).search("york")],
        )
        self.assertEqual(
            ["d1", "d3", "d5"],
            [item["id"] for item in compacted.search("~york")],
        )


if __name__ == "__main__":
    unittest.main()
