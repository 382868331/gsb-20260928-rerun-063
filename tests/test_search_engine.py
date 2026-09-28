import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_search.engine import SearchEngine
from local_search.ranking import ConstantScoreRanker
from local_search.segment import write_segment
from tests.reference_scan import scan


DOCUMENTS = [
    {"id": "d1", "fields": {"title": "new york news", "body": "the city opens a new park"}},
    {"id": "d2", "fields": {"title": "york and new", "body": "different adjacent word order"}},
    {"id": "d3", "fields": {"title": "other", "body": "york appears after a new gap without adjacency"}},
    {"id": "d4", "fields": {"title": "repeat repeat repeat", "body": "repeat"}},
]


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
                expected = scan(DOCUMENTS, query)
                actual = [result["id"] for result in self.engine.search(query)]
                self.assertEqual(actual, expected)

    def test_phrase_uses_positions_within_one_field(self) -> None:
        self.assertEqual(["d1"], [item["id"] for item in self.engine.search('title:"new york"')])
        self.assertEqual([], [item["id"] for item in self.engine.search('body:"new york"')])
        self.assertEqual(["d4"], [item["id"] for item in self.engine.search('title:"repeat repeat"')])

    def test_field_conditions_are_scoped_per_field(self) -> None:
        self.assertEqual(["d1"], [item["id"] for item in self.engine.search("title:news body:park")])
        self.assertEqual([], [item["id"] for item in self.engine.search("title:park")])

    def test_batch_commit_updates_and_deletes(self) -> None:
        self.engine.upsert("d1", {"title": "updated story", "body": "changed content"})
        self.engine.delete("d2")
        self.assertEqual("new york news", self.engine.get("d1")["title"])
        self.assertIsNotNone(self.engine.get("d2"))

        self.engine.commit()

        self.assertEqual("updated story", self.engine.get("d1")["title"])
        self.assertIsNone(self.engine.get("d2"))
        self.assertEqual(["d3"], [item["id"] for item in self.engine.search("york")])
        self.assertEqual(["d1"], [item["id"] for item in self.engine.search("content")])

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


if __name__ == "__main__":
    unittest.main()
