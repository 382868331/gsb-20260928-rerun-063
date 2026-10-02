import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_search.engine import SearchEngine
from local_search.query import Clause, Query, parse_query
from local_search.ranking import BM25Ranker, ConstantScoreRanker
from local_search.segment import write_segment
from local_search.synonyms import SynonymRules
from tests.reference_scan import scan, scan_explained


DOCUMENTS = [
    {"id": "d1", "fields": {"title": "new york news", "body": "the city opens a new park"}},
    {"id": "d2", "fields": {"title": "york and new", "body": "different adjacent word order"}},
    {"id": "d3", "fields": {"title": "other", "body": "york appears after a new gap without adjacency"}},
    {"id": "d4", "fields": {"title": "repeat repeat repeat", "body": "repeat"}},
    {"id": "d5", "fields": {"title": "fresh city metro", "body": "the subway is fresh"}},
    {"id": "d6", "fields": {"title": "bank river", "body": "money at the river bank"}},
]


def rules_v1() -> SynonymRules:
    # Ambiguous: "bank" is both a river bank and a place for money.
    return SynonymRules.create(1, [["new", "fresh"], ["york", "city"], ["bank", "river", "money"]])


def rules_v2() -> SynonymRules:
    return SynonymRules.create(2, [["new", "fresh", "novel"], ["york", "city"], ["metro", "subway"]])


EXPANDED_QUERIES = [
    "~new",
    "~york",
    "~new york",
    '~"new york"',
    '~"fresh city"',
    '~"repeat repeat"',
    '~title:"fresh city"',
    '~body:"new park"',
    "~bank",
    '~"river bank"',
    "~metro",
    "~title:metro body:subway",
]

SLOP_AND_EXCLUSION_QUERIES = [
    '"york new"~2',
    '"york new"~1',
    'body:"new park"~3',
    'body:"new park"~1',
    '~"fresh park"~3',
    '~title:"fresh metro"~1',
    '"repeat repeat"~1',
    '~"repeat repeat"~2',
    'new -york',
    'new -"new york"',
    'new york -title:news',
    '~new -~fresh',
    '~"new york" -body:park',
    'title:new -body:park body:"new park"~10',
    '-"york new"~2 "york new"~3',
]


class SearchEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.engine = SearchEngine(self.root, ranker=ConstantScoreRanker())
        for document in DOCUMENTS:
            self.engine.upsert(document["id"], document["fields"])
        self.engine.commit()
        self.engine.set_synonym_rules(rules_v1())

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _docs(self):
        return [{"id": doc["id"], "fields": doc["fields"]} for doc in DOCUMENTS]

    def test_strict_queries_still_match_direct_scan(self) -> None:
        for query in [
            "new",
            "new york",
            "repeat",
            'title:"new york"',
            'body:"new york"',
            'title:"repeat repeat"',
            "title:news body:park",
            "title:park",
        ]:
            with self.subTest(query=query):
                self.assertEqual([r["id"] for r in self.engine.search(query)], scan(self._docs(), query))

    def test_expanded_queries_match_direct_scan(self) -> None:
        for query in EXPANDED_QUERIES:
            with self.subTest(query=query):
                expected = scan(self._docs(), query, rules_v1())
                actual = [result["id"] for result in self.engine.search(query)]
                self.assertEqual(actual, expected)

    def test_slop_and_exclusion_queries_match_direct_scan(self) -> None:
        for query in SLOP_AND_EXCLUSION_QUERIES:
            with self.subTest(query=query):
                expected = scan(self._docs(), query, rules_v1())
                actual = [result["id"] for result in self.engine.search(query)]
                self.assertEqual(actual, expected)

    def test_slop_explanations_match_scan_chains(self) -> None:
        for query in [
            '"york new"~2',
            'body:"new park"~3',
            '~"fresh park"~3',
        ]:
            with self.subTest(query=query):
                reference = scan_explained(self._docs(), query, rules_v1())
                results = {r["id"]: r for r in self.engine.search(query)}
                self.assertEqual(set(results), set(reference))
                for doc_id, field_hits in reference.items():
                    clauses = results[doc_id]["explanation"]["clauses"]
                    for clause_hit, (field, _raw, chain) in zip(clauses, field_hits):
                        self.assertEqual(clause_hit["matched_field"], field)
                        expected_pairs = [(term, matched) for term, matched, _ in chain]
                        actual_pairs = [(t["term"], t["matched_as"]) for t in clause_hit["terms"]]
                        self.assertEqual(actual_pairs, expected_pairs)
                        expected_positions = [position for _, _, position in chain]
                        self.assertEqual(clause_hit["positions"], expected_positions)

    def test_explanation_lists_excluded_clauses_and_positions(self) -> None:
        results = {r["id"]: r for r in self.engine.search('new york -title:news')}
        for result in results.values():
            self.assertEqual(result["explanation"]["excluded_clauses"], ["-title:news"])
            for clause in result["explanation"]["clauses"]:
                positions = [term["position"] for term in clause["terms"]]
                self.assertEqual(len(set(positions)), len(positions))

    def test_repeated_term_clause_uses_distinct_positions(self) -> None:
        results = {r["id"]: r for r in self.engine.search('title:repeat repeat')}
        terms = results["d4"]["explanation"]["clauses"][0]["terms"]
        positions = [term["position"] for term in terms]
        self.assertEqual(len(positions), len(set(positions)))

    def test_query_result_is_independent_of_clause_order(self) -> None:
        text = '~"new york" body:park -title:other "york new"~5'
        parsed = parse_query(text)
        seen = None
        for order in itertools.permutations(range(len(parsed.clauses))):
            reordered = Query([parsed.clauses[index] for index in order])
            results = self.engine.search(reordered)
            summary = sorted(
                (r["id"], r["explanation"]["clauses"], r["explanation"]["excluded_clauses"])
                for r in results
            )
            if seen is None:
                seen = summary
            else:
                self.assertEqual(summary, seen)

    def test_multi_segment_live_index_matches_scan_regardless_of_order(self) -> None:
        self.engine.upsert("d2", {"title": "york fresh gap new", "body": "rewritten d2"})
        self.engine.upsert("d7", {"title": "fresh city news", "body": "new gap gap york park"})
        self.engine.delete("d5")
        self.engine.commit()
        live_docs = self.engine.all_documents()
        for query in SLOP_AND_EXCLUSION_QUERIES + EXPANDED_QUERIES:
            with self.subTest(query=query):
                expected = scan(live_docs, query, rules_v1())
                parsed = parse_query(query)
                for permutation in itertools.permutations(parsed.clauses):
                    actual = [r["id"] for r in self.engine.search(Query(list(permutation)))]
                    self.assertEqual(actual, expected)

    def test_snapshot_matches_direct_scan_after_changes_and_compaction(self) -> None:
        self.engine.publish_snapshot("freeze")
        snapshot_docs = [
            {"id": doc_id, "fields": doc.fields}
            for doc_id, doc in self.engine.snapshot_view("freeze").documents.items()
        ]

        self.engine.upsert("d7", {"title": "fresh novel metro", "body": "wide gap york then new later"})
        self.engine.delete("d2")
        self.engine.commit()
        self.engine.set_synonym_rules(rules_v2())

        queries = SLOP_AND_EXCLUSION_QUERIES + EXPANDED_QUERIES
        for query in queries:
            with self.subTest(query=query):
                expected = scan(snapshot_docs, query, rules_v1())
                actual = [r["id"] for r in self.engine.search_snapshot("freeze", query)]
                self.assertEqual(actual, expected)

        # A compaction of the live index must not change snapshot answers.
        self.engine.compact()
        for query in queries:
            with self.subTest(query=query, compacted=True):
                expected = scan(snapshot_docs, query, rules_v1())
                actual = [r["id"] for r in self.engine.search_snapshot("freeze", query)]
                self.assertEqual(actual, expected)

    def test_explanation_reports_original_vs_synonym(self) -> None:
        reference = scan_explained(self._docs(), '~"new york"', rules_v1())
        results = {result["id"]: result for result in self.engine.search('~"new york"')}
        self.assertEqual(set(results), set(reference))
        d5 = results["d5"]["explanation"]
        term_kinds = {item["term"]: item["kind"] for item in d5["terms"]}
        self.assertEqual(term_kinds["new"], "synonym")
        self.assertEqual(term_kinds["york"], "synonym")
        matched_as = {item["term"]: item["matched_as"] for item in d5["terms"]}
        self.assertEqual(matched_as, {"new": "fresh", "york": "city"})
        self.assertEqual(d5["synonym_version"], 1)

        d1 = results["d1"]["explanation"]
        self.assertTrue(all(item["kind"] == "original" for item in d1["terms"]))

    def test_repeated_phrase_terms_do_not_collapse_positions(self) -> None:
        self.assertEqual(["d4"], [r["id"] for r in self.engine.search('~"repeat repeat"')])
        self.assertEqual([], [r["id"] for r in self.engine.search('~"repeat repeat repeat repeat"')])

    def test_phrase_never_spans_two_fields(self) -> None:
        # "new" (body start) followed by "park" would look adjacent if field
        # positions were concatenated; d1 must remain the only phrase hit.
        self.assertEqual(["d1"], [r["id"] for r in self.engine.search('body:"new park"')])
        self.assertEqual(["d1"], [r["id"] for r in self.engine.search('~body:"new park"')])
        self.assertEqual([], [r["id"] for r in self.engine.search('title:"metro subway"')])
        # Expansion must not bridge fields either.
        self.assertEqual([], [r["id"] for r in self.engine.search('~title:"metro subway"')])

    def test_ambiguous_synonym_reports_concrete_match(self) -> None:
        results = {result["id"]: result for result in self.engine.search('~"river bank"')}
        self.assertIn("d6", results)
        kinds = {item["term"]: item["kind"] for item in results["d6"]["explanation"]["terms"]}
        self.assertEqual(kinds, {"river": "original", "bank": "original"})
        results = {result["id"]: result for result in self.engine.search('~"money bank"')}
        self.assertIn("d6", results)
        matched = {item["term"]: item["matched_as"] for item in results["d6"]["explanation"]["terms"]}
        self.assertEqual(matched["bank"], "bank")

    def test_inflight_query_keeps_its_rule_version(self) -> None:
        self.engine.publish_snapshot("rule-v1-view")
        # v1 has no subway <-> metro link; only literal "subway" hits d5.
        in_flight = parse_query("~subway").with_rules(rules_v1())
        self.assertEqual(["d5"], [r["id"] for r in self.engine.search(in_flight)])
        self.engine.set_synonym_rules(rules_v2())
        # The same in-flight query object keeps complete v1 semantics;
        # a v2 query additionally expands subway into the title token metro.
        self.assertEqual(["d5"], [r["id"] for r in self.engine.search(in_flight)])
        self.assertEqual(["d5"], [r["id"] for r in self.engine.search("title:~subway")])
        # The v1 snapshot still answers with the complete v1 rule set.
        snapshot_results = self.engine.search_snapshot("rule-v1-view", "title:~subway")
        self.assertEqual([], [r["id"] for r in snapshot_results])
        snapshot_results = self.engine.search_snapshot("rule-v1-view", "~subway")
        self.assertEqual(["d5"], [r["id"] for r in snapshot_results])

    def test_modify_and_delete_compared_with_scan(self) -> None:
        self.engine.upsert("d1", {"title": "fresh news", "body": "novel content"})
        self.engine.delete("d5")
        self.engine.commit()
        live_docs = [
            {"id": "d1", "fields": {"title": "fresh news", "body": "novel content"}},
            *[doc for doc in DOCUMENTS if doc["id"] not in {"d1", "d5"}],
        ]
        for query in ["~new", "~york", '~"new york"', "new", "fresh"]:
            with self.subTest(query=query):
                self.assertEqual(
                    [r["id"] for r in self.engine.search(query)],
                    scan(live_docs, query, rules_v1()),
                )
        self.assertIsNone(self.engine.get("d5"))

    def test_ranking_changes_never_change_identities(self) -> None:
        query = "~new york"
        constant = [r["id"] for r in SearchEngine(self.root, ranker=ConstantScoreRanker()).search(query)]
        bm25 = [r["id"] for r in SearchEngine(self.root, ranker=BM25Ranker()).search(query)]
        self.assertEqual(set(constant), set(bm25))
        for result in SearchEngine(self.root, ranker=BM25Ranker()).search(query):
            explained_terms = result["explanation"]["terms"]
            self.assertTrue(explained_terms)
            for term in explained_terms:
                self.assertIn(term["kind"], {"original", "synonym"})

    def test_rules_persist_across_restart(self) -> None:
        restarted = SearchEngine(self.root)
        self.assertEqual(restarted.rules.version, 1)
        self.assertEqual(
            [r["id"] for r in restarted.search('~"fresh city"')],
            [r["id"] for r in self.engine.search('~"fresh city"')],
        )

    def test_snapshot_stays_queryable_after_live_changes(self) -> None:
        self.engine.publish_snapshot("before-edit")
        snapshot_ids_before = [r["id"] for r in self.engine.search_snapshot("before-edit", "~new")]

        self.engine.upsert("d5", {"title": "totally rewritten", "body": "no synonyms here"})
        self.engine.delete("d1")
        self.engine.commit()
        self.engine.set_synonym_rules(rules_v2())

        snapshot_results = self.engine.search_snapshot("before-edit", "~new")
        self.assertEqual([r["id"] for r in snapshot_results], snapshot_ids_before)
        self.assertIn("d1", {r["id"] for r in snapshot_results})
        self.assertEqual(["d5"], [r["id"] for r in self.engine.search_snapshot("before-edit", "~metro")])
        # Live index reflects the change and the new rules.
        self.assertIsNone(self.engine.get("d1"))
        # v2 keeps new <-> fresh, so the expansion still finds live docs.
        self.assertEqual(["d2", "d3"], [r["id"] for r in self.engine.search("~fresh")])

    def test_interrupted_compaction_keeps_published_snapshots(self) -> None:
        self.engine.publish_snapshot("pre-compact")
        expected_snapshot = [r["id"] for r in self.engine.search_snapshot("pre-compact", "~new york")]
        live_expected = [r["id"] for r in self.engine.search("~new york")]

        original_write_segment = write_segment

        def fail_segment(directory, segment_id, payload):
            raise RuntimeError("simulated compaction interruption")

        with self.assertRaises(RuntimeError):
            with patch("local_search.engine.write_segment", side_effect=fail_segment):
                self.engine.compact()

        recovered = SearchEngine(self.root, ranker=ConstantScoreRanker())
        self.assertEqual(
            {doc["id"] for doc in recovered.all_documents()},
            {doc["id"] for doc in DOCUMENTS},
        )
        self.assertEqual([r["id"] for r in recovered.search("~new york")], live_expected)
        self.assertEqual(
            [r["id"] for r in recovered.search_snapshot("pre-compact", "~new york")],
            expected_snapshot,
        )

        # After a successful compaction, live results and the old snapshot both work.
        recovered.compact()
        compacted = SearchEngine(self.root, ranker=ConstantScoreRanker())
        self.assertEqual([r["id"] for r in compacted.search("~new york")], live_expected)
        self.assertEqual(
            [r["id"] for r in compacted.search_snapshot("pre-compact", "~new york")],
            expected_snapshot,
        )

    def test_fixed_perspective_paging_is_stable_across_changes_restart_and_compact(self) -> None:
        for index in range(8):
            self.engine.upsert(
                f"tie{index:02d}",
                {"title": "word word", "body": f"word tie{index:02d}"},
            )
        self.engine.commit()
        expected_ids = [f"tie{index:02d}" for index in range(8)]

        first = self.engine.start_search('"word word"', page_size=3)
        self.assertEqual([item["id"] for item in first["results"]], expected_ids[:3])
        self.assertEqual(first["page"]["number"], 1)
        self.assertEqual(first["perspective"]["synonym_version"], 1)
        self.assertTrue(first["next_cursor"])
        self.assertFalse(first["exhausted"])

        # The page keeps the per-document segment version used by the snapshot.
        for item in first["results"]:
            self.assertIsNotNone(item["document_version"])
            self.assertEqual(item["explanation"]["synonym_version"], 1)
            positions = item["explanation"]["clauses"][0]["positions"]
            self.assertEqual(positions, [0, 1])

        self.engine.upsert("tie00", {"title": "rewritten", "body": "cannot change paged snapshot"})
        self.engine.upsert("tie99", {"title": "word word", "body": "newly inserted"})
        self.engine.delete("tie01")
        self.engine.commit()
        self.engine.set_synonym_rules(rules_v2())

        all_pages = [item["id"] for item in first["results"]]
        cursor = first["next_cursor"]
        while cursor is not None:
            page = self.engine.fetch_page(first["run_id"], cursor)
            self.assertEqual(page["perspective"]["snapshot"], first["perspective"]["snapshot"])
            self.assertEqual(page["perspective"]["synonym_version"], 1)
            all_pages.extend(item["id"] for item in page["results"])
            cursor = page["next_cursor"]
        self.assertEqual(all_pages, expected_ids)

        replayed = self.engine.start_search('"word word"', page_size=3, snapshot=first["perspective"]["snapshot"])
        replayed_ids = [item["id"] for item in replayed["results"]]
        cursor = replayed["next_cursor"]
        while cursor is not None:
            page = SearchEngine(self.root, ranker=ConstantScoreRanker()).fetch_page(replayed["run_id"], cursor)
            replayed_ids.extend(item["id"] for item in page["results"])
            cursor = page["next_cursor"]
        self.assertEqual(replayed_ids, expected_ids)

        self.engine.compact()
        compacted = SearchEngine(self.root, ranker=ConstantScoreRanker())
        compacted_ids = [item["id"] for item in compacted.search_snapshot(first["perspective"]["snapshot"], '"word word"')]
        self.assertEqual(compacted_ids, expected_ids)

    def test_budgeted_paging_marks_unexamined_results_and_keeps_confirmed_order(self) -> None:
        budget_dir = tempfile.TemporaryDirectory()
        self.addCleanup(budget_dir.cleanup)
        budget_engine = SearchEngine(Path(budget_dir.name), ranker=ConstantScoreRanker())
        for index in range(7):
            budget_engine.upsert(
                f"budget{index:02d}",
                {"title": "word word", "body": f"budget{index:02d}"},
            )
        budget_engine.upsert("d2", {"title": "word", "body": "extra"})
        budget_engine.commit()
        expected_ids = [f"budget{index:02d}" for index in range(7)] + ["d2"]
        expected_ids.sort()

        first = budget_engine.start_search("word", page_size=2, budget=2)
        self.assertEqual([item["id"] for item in first["results"]], expected_ids[:2])
        self.assertFalse(first["exhausted"])
        self.assertEqual(first["unexamined_candidates"], 6)
        self.assertEqual(len(first["results"]), 2)

        seen = [item["id"] for item in first["results"]]
        cursor = first["next_cursor"]
        pages = 1
        while cursor is not None:
            page = budget_engine.fetch_page(first["run_id"], cursor, budget=2)
            pages += 1
            seen.extend(item["id"] for item in page["results"])
            cursor = page["next_cursor"]
        self.assertEqual(seen, expected_ids)
        self.assertGreaterEqual(pages, 4)

    def test_rank_change_explanation_distinguishes_condition_score_and_relative_changes(self) -> None:
        ranking_dir = tempfile.TemporaryDirectory()
        self.addCleanup(ranking_dir.cleanup)
        ranking_engine = SearchEngine(Path(ranking_dir.name), ranker=BM25Ranker())
        initial = [
            ("filler", {"title": "unrelated", "body": "background text"}),
            ("c1", {"title": "cat", "body": "cat"}),
            ("c2", {"title": "cat cat", "body": "other"}),
            ("c3", {"title": "kitten", "body": "static"}),
        ]
        for doc_id, fields in initial:
            ranking_engine.upsert(doc_id, fields)
        ranking_engine.commit()
        ranking_engine.set_synonym_rules(SynonymRules.create(1, [["cat", "kitten"]]))
        old_run = ranking_engine.start_search("~cat", page_size=10)
        old_snapshot = old_run["perspective"]["snapshot"]

        # c1 keeps the same literal condition but changes BM25 score; c3 moves
        # from synonym to literal evidence; c4 enters and shifts a tie.
        ranking_engine.upsert("c1", {"title": "cat cat cat", "body": "cat cat"})
        ranking_engine.upsert("c3", {"title": "cat", "body": "now literal"})
        ranking_engine.upsert("c4", {"title": "cat", "body": "new entry"})
        ranking_engine.commit()
        ranking_engine.set_synonym_rules(SynonymRules.create(2, [["cat", "kitten"]]))
        new_run = ranking_engine.start_search("~cat", page_size=10)

        report = ranking_engine.explain_rank_changes(old_run["run_id"], new_run["run_id"])
        by_id = {item["id"]: item for item in report["changes"]}
        self.assertEqual(by_id["c1"]["reason"], "score_changed")
        self.assertEqual(by_id["c3"]["reason"], "match_condition_changed")
        self.assertEqual(by_id["c4"]["reason"], "match_condition_added")
        self.assertEqual(by_id["c1"]["old_evidence"]["snapshot"], old_snapshot)
        self.assertEqual(by_id["c1"]["new_evidence"]["synonym_version"], 2)
        self.assertIn("explanation", by_id["c3"]["old_evidence"])

    def test_rank_change_relative_position_when_another_document_enters(self) -> None:
        for index, doc_id in enumerate(["same1", "same2", "same3"]):
            self.engine.upsert(doc_id, {"title": "word", "body": f"same {index}"})
        self.engine.commit()
        old_run = self.engine.start_search("word", page_size=10)

        self.engine.upsert("same0", {"title": "word", "body": "sorts ahead"})
        self.engine.commit()
        new_run = self.engine.start_search("word", page_size=10)
        report = self.engine.explain_rank_changes(old_run["run_id"], new_run["run_id"])
        by_id = {item["id"]: item for item in report["changes"]}
        self.assertEqual(by_id["same0"]["reason"], "match_condition_added")
        self.assertEqual(by_id["same1"]["reason"], "relative_position_changed")
        self.assertEqual(by_id["same1"]["old_position"], 2)
        self.assertEqual(by_id["same1"]["new_position"], 3)
        self.assertEqual(by_id["same1"]["score_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
