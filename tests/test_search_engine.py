import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_search.engine import SearchEngine
from local_search.query import parse_query
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
    {
        "id": "d7",
        "fields": {
            "title": "alpha gap beta",
            "body": "new york times news today fresh city dispatch metro",
        },
    },
    {
        "id": "d8",
        "fields": {"title": "dup dup dup", "body": "dup"},
    },
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

    def test_near_phrases_match_direct_scan(self) -> None:
        queries = [
            '"york new"~2',
            '"new news"~1',
            '"new news"~2',
            '"fresh metro"~3',
            '~"fresh metro"~3',
            '~"new city"~3',
            '"new city"~3',
            'title:"alpha beta"~1',
            'title:"alpha beta"~0',
            '"dup dup dup"',
            '"dup dup"~1',
        ]
        for query in queries:
            with self.subTest(query=query):
                expected = scan(self._docs(), query, rules_v1())
                actual = [result["id"] for result in self.engine.search(query)]
                self.assertEqual(actual, expected, query)

    def test_ordered_near_phrase_does_not_match_reverse_order(self) -> None:
        # d2 title is "york and new": york -> new matches with one gap, but
        # Reverse order never matches, even with a large slop: restrict to
        # d2 by requiring another field unique to that document.
        self.assertEqual(
            ["d2"],
            [r["id"] for r in self.engine.search('body:"different adjacent word order" title:"york new"~1')],
        )
        self.assertEqual(
            [],
            [r["id"] for r in self.engine.search('body:"different adjacent word order" title:"new york"~5')],
        )

    def test_exclusion_clauses_match_direct_scan(self) -> None:
        queries = [
            "new -york",
            "new -body:park",
            '~new -"new york"',
            '~new -~"fresh city"',
            "-title:other",
            'body:river -body:bank',
        ]
        for query in queries:
            with self.subTest(query=query):
                expected = scan(self._docs(), query, rules_v1())
                actual = [result["id"] for result in self.engine.search(query)]
                self.assertEqual(actual, expected, query)

    def test_exclusion_is_listed_in_explanation(self) -> None:
        results = {result["id"]: result for result in self.engine.search("new -body:park")}
        self.assertIn("d2", results)
        self.assertNotIn("d1", results)
        self.assertIn("-body:park", results["d2"]["explanation"]["exclusions"])

    def test_repeated_terms_consume_distinct_positions_in_terms_clause(self) -> None:
        self.assertEqual(["d8"], [r["id"] for r in self.engine.search('"dup dup"~10')])
        self.assertEqual(["d8"], [r["id"] for r in self.engine.search('"dup dup dup"~10')])
        self.assertEqual([], [r["id"] for r in self.engine.search('"dup dup dup dup"~10')])
        # d8 title has three dups; the four-term near phrase cannot reuse one.
        self.assertEqual([], [r["id"] for r in self.engine.search('~title:"dup dup dup dup"~2')])
        results = {r["id"]: r for r in self.engine.search('title:"dup dup"')}
        positions = {
            (item["term"], tuple(item["positions"]))
            for item in results["d8"]["explanation"]["terms"]
        }
        self.assertIn(("dup", (0, 1)), positions)

    def test_explanation_positions_and_synonym_basis(self) -> None:
        results = {r["id"]: r for r in self.engine.search('~title:"york new"~5')}
        d2 = results["d2"]["explanation"]
        by_term = {item["term"]: item for item in d2["terms"]}
        self.assertEqual(by_term["new"]["kind"], "original")
        self.assertEqual(by_term["york"]["kind"], "original")
        # Expansion still reports the concrete synonym group as its basis.
        expanded = {r["id"]: r for r in self.engine.search('~body:"new"')}
        d5 = expanded["d5"]["explanation"]
        new_term = {item["term"]: item for item in d5["terms"]}["new"]
        self.assertEqual(new_term["kind"], "synonym")
        self.assertEqual(new_term["matched_as"], "fresh")
        self.assertEqual(new_term["synonym_group"], ["new", "fresh"])
        self.assertEqual(d5["synonym_version"], 1)

    def test_positions_and_explanation_match_scan(self) -> None:
        query = '~"new york"~3'
        reference = scan_explained(self._docs(), query, rules_v1())
        results = {r["id"]: r for r in self.engine.search(query)}
        self.assertEqual(set(results), set(reference))
        for doc_id, clauses in reference.items():
            engine_clauses = results[doc_id]["explanation"]["clauses"]
            for (_, _, ref_terms), engine_clause in zip(clauses, engine_clauses):
                actual = [
                    (term["term"], term["matched_as"], term["position"])
                    for term in engine_clause["terms"]
                ]
                self.assertEqual(actual, ref_terms)

    def test_clause_execution_order_does_not_change_results(self) -> None:
        import itertools

        from local_search.query import Query

        clauses = parse_query('~new york -body:park "news metro"~5').clauses
        seen = set()
        for permutation in itertools.permutations(clauses):
            ids = tuple(r["id"] for r in self.engine.search(Query(list(permutation))))
            seen.add(ids)
        self.assertEqual(seen, {tuple(scan(self._docs(), '~new york -body:park "news metro"~5', rules_v1()))})

    def test_exclusion_only_query_matches_scan(self) -> None:
        query = "-title:other -body:repeat"
        self.assertEqual(
            [r["id"] for r in self.engine.search(query)],
            scan(self._docs(), query, rules_v1()),
        )

    def test_near_and_exclusion_on_snapshot_match_scan(self) -> None:
        self.engine.publish_snapshot("s1")
        self.engine.set_synonym_rules(rules_v2())
        self.engine.upsert("d1", {"title": "changed beyond recognition", "body": "nothing"})
        self.engine.commit()
        for query in [
            '~"new city"~4',
            '"york new"~5',
            '~new -~"fresh city"',
            '~metro -title:metro',
            '~"dup dup"~1',
        ]:
            with self.subTest(query=query):
                expected = scan(self._docs(), query, rules_v1())
                actual = [r["id"] for r in self.engine.search_snapshot("s1", query)]
                self.assertEqual(actual, expected)

    def test_inflight_query_with_near_phrase_keeps_old_rules(self) -> None:
        in_flight = parse_query('~"new subway"~5').with_rules(rules_v1())
        self.engine.set_synonym_rules(rules_v2())
        # v1 has no new <-> subway link, so nothing matches; v2 does.
        self.assertEqual([], [r["id"] for r in self.engine.search(in_flight)])
        self.assertEqual(
            scan(self._docs(), '~"new subway"~5', rules_v2()),
            [r["id"] for r in self.engine.search('~"new subway"~5')],
        )

    def test_compaction_results_equal_direct_scan_for_all_new_queries(self) -> None:
        queries = [
            '"york new"~1',
            '"york new"~4',
            '~"fresh city"~2',
            '~new york -body:park',
            '~"metro subway"~5',
            'title:dup -body:dup',
            '~"dup dup dup"~1',
        ]
        self.engine.compact()
        recovered = SearchEngine(self.root, ranker=ConstantScoreRanker())
        for query in queries:
            with self.subTest(query=query):
                self.assertEqual(
                    [r["id"] for r in recovered.search(query)],
                    scan(self._docs(), query, rules_v1()),
                )

    def test_expanded_queries_match_direct_scan(self) -> None:
        for query in EXPANDED_QUERIES:
            with self.subTest(query=query):
                expected = scan(self._docs(), query, rules_v1())
                actual = [result["id"] for result in self.engine.search(query)]
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
            {r["id"] for r in restarted.search('~"fresh city"')},
            {r["id"] for r in self.engine.search('~"fresh city"')},
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
        self.assertEqual(
            {"d5", "d7"},
            {r["id"] for r in self.engine.search_snapshot("before-edit", "~metro")},
        )
        # Live index reflects the change and the new rules.
        self.assertIsNone(self.engine.get("d1"))
        # v2 keeps new <-> fresh, so the expansion still finds live docs.
        self.assertEqual(
            {"d2", "d3"},
            {r["id"] for r in self.engine.search("~fresh") if r["id"] != "d7"},
        )

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


if __name__ == "__main__":
    unittest.main()
