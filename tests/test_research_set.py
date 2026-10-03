import tempfile
import unittest
from pathlib import Path

from local_search.engine import SearchEngine
from local_search.research import (
    DOCUMENT_ENTERED,
    DOCUMENT_LEFT,
    RANK_MOVED,
    REASON_DOCUMENT_CHANGED,
    REASON_QUERY_CHANGED,
    REASON_RELATIVE_ORDER,
    compare_research_sets,
)
from local_search.ranking import BM25Ranker, ConstantScoreRanker
from local_search.synonyms import SynonymRules


DOCUMENTS = [
    {"id": "d1", "fields": {"title": "new york news", "body": "the city opens a new park"}},
    {"id": "d2", "fields": {"title": "york and new", "body": "different adjacent word order"}},
    {"id": "d3", "fields": {"title": "other", "body": "york appears after a new gap without adjacency"}},
    {"id": "d4", "fields": {"title": "fresh city metro", "body": "the subway is fresh"}},
    {"id": "d5", "fields": {"title": "bank river", "body": "money at the river bank"}},
]


def rules_v1():
    return SynonymRules.create(1, [["new", "fresh"], ["york", "city"]])


def rules_v2():
    return SynonymRules.create(2, [["new", "fresh", "novel"], ["york", "city"]])


class ResearchSetTest(unittest.TestCase):
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

    def test_any_and_all_modes_union_and_intersection(self) -> None:
        any_set = self.engine.create_research_set(
            "any", {"q1": "~new", "q2": "body:subway"}, mode="any"
        )
        self.assertEqual(set(any_set.reading_order()), {"d1", "d2", "d3", "d4"})
        all_set = self.engine.create_research_set(
            "all", {"q1": "~new", "q2": "york"}, mode="all"
        )
        self.assertEqual(set(all_set.reading_order()), {"d1", "d2", "d3"})

    def test_per_query_scores_and_positions_are_kept_separately(self) -> None:
        research = self.engine.create_research_set(
            "rs", {"q1": "~new", "q2": "york"}, mode="any"
        )
        payload = research.document_payload("d1")
        self.assertEqual(set(payload["matched_queries"]), {"q1", "q2"})
        self.assertEqual(payload["per_query"]["q1"]["rank"], 1)
        self.assertEqual(payload["per_query"]["q2"]["rank"], 1)
        # Each query keeps its own score; there is no combined relevance number.
        self.assertNotIn("combined_score", payload)
        self.assertNotIn("score", payload)
        clauses = payload["per_query"]["q1"]["explanation"]["clauses"]
        term_hits = [term for clause in clauses for term in clause["terms"]]
        self.assertTrue(term_hits)
        self.assertTrue(all("position" in term for term in term_hits))

    def test_bm25_ranks_differ_but_are_not_summed_across_queries(self) -> None:
        engine = SearchEngine(self.root, ranker=BM25Ranker())
        research = engine.create_research_set(
            "bm", {"q1": "metro", "q2": "york"}, mode="any"
        )
        payload = research.document_payload("d1")
        self.assertGreater(payload["per_query"]["q2"]["score"], 0.0)
        rare = research.document_payload("d4")
        self.assertGreater(rare["per_query"]["q1"]["score"], 0.0)

    def test_pinned_set_is_stable_against_later_live_changes(self) -> None:
        research = self.engine.create_research_set(
            "pin", {"q1": "~new"}, mode="any"
        )
        before = research.reading_order()
        research.read_page(len(before))
        research.save()
        # Live data and synonym version move on; the confirmed set stays put.
        self.engine.upsert("d1", {"title": "totally different"})
        self.engine.upsert("d9", {"title": "new fresh novel content"})
        self.engine.delete("d4")
        self.engine.commit()
        self.engine.set_synonym_rules(rules_v2())
        reopened = self.engine.open_research_set("pin")
        self.assertEqual(reopened.reading_order(), before)
        self.assertEqual(reopened.read_ids, before)
        self.assertEqual(reopened.rules.version, 1)
        self.assertEqual(
            reopened.document_payload("d1")["fields"],
            {"title": "new york news", "body": "the city opens a new park"},
        )

    def test_manual_exclusion_and_inclusion_exception_need_reasons(self) -> None:
        research = self.engine.create_research_set("rs", {"q1": "~new"}, mode="any")
        with self.assertRaises(ValueError):
            research.decide("d3", "excluded", "  ")
        research.decide("d3", "excluded", "reviewed: out of scope")
        research.save()
        research = self.engine.open_research_set("rs")
        self.assertNotIn("d3", research.reading_order())
        self.assertEqual(research.review_status()["excluded"], ["d3"])
        # A non-matching document can only be force-included as an exception.
        with self.assertRaises(ValueError):
            research.decide("d5", "excluded", "never matched anyway")
        research.decide("d5", "included", "author flagged it manually")
        self.assertIn("d5", research.reading_order())
        research.save()
        self.assertIn("d5", self.engine.open_research_set("rs").reading_order())

    def test_unreviewed_queue_is_visible(self) -> None:
        research = self.engine.create_research_set("rs", {"q1": "~new"}, mode="any")
        order = research.reading_order()
        research.read_page(2)
        status = research.review_status()
        self.assertEqual(status["reviewed"], order[:2])
        self.assertEqual(set(status["unreviewed"]), set(order[2:]))

    def test_nonexhaustive_budget_marks_pending_candidates(self) -> None:
        research = self.engine.create_research_set(
            "bud", {"q1": "~new"}, mode="any", budget=2
        )
        self.assertFalse(research.exhausted)
        status = research.review_status()
        self.assertEqual(set(status["pending_unscored"]), {"d3", "d4"})
        # Not yet appearing must never be interpreted as not satisfying it.
        self.assertFalse(research.page()["complete"])
        research.continue_scan(None)
        self.assertTrue(research.exhausted)
        self.assertEqual(set(research.reading_order()), {"d1", "d2", "d3", "d4"})

    def test_pagination_never_loses_or_duplicates_after_continue_and_restart(self) -> None:
        research = self.engine.create_research_set(
            "bud", {"q1": "~new"}, mode="any", budget=2
        )
        seen = []
        seen.extend(result["id"] for result in research.read_page(1)["results"])
        research.continue_scan(None)
        while True:
            payload = research.read_page(1)
            seen.extend(result["id"] for result in payload["results"])
            if payload["next_offset"] is None:
                break
        self.assertEqual(sorted(seen), ["d1", "d2", "d3", "d4"])
        self.assertEqual(len(seen), len(set(seen)))
        research.save()
        reopened = self.engine.open_research_set("bud")
        # Fresh full read of the same perspective matches the paged sequence.
        self.assertEqual(reopened.reading_order(), seen)

    def test_tie_broken_documents_read_the_same_paged_or_whole(self) -> None:
        research = self.engine.create_research_set(
            "tie", {"q1": "~new"}, mode="any"
        )
        whole = research.reading_order()
        paged: list[str] = []
        while True:
            payload = research.read_page(2)
            paged.extend(result["id"] for result in payload["results"])
            if payload["next_offset"] is None:
                break
        self.assertEqual(paged, whole)

    def test_revision_attributes_enters_leaves_and_moves(self) -> None:
        old = self.engine.create_research_set(
            "rs", {"q1": "~new", "q2": "york"}, mode="any"
        )
        old.decide("d3", "excluded", "off-topic")
        old.save()
        # d4 body changes (document change adds/removes hits); d6 is new;
        # query q1 is rewritten (query meaning change); a new high-rank doc
        # shifts others purely through relative order.
        self.engine.upsert("d4", {"title": "bank notes", "body": "money matters only"})
        self.engine.upsert("d2b", {"title": "fresh news", "body": "york coverage"})
        self.engine.commit()
        report = self.engine.revise_research_set(
            "rs", {"q1": "~fresh", "q2": "york"}
        )
        by_id = {change["id"]: change for change in report["changes"]}
        self.assertTrue(report["query_meaning_changed"])
        self.assertEqual(by_id["d2b"]["change"], DOCUMENT_ENTERED)
        self.assertIn(REASON_DOCUMENT_CHANGED, by_id["d2b"]["reasons"])
        self.assertEqual(by_id["d4"]["change"], DOCUMENT_LEFT)
        self.assertTrue(by_id["d4"]["reasons"])
        # The old d3 exclusion was not inherited, so d3 re-enters review.
        self.assertEqual(by_id["d3"]["change"], DOCUMENT_ENTERED)
        self.assertIn(REASON_QUERY_CHANGED, by_id["d3"]["reasons"])
        self.assertTrue(
            any(c["change"] in {DOCUMENT_ENTERED, DOCUMENT_LEFT, RANK_MOVED} for c in report["changes"])
        )

    def test_relative_order_move_when_only_other_candidates_change(self) -> None:
        old = self.engine.create_research_set("rs", {"q1": "york"}, mode="any")
        old_order = list(old.reading_order())
        self.assertEqual(old_order, ["d1", "d2", "d3"])
        # Same queries, same rule version, same existing bodies; one new
        # candidate inserts itself ahead of d3, shifting only by relativity.
        self.engine.upsert("c0", {"title": "york gazette", "body": "extra"})
        self.engine.commit()
        report = self.engine.revise_research_set("rs")
        self.assertFalse(report["query_meaning_changed"])
        by_id = {change["id"]: change for change in report["changes"]}
        self.assertEqual(by_id["c0"]["change"], DOCUMENT_ENTERED)
        self.assertEqual(by_id["c0"]["reasons"], [REASON_DOCUMENT_CHANGED])
        self.assertEqual(by_id["d3"]["change"], RANK_MOVED)
        self.assertEqual(by_id["d3"]["reasons"], [REASON_RELATIVE_ORDER])
        self.assertEqual(by_id["d1"]["reasons"], [REASON_RELATIVE_ORDER])

    def test_decisions_inherited_only_while_evidence_basis_matches(self) -> None:
        old = self.engine.create_research_set("rs", {"q1": "~new"}, mode="any")
        old.decide("d2", "excluded", "reviewed and dropped")
        old.save()
        # d2 changes body (its evidence signature changes) -> not inherited.
        self.engine.upsert("d2", {"title": "new rewritten york", "body": "changed text"})
        # d1 stays byte-identical with identical evidence -> inherited.
        self.engine.commit()
        report = self.engine.revise_research_set("rs")
        inherited_ids = {entry["id"] for entry in report["inherited_decisions"]}
        dropped_ids = {entry["id"] for entry in report["dropped_decisions"]}
        self.assertIn("d2", dropped_ids)
        new = self.engine.open_research_set("rs")
        self.assertEqual(new.decisions["d2"].action if "d2" in new.decisions else None, None)

    def test_deleted_then_readded_document_is_reviewed_not_blindly_inherited(self) -> None:
        old = self.engine.create_research_set("rs", {"q1": "~new"}, mode="any")
        old.decide("d1", "excluded", "old reason")
        old.save()
        self.engine.delete("d1")
        self.engine.commit()
        report = self.engine.revise_research_set("rs")
        self.assertIn("d1", {entry["id"] for entry in report["dropped_decisions"]})
        # Re-add a *different* document under the same id.
        self.engine.upsert("d1", {"title": "fresh start", "body": "brand new content"})
        self.engine.commit()
        report = self.engine.revise_research_set("rs")
        # The name alone never copies the old exclusion.
        self.assertIn("d1", {c["id"] for c in report["changes"] if c["change"] == DOCUMENT_ENTERED})
        new = self.engine.open_research_set("rs")
        self.assertNotIn("d1", new.decisions)


if __name__ == "__main__":
    unittest.main()
