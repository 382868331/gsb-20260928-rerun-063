import tempfile
import unittest
from pathlib import Path

from local_search.engine import SearchEngine
from local_search.ranking import ConstantScoreRanker
from local_search.research import (
    CONSENSUS_EXCLUDED,
    CONSENSUS_INCLUDED,
    MERGE_CONFLICT,
    MERGE_RESOLVED,
    MERGE_UNREVIEWED,
)
from local_search.synonyms import SynonymRules


DOCUMENTS = [
    {"id": "d1", "fields": {"title": "new york news", "body": "the city opens a new park"}},
    {"id": "d2", "fields": {"title": "york and new", "body": "different adjacent word order"}},
    {"id": "d3", "fields": {"title": "other", "body": "york appears after a new gap"}},
    {"id": "d4", "fields": {"title": "fresh city metro", "body": "the subway is fresh"}},
    {"id": "d5", "fields": {"title": "bank river", "body": "money at the river bank"}},
    {"id": "d6", "fields": {"title": "new novel", "body": "a fresh novel appears"}},
]


class PerspectiveTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.engine = SearchEngine(self.root, ranker=ConstantScoreRanker())
        for document in DOCUMENTS:
            self.engine.upsert(document["id"], document["fields"])
        self.engine.commit()
        self.engine.set_synonym_rules(SynonymRules.create(1, [["new", "fresh"]]))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def create(self, name="study", budget=None):
        return self.engine.create_research_set(
            name, {"q1": "~new", "q2": "york"}, mode="any", budget=budget
        )

    def test_independent_read_positions_and_decisions(self) -> None:
        research = self.create()
        alice_first = research.read_page(2, reviewer="alice")
        # Bob reads in a different order: he takes a later page first.
        bob_page = research.page(3, 2, reviewer="bob")
        for result in bob_page["results"]:
            research.reviewer_state("bob").read_ids.append(result["id"])

        alice_read = research.reviewer_state("alice").read_ids
        bob_read = research.reviewer_state("bob").read_ids
        self.assertEqual(len(alice_read), 2)
        self.assertEqual(set(bob_read), {r["id"] for r in bob_page["results"]})
        self.assertEqual(set(alice_read) & set(bob_read), set())

        # Alice's decision must not change Bob's unreviewed state.
        bob_unreviewed_before = research.unreviewed("bob")
        research.decide(alice_read[0], "included", "alice: relevant", reviewer="alice")
        self.assertEqual(research.unreviewed("bob"), bob_unreviewed_before)
        self.assertNotIn(alice_read[0], research.unreviewed("alice")[:0])
        # Bob has no decision for Alice's document.
        self.assertNotIn(alice_read[0], research.reviewer_state("bob").decisions)

        # Persist and reload: both perspectives survive independently.
        research.save()
        reopened = self.engine.open_research_set("study")
        self.assertEqual(reopened.reviewer_state("alice").read_ids, alice_read)
        self.assertEqual(reopened.reviewer_state("bob").read_ids, bob_read)
        self.assertEqual(
            reopened.reviewer_state("alice").decisions[alice_read[0]].reason,
            "alice: relevant",
        )
        self.assertEqual(reopened.unreviewed("bob"), bob_unreviewed_before)

    def test_merge_distinguishes_consensus_unreviewed_and_conflict(self) -> None:
        research = self.create()
        research.decide("d1", "included", "alice: on topic", reviewer="alice")
        research.decide("d1", "included", "bob: agreed", reviewer="bob")
        research.decide("d2", "excluded", "alice: off topic", reviewer="alice")
        research.decide("d2", "excluded", "bob: agreed off topic", reviewer="bob")
        research.decide("d3", "included", "alice: keep", reviewer="alice")
        research.decide("d3", "excluded", "bob: drop", reviewer="bob")

        merged = research.merge_review()
        self.assertEqual(merged["documents"]["d1"]["status"], CONSENSUS_INCLUDED)
        self.assertEqual(merged["documents"]["d2"]["status"], CONSENSUS_EXCLUDED)
        self.assertEqual(merged["documents"]["d3"]["status"], MERGE_CONFLICT)
        self.assertEqual(merged["conflicts"], ["d3"])
        self.assertIn("d4", merged["unreviewed"])
        self.assertEqual(merged["documents"]["d4"]["status"], MERGE_UNREVIEWED)
        # Conflict keeps both original judgements.
        self.assertEqual(
            merged["documents"]["d3"]["decisions"],
            {
                "alice": {"action": "included", "reason": "alice: keep"},
                "bob": {"action": "excluded", "reason": "bob: drop"},
            },
        )

    def test_conflict_is_not_overwritten_by_last_write(self) -> None:
        research = self.create()
        research.decide("d3", "included", "alice: keep", reviewer="alice")
        research.decide("d3", "excluded", "bob: drop", reviewer="bob")
        # Bob re-deciding only replaces his own judgement; Alice's survives.
        research.decide("d3", "excluded", "bob: still drop", reviewer="bob")
        merged = research.merge_review()
        entry = merged["documents"]["d3"]
        self.assertEqual(entry["status"], MERGE_CONFLICT)
        self.assertEqual(entry["decisions"]["alice"]["action"], "included")
        self.assertEqual(entry["decisions"]["bob"]["reason"], "bob: still drop")

    def test_resolve_conflict_cites_original_decisions(self) -> None:
        research = self.create()
        research.decide("d3", "included", "alice: keep", reviewer="alice")
        research.decide("d3", "excluded", "bob: drop", reviewer="bob")
        resolution = research.resolve_conflict("d3", "included", "lead: final say")
        self.assertEqual(
            resolution.resolves,
            [
                {"reviewer": "alice", "action": "included", "reason": "alice: keep"},
                {"reviewer": "bob", "action": "excluded", "reason": "bob: drop"},
            ],
        )
        merged = research.merge_review()
        entry = merged["documents"]["d3"]
        self.assertEqual(entry["status"], MERGE_RESOLVED)
        self.assertEqual(entry["resolution"]["action"], "included")
        # Original conflicting decisions are still there.
        self.assertEqual(entry["decisions"]["bob"]["action"], "excluded")

        # Resolving a non-conflict is rejected.
        with self.assertRaises(ValueError):
            research.resolve_conflict("d1", "included", "no conflict here")

        research.save()
        reopened = self.engine.open_research_set("study")
        self.assertEqual(
            reopened.merge_review()["documents"]["d3"]["resolution"]["resolves"],
            resolution.resolves,
        )

    def test_derive_after_index_update(self) -> None:
        research = self.create()
        research.read_page(3, reviewer="alice")
        research.read_page(2, reviewer="bob")
        research.decide("d1", "included", "alice: relevant", reviewer="alice")
        research.decide("d2", "excluded", "bob: not relevant", reviewer="bob")
        research.decide("d3", "included", "alice: will change", reviewer="alice")
        research.decide("d4", "excluded", "bob: will be deleted", reviewer="bob")
        research.save()

        # Index update: d3 body changes, d4 is deleted, d7 is new.
        self.engine.upsert("d3", {"title": "other", "body": "completely rewritten body"})
        self.engine.delete("d4")
        self.engine.upsert("d7", {"title": "new arrival", "body": "new york again"})
        self.engine.commit()

        report = self.engine.derive_research_set("study", "study-v2")
        inherited = {(d["id"], d["reviewer"]) for d in report["inherited_decisions"]}
        dropped = {(d["id"], d["reviewer"], d["reason"]) for d in report["dropped_decisions"]}
        # Unchanged records keep their evidence-based judgements per reviewer.
        self.assertIn(("d1", "alice"), inherited)
        self.assertIn(("d2", "bob"), inherited)
        # Changed and deleted records are handled explicitly.
        self.assertIn(("d3", "alice", "evidence_basis_changed"), dropped)
        self.assertIn(("d4", "bob", "document_absent"), dropped)

        derived = self.engine.open_research_set("study-v2")
        self.assertEqual(
            derived.reviewer_state("alice").decisions["d1"].reason, "alice: relevant"
        )
        self.assertTrue(derived.reviewer_state("alice").decisions["d1"].inherited)
        self.assertNotIn("d3", derived.reviewer_state("alice").decisions)
        self.assertNotIn("d4", derived.reviewer_state("bob").decisions)
        # Read positions survive only for documents still present, no duplicates.
        for reviewer in ("alice", "bob"):
            read_ids = derived.reviewer_state(reviewer).read_ids
            self.assertEqual(len(read_ids), len(set(read_ids)))
            self.assertNotIn("d4", read_ids)
        # New document appears as a fresh candidate.
        self.assertIn("d7", derived.reading_order())

    def test_derive_then_merge_keeps_conflict_evidence(self) -> None:
        research = self.create()
        research.decide("d1", "included", "alice: keep", reviewer="alice")
        research.decide("d1", "excluded", "bob: drop", reviewer="bob")
        research.resolve_conflict("d1", "included", "lead: final")
        research.decide("d2", "included", "alice: ok", reviewer="alice")
        research.save()

        self.engine.upsert("d6", {"title": "new novel", "body": "edited but still new"})
        self.engine.commit()
        self.engine.derive_research_set("study", "study-v2")

        derived = self.engine.open_research_set("study-v2")
        merged = derived.merge_review()
        entry = merged["documents"]["d1"]
        self.assertEqual(entry["status"], MERGE_RESOLVED)
        # The resolution and both original conflicting judgements survived.
        self.assertEqual(entry["resolution"]["action"], "included")
        self.assertEqual(
            {d["reviewer"] for d in entry["resolution"]["resolves"]},
            {"alice", "bob"},
        )
        self.assertEqual(entry["decisions"]["alice"]["action"], "included")
        self.assertEqual(entry["decisions"]["bob"]["action"], "excluded")
        self.assertEqual(merged["documents"]["d2"]["status"], CONSENSUS_INCLUDED)

    def test_tie_ranking_and_budget_continuation_across_reviewers(self) -> None:
        research = self.create(budget=2)
        self.assertFalse(research.exhausted)
        # Equal scores (constant ranker) break ties by document id.
        full_order = research.reading_order()
        self.assertEqual(full_order, sorted(full_order))

        # Alice reads page by page; Bob reads everything at once later.
        alice_ids = []
        while True:
            payload = research.read_page(1, reviewer="alice")
            alice_ids.extend(r["id"] for r in payload["results"])
            if payload["next_offset"] is None:
                break
        self.assertEqual(len(alice_ids), len(set(alice_ids)))

        research.continue_scan(None)
        self.assertTrue(research.exhausted)
        continued_order = research.reading_order()
        # Already-read prefix is frozen; continuation appends, never repeats.
        self.assertEqual(continued_order[: len(alice_ids)], alice_ids)
        self.assertEqual(len(continued_order), len(set(continued_order)))

        bob_ids = [r["id"] for r in research.read_page(None, reviewer="bob")["results"]]
        # Bob's one-shot exhaustive read matches Alice's paged order exactly.
        self.assertEqual(bob_ids, continued_order)
        # Alice's earlier paging did not consume any of Bob's positions.
        self.assertEqual(len(bob_ids), len(continued_order))


if __name__ == "__main__":
    unittest.main()
