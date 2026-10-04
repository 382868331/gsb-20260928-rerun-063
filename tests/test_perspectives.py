import tempfile
import unittest
from pathlib import Path

from local_search.engine import SearchEngine
from local_search.ranking import ConstantScoreRanker
from local_search.research import (
    DECISION_WITHDRAWN,
    MERGE_RECHECK,
    MERGE_CONFLICT,
    MERGE_CONSENSUS,
    MERGE_PARTIAL,
    MERGE_RESOLVED,
    MERGE_UNREVIEWED,
)
from local_search.synonyms import SynonymRules


DOCUMENTS = [
    {"id": "d1", "fields": {"title": "new york news", "body": "the city opens a new park"}},
    {"id": "d2", "fields": {"title": "york and new", "body": "different adjacent word order"}},
    {"id": "d3", "fields": {"title": "other", "body": "york appears after a new gap without adjacency"}},
    {"id": "d4", "fields": {"title": "fresh city metro", "body": "the subway is fresh"}},
    {"id": "d5", "fields": {"title": "bank river", "body": "money at the river bank"}},
]


class PerspectiveTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.engine = SearchEngine(self.root, ranker=ConstantScoreRanker())
        for document in DOCUMENTS:
            self.engine.upsert(document["id"], document["fields"])
        self.engine.commit()
        self.engine.set_synonym_rules(SynonymRules.create(1, [["new", "fresh"], ["york", "city"]]))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_set(self, name="rs", budget=None):
        return self.engine.create_research_set(name, {"q1": "~new"}, mode="any", budget=budget)

    def test_perspectives_keep_independent_positions_and_decisions(self) -> None:
        research = self.make_set()
        alice = research.perspective("alice")
        bob = research.perspective("bob")
        # Alice reads one document at a time; Bob reads two at a time.
        alice.read_page(1)
        bob.read_page(2)
        self.assertEqual(len(alice.review_status()["reviewed"]), 1)
        self.assertEqual(len(bob.review_status()["reviewed"]), 2)
        # Alice turning another page never changes Bob's unreviewed queue.
        bob_unreviewed_before = bob.unreviewed()
        alice.read_page(1)
        self.assertEqual(bob.unreviewed(), bob_unreviewed_before)
        # Opposite decisions on the same evidence coexist per perspective.
        alice.decide("d1", "included", "alice: directly relevant")
        bob.decide("d1", "excluded", "bob: out of scope")
        self.assertEqual(alice.document_payload("d1")["review"], "included")
        self.assertEqual(bob.document_payload("d1")["review"], "excluded")
        # The default perspective is untouched by both reviewers.
        self.assertEqual(research.review_status()["reviewed"], [])
        self.assertEqual(research.decisions, {})
        # State survives a save/reload cycle per perspective.
        research.save()
        reopened = self.engine.open_research_set("rs")
        self.assertEqual(
            reopened.review_status("alice")["reviewed"],
            research.review_status("alice")["reviewed"],
        )
        self.assertEqual(reopened.perspectives["bob"].decisions["d1"].action, "excluded")
        self.assertEqual(reopened.perspectives["alice"].decisions["d1"].action, "included")

    def test_merge_distinguishes_consensus_unreviewed_and_conflict(self) -> None:
        research = self.make_set()
        research.perspective("alice").decide("d1", "included", "alice: yes")
        research.perspective("bob").decide("d1", "included", "bob: yes too")
        research.perspective("alice").decide("d2", "included", "alice: relevant")
        research.perspective("bob").decide("d2", "excluded", "bob: not relevant")
        research.perspective("alice").decide("d3", "excluded", "alice: off-topic")
        merged = research.merge_perspectives()
        by_id = {entry["id"]: entry for entry in merged["documents"]}
        self.assertEqual(by_id["d1"]["status"], MERGE_CONSENSUS)
        self.assertEqual(by_id["d2"]["status"], MERGE_CONFLICT)
        self.assertEqual(by_id["d3"]["status"], MERGE_PARTIAL)
        self.assertEqual(by_id["d4"]["status"], MERGE_UNREVIEWED)
        self.assertEqual(merged["conflicts"], ["d2"])
        # The conflict keeps both original judgements; nothing is overwritten.
        self.assertEqual(
            by_id["d2"]["decisions"],
            {
                "alice": {"action": "included", "reason": "alice: relevant"},
                "bob": {"action": "excluded", "reason": "bob: not relevant"},
            },
        )

    def test_disagreement_is_not_settled_by_last_write(self) -> None:
        research = self.make_set()
        research.perspective("alice").decide("d1", "included", "alice: first")
        research.perspective("bob").decide("d1", "excluded", "bob: disagrees")
        # Alice re-affirming her own view must not erase Bob's judgement.
        research.perspective("alice").decide("d1", "included", "alice: still yes")
        merged = research.merge_perspectives()
        by_id = {entry["id"]: entry for entry in merged["documents"]}
        self.assertEqual(by_id["d1"]["status"], MERGE_CONFLICT)
        self.assertEqual(by_id["d1"]["decisions"]["bob"]["action"], "excluded")

    def test_resolution_creates_new_decision_referencing_originals(self) -> None:
        research = self.make_set()
        research.perspective("alice").decide("d1", "included", "alice: yes")
        research.perspective("bob").decide("d1", "excluded", "bob: no")
        # No disagreement -> resolution is refused.
        with self.assertRaises(ValueError):
            research.resolve("d2", "included", "nobody disagreed")
        resolution = research.resolve("d1", "excluded", "lead reviewer: out of scope")
        self.assertEqual(resolution.action, "excluded")
        self.assertEqual(
            resolution.references,
            [
                {"perspective": "alice", "action": "included", "reason": "alice: yes"},
                {"perspective": "bob", "action": "excluded", "reason": "bob: no"},
            ],
        )
        # Original per-perspective decisions are preserved as evidence.
        self.assertEqual(research.perspectives["alice"].decisions["d1"].action, "included")
        self.assertEqual(research.perspectives["bob"].decisions["d1"].action, "excluded")
        merged = research.merge_perspectives()
        by_id = {entry["id"]: entry for entry in merged["documents"]}
        self.assertEqual(by_id["d1"]["status"], MERGE_RESOLVED)
        self.assertEqual(by_id["d1"]["resolution"]["action"], "excluded")
        self.assertEqual(len(by_id["d1"]["resolution"]["references"]), 2)
        # Resolutions persist across reloads.
        research.save()
        reopened = self.engine.open_research_set("rs")
        self.assertEqual(reopened.resolutions["d1"].reason, "lead reviewer: out of scope")
        self.assertEqual(len(reopened.resolutions["d1"].references), 2)

    def test_batch_withdrawal_is_scoped_and_marks_cited_resolution_recheck(self) -> None:
        research = self.make_set()
        alice = research.perspective("alice")
        bob = research.perspective("bob")
        carol = research.perspective("carol")
        alice.decide("d1", "included", "alice: keep d1")
        bob.decide("d1", "included", "bob: keep d1")
        alice.decide("d2", "included", "alice: keep d2")
        bob.decide("d2", "excluded", "bob: drop d2")
        alice.decide("d3", "included", "alice: keep d3")
        carol.decide("d3", "excluded", "carol: drop d3")
        research.resolve("d2", "included", "lead: accepted alice")
        research.resolve("d3", "excluded", "lead: accepted carol")

        batch = alice.withdraw_decisions(["d1", "d2"], "alice: source was misread")
        self.assertEqual(sorted(batch.decisions), ["d1", "d2"])
        self.assertEqual(alice.document_payload("d1")["review"], DECISION_WITHDRAWN)
        self.assertEqual(alice.document_payload("d1")["review_reason"], "alice: keep d1")
        self.assertEqual(alice.document_payload("d1")["withdrawal"]["reason"], "alice: source was misread")
        # Other researchers' opinions and an unrelated lead decision stay active.
        self.assertEqual(bob.document_payload("d1")["review"], "included")
        self.assertEqual(research.perspectives["alice"].decisions["d3"].action, "included")
        self.assertEqual(research.resolutions["d3"].status, "active")
        merged = research.merge_perspectives()
        by_id = {entry["id"]: entry for entry in merged["documents"]}
        self.assertEqual(by_id["d2"]["status"], MERGE_RECHECK)
        self.assertEqual(by_id["d2"]["resolution"]["action"], "included")
        self.assertEqual(
            by_id["d2"]["resolution"]["references"][0]["reason"],
            "alice: keep d2",
        )
        self.assertEqual(by_id["d3"]["status"], "resolved")
        self.assertEqual(merged["recheck"], ["d2"])
        self.assertEqual(merged["resolution_history"]["d2"][0]["status"], "active")

        research.save()
        reopened = self.engine.open_research_set("rs")
        self.assertEqual(reopened.withdrawals[batch.batch_id].reason, "alice: source was misread")
        self.assertEqual(
            reopened.perspectives["alice"].decisions["d2"].status,
            DECISION_WITHDRAWN,
        )
        self.assertEqual(reopened.resolutions["d2"].status, "pending_recheck")
        self.assertEqual(reopened.resolution_history["d2"][0].reason, "lead: accepted alice")

    def test_withdrawal_batch_propagates_only_to_dependent_derived_conclusions(self) -> None:
        old = self.make_set()
        alice = old.perspective("alice")
        bob = old.perspective("bob")
        carol = old.perspective("carol")
        alice.decide("d1", "included", "alice: d1")
        bob.decide("d1", "included", "bob: d1")
        alice.decide("d2", "included", "alice: d2")
        bob.decide("d2", "excluded", "bob: d2")
        alice.decide("d3", "included", "alice: d3")
        carol.decide("d3", "excluded", "carol: d3")
        old.resolve("d2", "included", "lead: d2")
        old.resolve("d3", "excluded", "lead: d3")
        batch = alice.withdraw_decisions(["d1", "d2"], "alice: withdrawn")
        old.save()

        report = self.engine.revise_research_set("rs", new_name="derived")
        inherited = {(entry["id"], entry.get("recheck", False)) for entry in report["inherited_decisions"]}
        self.assertIn(("d2", True), inherited)
        self.assertIn(("d3", False), inherited)
        derived = self.engine.open_research_set("derived")
        self.assertIn(batch.batch_id, derived.withdrawals)
        self.assertEqual(
            derived.perspectives["alice"].decisions["d1"].status,
            DECISION_WITHDRAWN,
        )
        self.assertEqual(derived.perspectives["bob"].decisions["d1"].status, "active")
        self.assertEqual(derived.resolutions["d2"].status, "pending_recheck")
        self.assertEqual(derived.resolutions["d3"].status, "active")
        self.assertEqual(derived.resolution_history["d2"][0].references[0]["reason"], "alice: d2")

    def test_derive_after_index_update_then_merge(self) -> None:
        old = self.make_set()
        alice = old.perspective("alice")
        bob = old.perspective("bob")
        alice.read_page(2)
        bob.read_page(1)
        alice.decide("d1", "included", "alice: relevant")
        bob.decide("d1", "included", "bob: relevant")
        alice.decide("d2", "excluded", "alice: off-topic")
        bob.decide("d2", "included", "bob: keep it")
        old.resolve("d2", "excluded", "lead: agreed off-topic")
        old.save()
        # Index update: d2's body changes, d4 is deleted, d1 stays identical.
        self.engine.upsert("d2", {"title": "new rewritten york", "body": "changed text"})
        self.engine.delete("d4")
        self.engine.commit()
        report = self.engine.revise_research_set("rs", new_name="rs-v2")
        inherited = {(e["id"], e["perspective"]) for e in report["inherited_decisions"]}
        dropped = {(e["id"], e["reason"]) for e in report["dropped_decisions"]}
        # Unchanged evidence: both reviewers' d1 decisions are inherited.
        self.assertIn(("d1", "alice"), inherited)
        self.assertIn(("d1", "bob"), inherited)
        # Changed body: every d2 judgement (and its resolution) is explicit.
        self.assertIn(("d2", "evidence_basis_changed"), dropped)
        new = self.engine.open_research_set("rs-v2")
        self.assertNotIn("d2", new.perspectives["alice"].decisions)
        self.assertNotIn("d2", new.resolutions)
        # Read positions carry over for surviving documents, without gaps
        # or duplicates from the deleted d4.
        for name in ("alice", "bob"):
            read = new.perspectives[name].read_ids
            self.assertEqual(len(read), len(set(read)))
            self.assertNotIn("d4", read)
        # Merging the derived set still sees the inherited consensus and
        # treats the changed d2 as fresh, unreviewed work.
        merged = new.merge_perspectives()
        by_id = {entry["id"]: entry for entry in merged["documents"]}
        self.assertEqual(by_id["d1"]["status"], MERGE_CONSENSUS)
        self.assertEqual(by_id["d2"]["status"], MERGE_UNREVIEWED)

    def test_derive_drops_decisions_for_deleted_documents(self) -> None:
        old = self.make_set()
        old.perspective("alice").decide("d4", "excluded", "alice: not relevant")
        old.save()
        self.engine.delete("d4")
        self.engine.commit()
        report = self.engine.revise_research_set("rs")
        dropped = {(e["id"], e["perspective"], e["reason"]) for e in report["dropped_decisions"]}
        self.assertIn(("d4", "alice", "document_absent"), dropped)
        new = self.engine.open_research_set("rs")
        self.assertNotIn("d4", new.perspectives["alice"].decisions)

    def test_tie_break_and_budget_continuation_match_per_perspective(self) -> None:
        research = self.make_set("tie", budget=2)
        self.assertFalse(research.exhausted)
        alice = research.perspective("alice")
        bob = research.perspective("bob")
        alice.read_page(1)
        bob.read_page(2)
        research.continue_scan(None)
        self.assertTrue(research.exhausted)
        # Paged reading after continuation equals one fresh exhaustive read,
        # per perspective, with no duplicated read positions.
        for view in (alice, bob):
            seen = list(view.review_status()["reviewed"])
            while True:
                payload = view.read_page(2)
                seen.extend(result["id"] for result in payload["results"])
                if payload["next_offset"] is None:
                    break
            self.assertEqual(len(seen), len(set(seen)))
            self.assertEqual(seen, view.reading_order())
            self.assertEqual(sorted(seen), ["d1", "d2", "d3", "d4"])
        # Equal scores are broken by document id: both perspectives agree on
        # the same deterministic tail order.
        self.assertEqual(
            alice.reading_order()[len(alice.review_status()["reviewed"]):],
            bob.reading_order()[len(bob.review_status()["reviewed"]):],
        )


if __name__ == "__main__":
    unittest.main()
