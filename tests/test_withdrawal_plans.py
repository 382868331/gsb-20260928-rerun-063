import tempfile
import unittest
from pathlib import Path

from local_search.engine import SearchEngine
from local_search.ranking import ConstantScoreRanker
from local_search.research import (
    DECISION_ACTIVE,
    DECISION_WITHDRAWN,
    PLAN_APPLIED,
    PLAN_CANCELLED,
    PLAN_DRAFT,
    RESOLUTION_PENDING_RECHECK,
    Decision,
)
from local_search.synonyms import SynonymRules


DOCUMENTS = [
    {"id": "d1", "fields": {"title": "new york news", "body": "the city opens a new park"}},
    {"id": "d2", "fields": {"title": "york and new", "body": "different adjacent word order"}},
    {"id": "d3", "fields": {"title": "other", "body": "york appears after a new gap without adjacency"}},
    {"id": "d4", "fields": {"title": "fresh city metro", "body": "the subway is fresh"}},
    {"id": "d5", "fields": {"title": "bank river", "body": "money at the river bank"}},
]


class WithdrawalPlanTest(unittest.TestCase):
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

    def make_set(self, name="rs"):
        return self.engine.create_research_set(name, {"q1": "~new"}, mode="any")

    def disagree(self, research, doc_id):
        research.perspective("alice").decide(doc_id, "included", "alice: relevant")
        research.perspective("bob").decide(doc_id, "excluded", "bob: out of scope")
        return research.resolve(doc_id, "excluded", "lead: out of scope")

    def test_plan_preview_splits_recheck_and_supported_resolutions(self) -> None:
        research = self.make_set()
        research.perspective("alice").decide("d1", "included", "alice: relevant")
        # d1: a confirmed resolution whose only cited opinion is alice's.
        research.resolutions["d1"] = Decision(
            "excluded",
            "lead call",
            references=[{"perspective": "alice", "action": "included", "reason": "alice: relevant"}],
        )
        # d2: resolution keeps bob's independent support after alice withdraws.
        self.disagree(research, "d2")

        plan = research.perspective("alice").plan_withdrawal(
            ["d1", "d2"], "alice rechecked sources"
        )
        report = plan.report
        self.assertEqual(report["withdraw"], ["d1", "d2"])
        self.assertEqual([entry["id"] for entry in report["recheck_resolutions"]], ["d1"])
        self.assertEqual([entry["id"] for entry in report["supported_resolutions"]], ["d2"])
        supported = report["supported_resolutions"][0]
        self.assertEqual(supported["supported_by"], [{"perspective": "bob", "doc_id": "d2"}])
        self.assertEqual(report["conflicts"], [])

        research.apply_withdrawal_plan(plan.plan_id)
        self.assertEqual(research.resolutions["d1"].status, RESOLUTION_PENDING_RECHECK)
        self.assertEqual(research.resolutions["d2"].status, DECISION_ACTIVE)
        merge = research.merge_perspectives()
        self.assertIn("d1", merge["recheck"])
        self.assertIn("d2", merge["resolved"])

    def test_keep_constraint_shrinks_scope_and_reports_conflicts(self) -> None:
        research = self.make_set()
        research.perspective("alice").decide("d1", "included", "alice: relevant")
        research.resolutions["d1"] = Decision(
            "included",
            "lead: keep it",
            references=[{"perspective": "alice", "action": "included", "reason": "alice: relevant"}],
        )
        research.perspective("alice").decide("d2", "excluded", "alice: no")

        plan = research.perspective("alice").plan_withdrawal(
            ["d1", "d2"], "alice rechecked", keep=["d1"]
        )
        report = plan.report
        # Minimal feasible scope: d1's opinion is the sole support of the kept
        # resolution, so it is blocked; d2 can still be withdrawn.
        self.assertEqual(report["withdraw"], ["d2"])
        self.assertEqual([entry["id"] for entry in report["blocked"]], ["d1"])
        self.assertEqual(report["blocked"][0]["kept_resolution"], "d1")
        self.assertEqual(report["recheck_resolutions"], [])

        research.apply_withdrawal_plan(plan.plan_id)
        self.assertEqual(research.resolutions["d1"].status, DECISION_ACTIVE)
        self.assertEqual(
            research.perspectives["alice"].decisions["d1"].status, DECISION_ACTIVE
        )
        self.assertEqual(
            research.perspectives["alice"].decisions["d2"].status, DECISION_WITHDRAWN
        )

    def test_impossible_keep_reports_conflict_instead_of_overriding(self) -> None:
        research = self.make_set()
        research.perspective("alice").decide("d1", "included", "alice: relevant")
        research.resolutions["d1"] = Decision(
            "included",
            "lead: keep it",
            references=[{"perspective": "alice", "action": "included", "reason": "alice: relevant"}],
            status=RESOLUTION_PENDING_RECHECK,
        )
        self.assertEqual(research.resolutions["d1"].status, RESOLUTION_PENDING_RECHECK)
        research.perspective("alice").decide("d2", "excluded", "alice: no")

        plan = research.perspective("alice").plan_withdrawal(
            ["d2"], "alice rechecked", keep=["d1"]
        )
        conflicts = plan.report["conflicts"]
        self.assertEqual([entry["conflict"] for entry in conflicts], ["keep_target_not_confirmed"])
        # Applying aborts with the conflict evidence; nothing is overridden.
        with self.assertRaises(ValueError):
            research.apply_withdrawal_plan(plan.plan_id)
        self.assertEqual(research.resolutions["d1"].status, RESOLUTION_PENDING_RECHECK)
        self.assertEqual(
            research.perspectives["alice"].decisions["d2"].status, DECISION_ACTIVE
        )

    def test_references_do_not_delete_original_opinion(self) -> None:
        research = self.make_set()
        research.perspective("alice").decide("d1", "included", "alice: relevant")
        research.perspective("alice").decide("d2", "excluded", "alice: no")
        plan = research.perspective("alice").plan_withdrawal(
            ["d1"], "first batch", references=["d2"]
        )
        research.apply_withdrawal_plan(plan.plan_id)
        # The referenced opinion is untouched: a citation is not a withdrawal.
        self.assertEqual(
            research.perspectives["alice"].decisions["d2"].status, DECISION_ACTIVE
        )
        # A second batch citing the same already-withdrawn opinion changes nothing.
        research.perspective("alice").decide("d3", "excluded", "alice: no")
        plan2 = research.perspective("alice").plan_withdrawal(
            ["d3"], "second batch", references=["d1"]
        )
        research.apply_withdrawal_plan(plan2.plan_id)
        self.assertEqual(len(research.withdrawals), 2)
        self.assertEqual(
            research.perspectives["alice"].decisions["d1"].withdrawal_batches,
            [plan.batch_id],
        )

    def test_cross_document_dependency_is_not_missed(self) -> None:
        research = self.make_set()
        research.perspective("alice").decide("d1", "included", "alice: relevant")
        research.perspective("alice").decide("d2", "excluded", "alice: no")
        research.perspective("bob").decide("d2", "included", "bob: yes")
        research.resolve("d2", "excluded", "lead: no")
        # The d2 resolution depends only on alice's opinion on d1 (cross-document).
        resolution = research.resolutions["d2"]
        resolution.references = [
            {"perspective": "alice", "doc_id": "d1", "action": "included", "reason": "alice: relevant"},
        ]

        plan = research.perspective("alice").plan_withdrawal(["d1"], "alice out")
        recheck_ids = [entry["id"] for entry in plan.report["recheck_resolutions"]]
        self.assertEqual(recheck_ids, ["d2"])
        research.apply_withdrawal_plan(plan.plan_id)
        self.assertEqual(research.resolutions["d2"].status, RESOLUTION_PENDING_RECHECK)

    def test_reopen_apply_skips_stale_and_never_double_applies(self) -> None:
        research = self.make_set()
        research.perspective("alice").decide("d1", "included", "alice: relevant")
        research.perspective("alice").decide("d2", "excluded", "alice: no")
        plan = research.perspective("alice").plan_withdrawal(["d1", "d2"], "alice out")
        research.save()

        reopened = self.engine.open_research_set("rs")
        self.assertEqual(reopened.withdrawal_plans[plan.plan_id].status, PLAN_DRAFT)
        # d2's opinion is replaced after planning: it must not be mixed in.
        reopened.perspective("alice").decide("d2", "included", "alice: changed her mind")
        reopened.apply_withdrawal_plan(plan.plan_id)
        applied = reopened.withdrawal_plans[plan.plan_id]
        self.assertEqual(applied.status, PLAN_APPLIED)
        self.assertEqual(applied.applied["withdrawn"], ["d1"])
        self.assertEqual(
            applied.applied["skipped"], {"d2": "opinion_changed_since_plan"}
        )
        self.assertEqual(
            reopened.perspectives["alice"].decisions["d2"].status, DECISION_ACTIVE
        )
        with self.assertRaises(ValueError):
            reopened.apply_withdrawal_plan(plan.plan_id)

        reopened.save()
        again = self.engine.open_research_set("rs")
        self.assertEqual(again.withdrawal_plans[plan.plan_id].status, PLAN_APPLIED)
        with self.assertRaises(ValueError):
            again.apply_withdrawal_plan(plan.plan_id)

    def test_already_withdrawn_opinion_is_skipped_on_apply(self) -> None:
        research = self.make_set()
        research.perspective("alice").decide("d1", "included", "alice: relevant")
        research.perspective("alice").decide("d2", "excluded", "alice: no")
        plan = research.perspective("alice").plan_withdrawal(["d1", "d2"], "alice out")
        # An independent direct withdrawal beats the plan to d2.
        research.perspective("alice").withdraw_decisions(["d2"], "direct")
        research.apply_withdrawal_plan(plan.plan_id)
        applied = research.withdrawal_plans[plan.plan_id].applied
        self.assertEqual(applied["withdrawn"], ["d1"])
        self.assertEqual(applied["skipped"], {"d2": "already_withdrawn"})
        self.assertEqual(
            research.perspectives["alice"].decisions["d2"].withdrawal_batches,
            ["withdrawal-1"],
        )

    def test_cancelled_plan_cannot_apply_and_is_listed(self) -> None:
        research = self.make_set()
        research.perspective("alice").decide("d1", "included", "alice: relevant")
        plan = research.perspective("alice").plan_withdrawal(["d1"], "alice out")
        research.cancel_withdrawal_plan(plan.plan_id)
        self.assertEqual(plan.status, PLAN_CANCELLED)
        with self.assertRaises(ValueError):
            research.apply_withdrawal_plan(plan.plan_id)
        self.assertEqual(
            research.perspectives["alice"].decisions["d1"].status, DECISION_ACTIVE
        )
        research.save()
        reopened = self.engine.open_research_set("rs")
        payload = reopened.withdrawal_plan_payload(plan.plan_id)
        self.assertEqual(payload["status"], PLAN_CANCELLED)

    def test_plan_requires_active_opinions_and_reason(self) -> None:
        research = self.make_set()
        with self.assertRaises(KeyError):
            research.perspective("alice").plan_withdrawal(["d1"], "nothing decided")
        research.perspective("alice").decide("d1", "included", "alice: relevant")
        with self.assertRaises(ValueError):
            research.perspective("alice").plan_withdrawal(["d1"], "  ")
        with self.assertRaises(ValueError):
            research.perspective("alice").plan_withdrawal([], "empty")


if __name__ == "__main__":
    unittest.main()
