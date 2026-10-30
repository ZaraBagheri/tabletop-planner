import json
from pathlib import Path
import tempfile
import unittest

from tabletop_planner.data import generate_dataset, messages_for
from tabletop_planner.domain import Action, grade
from tabletop_planner.evaluation import rollout
from tabletop_planner.symbolic import SymbolicBackend
from tabletop_planner.tasks import make_episode, PHRASES


class CoreTests(unittest.TestCase):
    def test_strict_action_parser(self):
        self.assertEqual(Action.parse('{"action":"pick","target":"block_0"}'), Action("pick", "block_0"))
        for text in ['[]', '{"action":"shell","target":"x"}', '{"action":"done","target":1}',
                     '{"action":"done","target":"","extra":0}', 'Here: {"action":"done","target":""}',
                     '{"action":"done","target":"x"}', '{"action":"pick","target":""}']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                Action.parse(text)

    def test_gripper_preconditions(self):
        env = SymbolicBackend(make_episode(0))
        self.assertEqual(env.execute(Action("place", "left_tray"))["error"], "GRIPPER_EMPTY")
        self.assertEqual(env.execute(Action("pick", "unknown"))["error"], "OBJECT_NOT_FOUND")
        self.assertTrue(env.execute(Action("pick", "block_0"))["ok"])
        self.assertEqual(env.execute(Action("pick", "block_1"))["error"], "GRIPPER_FULL")
        self.assertEqual(env.execute(Action("place", "moon"))["error"], "UNKNOWN_TRAY")

    def test_invalid_direct_action_is_rejected(self):
        env = SymbolicBackend(make_episode(0))
        self.assertEqual(env.execute(Action("teleport", "anything"))["error"], "INVALID_ACTION")

    def test_done_claim_does_not_pass_unfinished_task(self):
        ep = make_episode(0, kind="single")
        env = SymbolicBackend(ep)
        env.execute(Action("done"))
        self.assertFalse(grade(ep, env.observe(), "done"))

    def test_wrong_order_fails_even_with_correct_final_positions(self):
        ep = make_episode(0, kind="double")
        env = SymbolicBackend(ep)
        for goal in reversed(ep.goals):
            env.execute(Action("pick", goal["object_id"]))
            env.execute(Action("place", goal["zone"]))
        self.assertFalse(grade(ep, env.observe(), "done"))

    def test_ambiguity_requires_no_movement(self):
        ep = make_episode(0, kind="ambiguous")
        env = SymbolicBackend(ep)
        self.assertTrue(grade(ep, env.observe(), "ask"))
        env.execute(Action("pick", "block_0"))
        self.assertFalse(grade(ep, env.observe(), "ask"))

    def test_observation_is_not_a_mutable_reference(self):
        env = SymbolicBackend(make_episode(0))
        env.observe()["objects"][0]["zone"] = "left_tray"
        self.assertEqual(env.observe()["objects"][0]["zone"], "table")

    def test_prompt_does_not_contain_oracle_specification(self):
        ep = make_episode(0, kind="double")
        payload = json.loads(messages_for(ep.instruction, SymbolicBackend(ep).observe())[1]["content"])
        self.assertEqual(set(payload), {"instruction", "observation"})
        self.assertNotIn("goals", payload["observation"])
        self.assertNotIn("expected", payload["observation"])

    def test_train_val_holdout_combination_and_distinct_phrasings(self):
        self.assertFalse(set(PHRASES["train"]) & set(PHRASES["test"]))
        for split in ("train", "val"):
            for i in range(100):
                ep = make_episode(i, split)
                objects = {o["id"]: o for o in ep.objects}
                for goal in ep.goals:
                    self.assertNotEqual((objects[goal["object_id"]]["color"], goal["zone"]),
                                        ("yellow", "right_tray"))

    def test_dataset_is_reproducible_and_episode_disjoint(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder)/"a", Path(folder)/"b"
            ma = generate_dataset(a, 20, 10, 10)
            mb = generate_dataset(b, 20, 10, 10)
            self.assertEqual(ma, mb)
            sets = []
            for split in ("train", "val", "test"):
                rows = [json.loads(x) for x in (a/f"{split}.jsonl").read_text().splitlines()]
                sets.append({r["episode_id"] for r in rows})
                for row in rows:
                    Action.parse(row["messages"][-1]["content"])
            self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_invalid_model_output_is_not_replaced_by_expert(self):
        class Broken:
            last_text = "I delivered it!"
            def next_action(self, *args):
                raise ValueError("not JSON")
        ep = make_episode(0)
        r = rollout(ep, SymbolicBackend(ep), Broken())
        self.assertFalse(r["passed"])
        self.assertEqual(r["malformed_outputs"], 1)
        self.assertEqual(r["final"]["holding"], None)

    def test_action_budget(self):
        class NeverDone:
            def next_action(self, instruction, observation):
                return Action("place", "left_tray")
        ep = make_episode(0)
        r = rollout(ep, SymbolicBackend(ep), NeverDone(), max_actions=3)
        self.assertEqual(r["terminal"], "budget_exhausted")
        self.assertEqual(r["invalid_actions"], 3)


if __name__ == "__main__":
    unittest.main()
