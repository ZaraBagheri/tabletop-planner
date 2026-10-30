from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tabletop_planner.cli import main
from tabletop_planner.domain import Action
from tabletop_planner.tasks import make_episode


class DemoTests(unittest.TestCase):
    def run_demo(self, output, actions, *arguments):
        stdout = io.StringIO()
        argv = ["tabletop", "demo", "--backend", "symbolic", "--output", str(output), *arguments]
        with patch("sys.argv", argv), redirect_stdout(stdout), patch("tabletop_planner.planners.ModelPlanner") as model:
            model.return_value.name = "test_model"
            model.return_value.next_action.side_effect = actions
            main()
        return json.loads((output / "rollout.json").read_text()), stdout.getvalue(), model

    def test_custom_instruction_executes_without_grading_generated_goals(self):
        instruction = "Put the red block in the left tray. Then move the blue block to the right tray."
        objects = {obj["color"]: obj["id"] for obj in make_episode(0, "test", kind="double").objects}
        actions = [Action("pick", objects["red"]), Action("place", "left_tray"),
                   Action("pick", objects["blue"]), Action("place", "right_tray"), Action("done")]
        with tempfile.TemporaryDirectory() as folder:
            with patch("tabletop_planner.evaluation.grade", side_effect=AssertionError("Unrelated goals")):
                result, stdout, model = self.run_demo(Path(folder), actions, "--planner", "model",
                                                      "--adapter", "test_adapter", "--instruction", f"  {instruction}  ")
        model.assert_called_once_with("Qwen/Qwen2.5-0.5B-Instruct", "test_adapter", "auto", "main")
        self.assertEqual(model.return_value.next_action.call_args_list[0].args[0], instruction)
        self.assertEqual(result["instruction"], instruction)
        self.assertEqual(result["terminal"], "done")
        self.assertEqual(result["actions"], 5)
        self.assertFalse(result["graded"])
        self.assertIsNone(result["passed"])
        objects = {obj["color"]: obj for obj in result["final"]["objects"]}
        self.assertEqual(objects["red"]["zone"], "left_tray")
        self.assertEqual(objects["blue"]["zone"], "right_tray")
        self.assertIn("FINISHED (ungraded custom instruction)", stdout)
        self.assertNotIn("PASS", stdout)
        self.assertNotIn("FAIL", stdout)

    def test_generated_demo_keeps_success_grading(self):
        with tempfile.TemporaryDirectory() as folder:
            result, stdout, model = self.run_demo(Path(folder), [Action("done")], "--kind", "already_done")
        model.assert_called_once_with("Qwen/Qwen2.5-0.5B-Instruct", None, "auto", "main")
        self.assertTrue(result["graded"])
        self.assertTrue(result["passed"])
        self.assertIn("PASS", stdout)

    def test_explicit_custom_goals_enable_independent_grading(self):
        episode = make_episode(0, "test", kind="double")
        red = next(obj["id"] for obj in episode.objects if obj["color"] == "red")
        actions = [Action("pick", red), Action("place", "left_tray"), Action("done")]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"goals.json"
            path.write_text(json.dumps({"goals": [{"color": "red", "zone": "left_tray"}]}))
            result, stdout, _ = self.run_demo(Path(folder), actions, "--instruction", "Put the red block in the left tray.", "--goals", str(path))
        self.assertEqual(result["kind"], "custom")
        self.assertTrue(result["graded"])
        self.assertTrue(result["passed"])
        self.assertIn("PASS", stdout)

    def test_blank_instruction_is_rejected_before_loading_planner(self):
        argv = ["tabletop", "demo", "--planner", "model", "--instruction", "  "]
        stderr = io.StringIO()
        with patch("sys.argv", argv), patch("tabletop_planner.cli.planner_from") as planner:
            with redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                main()
        self.assertEqual(error.exception.code, 2)
        planner.assert_not_called()
        self.assertIn("--instruction must not be empty", stderr.getvalue())

    def test_removed_rules_option_is_rejected_before_loading_model(self):
        argv = ["tabletop", "demo", "--planner", "rules"]
        stderr = io.StringIO()
        with patch("sys.argv", argv), patch("tabletop_planner.cli.planner_from") as planner:
            with redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                main()
        self.assertEqual(error.exception.code, 2)
        planner.assert_not_called()
        self.assertIn("invalid choice", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
