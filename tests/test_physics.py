import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from tabletop_planner.domain import Action
from tabletop_planner.tasks import make_episode
from tabletop_planner.evaluation import rollout


@unittest.skipUnless(importlib.util.find_spec("pybullet"), "Install .[sim] for physics tests")
class PhysicsTests(unittest.TestCase):
    """Replay fixed action fixtures to verify the controller independently of model quality."""

    def test_two_transfers_are_verified_from_body_poses(self):
        from tabletop_planner.bullet_backend import BulletBackend
        ep = make_episode(0, "test", kind="double")
        env = BulletBackend(ep)
        actions = []
        for goal in ep.goals:
            actions.extend([Action("pick", goal["object_id"]), Action("place", goal["zone"])])
        planner = Mock()
        planner.next_action.side_effect = actions + [Action("done")]
        try:
            result = rollout(ep, env, planner)
            self.assertTrue(result["passed"], result)
            for goal in ep.goals:
                self.assertEqual(env.objects[goal["object_id"]]["zone"], goal["zone"])
        finally:
            env.close()

    def test_missing_target_does_not_move_arm(self):
        from tabletop_planner.bullet_backend import BulletBackend
        ep = make_episode(2, "test", kind="missing")
        env = BulletBackend(ep)
        planner = Mock()
        planner.next_action.return_value = Action("fail", "The requested block is missing.")
        try:
            before = env._ee_position()
            result = rollout(ep, env, planner)
            self.assertTrue(result["passed"])
            self.assertEqual(before, env._ee_position())
        finally:
            env.close()

    def test_recording_accepts_tuple_camera_pixels(self):
        from tabletop_planner.bullet_backend import BulletBackend
        env = BulletBackend(make_episode(0, "test", kind="single"))
        rgba = (0, 0, 0, 255)*(640*480)
        try:
            with patch.object(env.p, "getCameraImage", return_value=(640, 480, rgba, (), ())):
                frame = env.image()
                self.assertEqual(frame.size, (640, 480))
                self.assertEqual(frame.getpixel((0, 0)), (0, 0, 0))
                env.frames = [frame]
                with tempfile.TemporaryDirectory() as folder:
                    env.save_media(folder)
                    self.assertTrue((Path(folder)/"replay.gif").exists())
                    self.assertTrue((Path(folder)/"final.png").exists())
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
