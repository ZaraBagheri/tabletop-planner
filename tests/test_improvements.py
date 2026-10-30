from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tabletop_planner.config import RobotConfig
from tabletop_planner.data import generate_dataset
from tabletop_planner.domain import Action, apply_goal_spec
from tabletop_planner.evaluation import evaluate, rollout
from tabletop_planner.observations import (DEFAULT_FORMAT, LEGACY_FORMAT, config_from_rows,
                                          load_planner_config, messages_for, model_observation, planner_config)
from tabletop_planner.symbolic import SymbolicBackend
from tabletop_planner.tasks import make_episode


class ObservationTests(unittest.TestCase):
    def test_coordinates_order_and_extra_fields_do_not_change_model_input(self):
        obs = SymbolicBackend(make_episode(0)).observe()
        changed = deepcopy(obs)
        changed['objects'].reverse()
        for obj in changed['objects']:
            obj['position'] = [99, -17, .02]
            obj['hidden_goal'] = 'secret'
        changed['trays']['left_tray'] = [-5, -5, -5]
        changed['goals'] = ['secret']
        self.assertEqual(messages_for('Move red left', obs), messages_for('Move red left', changed))
        self.assertNotIn('position', model_observation(obs)['objects'][0])
        self.assertIn('position', obs['objects'][0])

    def test_legacy_adapter_and_prompt_are_preserved(self):
        obs = SymbolicBackend(make_episode(0)).observe()
        config = load_planner_config('an_old_adapter_without_metadata')
        self.assertEqual(config['observation_format'], LEGACY_FORMAT)
        self.assertIn('Coordinates are metres', config['system_prompt'])
        self.assertEqual(model_observation(obs, LEGACY_FORMAT), obs)
        self.assertEqual(load_planner_config()['observation_format'], DEFAULT_FORMAT)

    def test_saved_prompt_is_verified_and_used(self):
        with tempfile.TemporaryDirectory() as directory:
            config = planner_config(prompt='A saved experiment prompt')
            path = Path(directory)/'planner_config.json'
            path.write_text(json.dumps(config))
            self.assertEqual(load_planner_config(directory), config)
            config['system_prompt'] = 'changed without updating hash'
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_planner_config(directory)

    def test_mixed_formats_are_rejected(self):
        obs = SymbolicBackend(make_episode(0)).observe()
        rows = [{'messages': messages_for('test', obs, fmt)} for fmt in (DEFAULT_FORMAT, LEGACY_FORMAT)]
        with self.assertRaises(ValueError):
            config_from_rows(rows)


class DataAndConfigTests(unittest.TestCase):
    def test_failed_expert_episode_does_not_publish_partial_data(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'data'
            with patch('tabletop_planner.data.expert_action', return_value=Action('done')):
                with self.assertRaises(RuntimeError):
                    generate_dataset(output, 1, 1, 1)
            self.assertFalse(output.exists())

    def test_precreated_empty_dataset_directory_is_supported(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory)/'data'
            output.mkdir()
            generate_dataset(output, 1, 1, 1)
            self.assertTrue((output/'manifest.json').exists())

    def test_recovery_examples_are_training_only_and_outputs_are_protected(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory)/'data'
            manifest = generate_dataset(output, 10, 5, 5, recovery_rate=1)
            self.assertGreater(manifest['splits']['train']['recovery_observations'], 0)
            self.assertEqual(manifest['splits']['val']['recovery_observations'], 0)
            self.assertEqual(manifest['splits']['test']['recovery_observations'], 0)
            rows = [json.loads(line) for line in (output/'train.jsonl').read_text().splitlines()]
            observations = [json.loads(row['messages'][1]['content'])['observation'] for row in rows]
            self.assertTrue(any(any(e.get('error') == 'SIMULATED_EXECUTION_FAILURE' for e in obs['history']) for obs in observations))
            original = (output/'train.jsonl').read_bytes()
            with self.assertRaises(ValueError):
                generate_dataset(output, 10, 5, 5)
            self.assertEqual((output/'train.jsonl').read_bytes(), original)

    def test_robot_profile_controls_generated_and_executed_positions(self):
        config = RobotConfig(trays={'left_tray': [.48, .22, .035], 'right_tray': [.48, -.22, .035]})
        episode = make_episode(0, kind='already_done', config=config)
        self.assertEqual(episode.objects[int(episode.goals[0]['object_id'].split('_')[1])]['position'], config.trays[episode.goals[0]['zone']])
        env = SymbolicBackend(make_episode(0), config=config)
        env.execute(Action('pick', 'block_0'))
        env.execute(Action('place', 'left_tray'))
        self.assertEqual(env.objects['block_0']['position'], config.trays['left_tray'])
        self.assertNotEqual(RobotConfig().trays, config.trays)
        for invalid in ({'workspace_lower': [1, 1, 1]}, {'arm_force': float('nan')}, {'home': [0]}, {'scene_jitter': -1}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                RobotConfig(**invalid)

    @unittest.skipUnless(importlib.util.find_spec('pybullet'), 'Install .[sim]')
    def test_mixed_data_contains_actual_physics_observations(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory)/'data'
            manifest = generate_dataset(output, 10, 5, 5, backend='mixed')
            self.assertEqual(manifest['splits']['train']['backends'], {'symbolic': 5, 'bullet': 5})
            rows = [json.loads(line) for line in (output/'train.jsonl').read_text().splitlines()]
            self.assertEqual({row['backend'] for row in rows}, {'symbolic', 'bullet'})
            self.assertEqual({row['observation_format'] for row in rows}, {DEFAULT_FORMAT})


class EvaluationTests(unittest.TestCase):
    def test_false_missing_and_false_completion_are_distinguished(self):
        episode = make_episode(0, kind='single')
        for action, category in ((Action('fail', 'missing'), 'unexpected_refusal'),
                                 (Action('done'), 'premature_completion')):
            planner = Mock()
            planner.next_action.return_value = action
            result = rollout(episode, SymbolicBackend(episode), planner)
            self.assertFalse(result['passed'])
            self.assertIn(category, result['failure_categories'])

    def test_wrong_order_and_invalid_ids_are_diagnosed(self):
        episode = make_episode(0, kind='double')
        planner = Mock()
        planner.next_action.side_effect = [Action('pick', episode.goals[1]['object_id']), Action('done')]
        result = rollout(episode, SymbolicBackend(episode), planner)
        self.assertIn('wrong_order', result['failure_categories'])
        planner.next_action.side_effect = [Action('pick', 'invented_id'), Action('done')]
        result = rollout(episode, SymbolicBackend(episode), planner)
        self.assertEqual(result['invalid_actions'], 1)
        self.assertEqual(result['execution_failures'], 0)

    def test_execution_failures_are_separate_from_invalid_actions(self):
        episode = make_episode(0, kind='single')
        env = SymbolicBackend(episode)
        env.execute = lambda action: env.record(action, 'APPROACH_FAILED' if action.action == 'pick' else None)
        planner = Mock()
        planner.next_action.side_effect = [Action('pick', episode.goals[0]['object_id']), Action('done')]
        result = rollout(episode, env, planner)
        self.assertEqual(result['invalid_actions'], 0)
        self.assertEqual(result['execution_failures'], 1)
        self.assertIn('execution_failure', result['failure_categories'])

    def test_custom_goal_labels_are_independent_and_validate_objects(self):
        episode = make_episode(0, kind='single')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'goals.json'
            path.write_text(json.dumps({'expected': 'done', 'goals': [{'color': 'red', 'zone': 'left_tray'}]}))
            apply_goal_spec(episode, path)
            red = next(obj for obj in episode.objects if obj['color'] == 'red')
            self.assertEqual(episode.goals, [{'object_id': red['id'], 'zone': 'left_tray'}])
            planner = Mock()
            planner.next_action.side_effect = [Action('pick', red['id']), Action('place', 'left_tray'), Action('done')]
            self.assertTrue(rollout(episode, SymbolicBackend(episode), planner)['passed'])
            path.write_text(json.dumps({'goals': [{'color': 'purple', 'zone': 'left_tray'}]}))
            with self.assertRaises(ValueError):
                apply_goal_spec(episode, path)

    @unittest.skipUnless(importlib.util.find_spec('pybullet'), 'Install .[sim]')
    def test_paired_evaluation_reuses_the_same_episode(self):
        episode = make_episode(0, kind='single')
        goal = episode.goals[0]
        actions = [Action('pick', goal['object_id']), Action('place', goal['zone']), Action('done')]
        planner = Mock()
        planner.name = 'fixture'
        planner.next_action.side_effect = actions*2
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = Path(directory)/'episodes.jsonl'
            path.write_text(json.dumps(episode.to_dict())+'\n')
            output = Path(directory)/'evaluation'
            summary = evaluate(path, output, planner, backend_type='both')
            self.assertEqual(summary['agreement'], {'both_passed': 1})
            for backend in ('symbolic', 'bullet'):
                result = json.loads((output/backend/'rollouts.jsonl').read_text())
                self.assertEqual(result['episode_id'], episode.episode_id)
                self.assertEqual(result['instruction'], episode.instruction)
                self.assertEqual(result['failure_categories'], [])
