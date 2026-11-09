"""Offline resume and scheduler regressions; no model SDKs or services required."""
import ast
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import resume
from runner.statistics_manager import StatisticsManager


def manager_class():
    tree = ast.parse(Path('src/runner/run_manager.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    scope = dict(vars(resume))
    import os
    import fcntl
    from typing import Any, List, Dict, Tuple
    scope.update(locals())
    scope.update({'StatisticsManager': StatisticsManager, 'SystemState': types.SimpleNamespace,
                  'Task': types.SimpleNamespace, 'Event': Event, 'json': json, 'Path': Path})
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<manager>', 'exec'), scope)
    return scope['RunManager'], scope


def history():
    steps = [{'tool_name': t, 'status': 'success'} for t in sorted(resume.LEGACY_TOOLS)]
    steps[-1]['final_SQL'] = {'PREDICTED_SQL': 'SELECT 1', 'exec_res': 0,
                              'exec_err': 'incorrect answer'}
    return steps


class ResumeTests(unittest.TestCase):
    def setUp(self):
        resume.init_worker(Event())
        resume.reset_task()

    def test_legacy_requires_every_tool_and_nonempty_sql(self):
        self.assertEqual(resume.completed_sql(history()), 'SELECT 1')
        self.assertIsNone(resume.completed_sql(history()[1:]))
        failed = history()
        failed[0]['status'] = 'error'
        self.assertIsNone(resume.completed_sql(failed))
        failed = history()
        failed[-1]['final_SQL'] = None
        self.assertIsNone(resume.completed_sql(failed))
        self.assertIsNone(resume.completed_sql({'invalid': True}))

    def test_new_incomplete_and_failed_markers_override_legacy_evidence(self):
        for status in ('running', 'error'):
            self.assertIsNone(resume.completed_sql(history() + [
                {'tool_name': 'run_status', 'status': status}]))
        self.assertEqual(resume.completed_sql([history()[-1],
            {'tool_name': 'run_status', 'status': 'success'}]), 'SELECT 1')

    def test_threaded_permission_error_sets_shared_stop(self):
        from threading_utils import ordered_concurrent_function_calls
        error = RuntimeError('AccessDenied.Unpurchased')
        def fail():
            raise error
        self.assertEqual(ordered_concurrent_function_calls([
            {'function': fail, 'kwargs': {}}]), [None])
        self.assertTrue(resume.task_failed())
        with self.assertRaisesRegex(RuntimeError, 'stopped'):
            resume.check_stop()
        resume.reset_task()
        with self.assertRaises(RuntimeError):
            resume.check_stop()  # A new task cannot clear a run-wide fatal error.

    def test_nonfatal_error_does_not_stop_other_questions(self):
        resume.record_failure(ValueError('bad parse'))
        self.assertTrue(resume.task_failed())
        resume.check_stop()
        resume.reset_task()
        self.assertFalse(resume.task_failed())

    def test_restore_saved_model_without_overwriting_operational_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = {'local_model': 'qwen', 'base_url': 'https://example.test/v1',
                     'config': {'setting_name': 'native'}, 'num_workers': 8}
            path = Path(tmp) / '-args.json'
            path.write_text(json.dumps(saved))
            args = types.SimpleNamespace(output_dir=tmp, resume=True, local_model='other',
                                         num_workers=2)
            resume.restore_arguments(args)
            self.assertEqual(args.local_model, 'qwen')
            self.assertEqual(args.config, saved['config'])
            self.assertEqual(args.num_workers, 2)
            self.assertEqual(json.loads(path.read_text()), saved)

    def test_resume_rebuilds_predictions_and_keeps_histories(self):
        cls, _ = manager_class()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good, bad = root / '0_db.json', root / '1_db.json'
            good.write_text(json.dumps(history()))
            bad.write_text(json.dumps(history()[1:]))
            original = {p: p.read_bytes() for p in (good, bad)}
            (root / '-args.json').write_text('{"old": true}')
            (root / '-predictions.json').write_text('{"1": "interrupted write')
            args = types.SimpleNamespace(output_dir=tmp, resume=True, data_mode='dev',
                config={'setting_name': 'native'}, data_path='dev.json', run_start_time='now')
            manager = cls(args)
            manager.initialize_tasks([{'db_id': 'db'}, {'db_id': 'db'}, {'db_id': 'db'}])
            self.assertEqual([t.question_id for t in manager.tasks], [1, 2])
            predictions = json.loads((root / '-predictions.json').read_text())
            self.assertEqual(predictions, {'0': 'SELECT 1\t----- bird -----\tdb', '1': 0, '2': 0})
            self.assertEqual((root / '-args.json').read_text(), '{"old": true}')
            for p, content in original.items():
                self.assertEqual(p.read_bytes(), content)
            args.resume = False
            with self.assertRaises(FileExistsError):
                cls(args)

    def test_bounded_scheduler_saves_batch_and_stops(self):
        cls, scope = manager_class()
        for workers in (1, 2):
            with self.subTest(workers=workers):
                manager = cls.__new__(cls)
                manager.args = types.SimpleNamespace(num_workers=workers)
                manager.tasks = list(range(12))
                submitted = []
                def worker(task):
                    submitted.append(task)
                    resume.record_failure(RuntimeError('AccessDenied.Unpurchased'))
                    return task
                manager.worker = worker
                manager.task_done = Mock()
                pool = Mock()
                pool.__enter__ = Mock(return_value=pool)
                pool.__exit__ = Mock(return_value=False)
                pool.imap_unordered.side_effect = lambda fn, tasks: map(fn, tasks)
                scope['Pool'] = Mock(return_value=pool)
                with self.assertRaisesRegex(RuntimeError, 'progress saved'):
                    manager.run_tasks()
                self.assertEqual(submitted, list(range(workers)))
                self.assertEqual(manager.task_done.call_count, workers)

    def test_worker_swallowed_failure_is_persisted(self):
        cls, scope = manager_class()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(sys.modules, {'runner.task': types.SimpleNamespace(Task=object)}):
                from runner.logger import Logger
            scope['Logger'] = Logger
            scope['DatabaseManager'] = Mock()
            def stream(state, *args, **kwargs):
                resume.record_failure(RuntimeError('AccessDenied.Unpurchased'))
                yield {'execution_history': history()}
            scope['build_team'] = Mock(return_value=types.SimpleNamespace(stream=stream))
            manager = cls.__new__(cls)
            manager.result_directory = tmp
            manager.args = types.SimpleNamespace(data_mode='dev', log_level='warning',
                                                 config={}, run_start_time='now')
            manager.worker(types.SimpleNamespace(db_id='db', question_id=0))
            stored = resume.read_history(Path(tmp) / '0_db.json')
            self.assertEqual(stored[-1]['status'], 'error')
            self.assertIsNone(resume.completed_sql(stored))


if __name__ == '__main__':
    unittest.main()
