"""Offline checks; no model services or third-party runtime are required."""
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import local_runtime as runtime
from threading_utils import ordered_concurrent_function_calls


class LocalRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {
            'CHESS_LOCAL_MODEL': 'local-model', 'CHESS_BASE_URL': 'http://localhost:8000/v1',
            'CHESS_TEMPERATURE': '0', 'CHESS_MAX_TOKENS': '16384',
            'CHESS_EMBEDDING_MODEL': 'bge-m3',
            'CHESS_EMBEDDING_BASE_URL': 'http://localhost:9000/v1',
        }, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_model_entry_overrides_every_engine_and_zero_temperature(self):
        # Execute the actual factory without importing unavailable cloud SDKs.
        source = Path('src/llm/models.py').read_text()
        tree = ast.parse(source)
        factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_llm_chain')
        chat = Mock()
        scope = {'Any': object, 'local_chat_params': runtime.local_chat_params}
        exec(compile(ast.Module(body=[factory], type_ignores=[]), '<factory>', 'exec'), scope)
        with patch.dict(sys.modules, {'langchain_openai': types.SimpleNamespace(ChatOpenAI=chat)}):
            for name in ['gpt-4o', 'gemini-1.5-flash', 'unknown-local-model']:
                scope['get_llm_chain'](name, temperature=0.8)
                self.assertEqual(chat.call_args.kwargs['temperature'], 0)
                self.assertEqual(chat.call_args.kwargs['model'], 'local-model')
                self.assertEqual(chat.call_args.kwargs['max_tokens'], 16384)
                self.assertEqual(chat.call_args.kwargs['timeout'], 120)
                self.assertEqual(chat.call_args.kwargs['max_retries'], 2)

    def test_empty_concurrent_call_list(self):
        self.assertEqual(ordered_concurrent_function_calls([]), [])

    def test_local_empty_response_does_not_fallback(self):
        tree = ast.parse(Path('src/llm/models.py').read_text())
        factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'call_llm_chain')
        fallback = Mock()
        class ParserError(Exception):
            pass
        engine = Mock()
        engine.invoke.return_value = types.SimpleNamespace(content='')
        prompt = Mock()
        prompt.__or__ = Mock(return_value=engine)
        prompt.invoke.return_value = types.SimpleNamespace(messages=[types.SimpleNamespace(content='hello')])
        scope = {'Any': object, 'Dict': dict, 'os': os, 'Logger': Mock,
                 'check_stop': Mock(), 'record_failure': Mock(),
                 'OutputParserException': ParserError, 'get_llm_chain': fallback,
                 'OutputFixingParser': types.SimpleNamespace(from_llm=Mock(return_value=Mock()))}
        # The original parser-error path composes a Runnable chain. A tiny stand-in suffices.
        engine.__or__ = Mock(return_value=engine)
        exec(compile(ast.Module(body=[factory], type_ignores=[]), '<caller>', 'exec'), scope)
        with self.assertRaises(ParserError):
            scope['call_llm_chain'](prompt, engine, Mock(), {}, 0, max_attempts=1)
        fallback.assert_not_called()

    def test_embedding_uses_text_and_separate_service(self):
        embeddings = Mock()
        with patch.dict(sys.modules, {'langchain_openai': types.SimpleNamespace(OpenAIEmbeddings=embeddings)}):
            for default in ['text-embedding-3-large', 'text-embedding-3-small']:
                runtime.make_embeddings(default)
                self.assertEqual(embeddings.call_args.kwargs['model'], 'bge-m3')
                self.assertFalse(embeddings.call_args.kwargs['check_embedding_ctx_length'])
                self.assertEqual(embeddings.call_args.kwargs['chunk_size'], 4)
                self.assertEqual(embeddings.call_args.kwargs['openai_api_base'], 'http://localhost:9000/v1')

    def test_config_before_import_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / 'dev_databases'
            db.mkdir()
            parser = argparse.ArgumentParser()
            runtime.add_local_arguments(parser)
            args = parser.parse_args(['--base_url', 'http://localhost:8000',
                                      '--db_root_directory', str(db), '--temperature', '0'])
            args.num_workers = 1
            args.data_mode = 'dev'
            runtime.configure_local_runtime(args, parser)
            self.assertEqual(os.environ['DB_ROOT_PATH'], tmp)
            self.assertEqual(runtime.local_chat_params()['temperature'], 0)
            self.assertEqual(args.base_url, 'http://localhost:8000/v1')

    def test_indexes_build_skip_and_reject_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / 'example'
            db.mkdir()
            (db / 'example.sqlite').touch()
            def build_vector(path, **kwargs):
                p = Path(path) / 'context_vector_db'
                p.mkdir()
                (p / 'chroma.sqlite3').touch()
            def build_lsh(path, **kwargs):
                p = Path(path) / 'preprocessed'
                p.mkdir(exist_ok=True)
                for suffix in ('lsh', 'minhashes'):
                    (p / f'example_{suffix}.pkl').touch()
            vector, lsh = Mock(side_effect=build_vector), Mock(side_effect=build_lsh)
            modules = {
                'database_utils.db_catalog.preprocess': types.SimpleNamespace(make_db_context_vec_db=vector),
                'database_utils.db_values.preprocess': types.SimpleNamespace(make_db_lsh=lsh),
            }
            with patch.dict(sys.modules, modules):
                rows = [{'db_id': 'example'}, {'db_id': 'example'}]
                runtime.ensure_indexes(rows, tmp)
                runtime.ensure_indexes(rows, tmp)
                vector.assert_called_once()
                lsh.assert_called_once()
                os.environ['CHESS_EMBEDDING_MODEL'] = 'different-model'
                with self.assertRaisesRegex(ValueError, 'incompatible'):
                    runtime.ensure_indexes(rows, tmp)

    def test_export_order_failures_and_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'chess_bird_dev_model_run_1'
            directory.mkdir()
            (directory / '-predictions.json').write_text(json.dumps({
                '7': 'SELECT 7\t----- bird -----\tdb_b', '91': 0,
                '42': 'SELECT 42\t----- bird -----\tdb_a',
            }))
            rows = [{'question_id': 42, 'db_id': 'db_a'},
                    {'question_id': 91, 'db_id': 'db_c'},
                    {'question_id': 7, 'db_id': 'db_b'}]
            target = runtime.export_bird_predictions(rows, str(directory))
            result = json.loads(target.read_text())
            self.assertEqual(target.name, directory.name + '.json')
            self.assertEqual(list(result), ['0', '1', '2'])
            self.assertEqual(result['0'], 'SELECT 42\t----- bird -----\tdb_a')
            self.assertEqual(result['1'], '-- generation failed\t----- bird -----\tdb_c')
            self.assertFalse((directory / 'predict_dev.json').exists())
            self.assertTrue(all(len(v.split('\t----- bird -----\t')) == 2 for v in result.values()))

    def test_script_arguments_without_running_python(self):
        script = Path('run_bird_dev.sh').read_text().replace('exec python3 -u', 'exec /bin/echo -u')
        result = subprocess.run(['bash', '-c', script], env=os.environ.copy(),
                                capture_output=True, text=True, check=True)
        self.assertIn('--temperature 0 ', result.stdout)
        self.assertIn('--num_workers 8 ', result.stdout)
        self.assertIn('--local_model Ministral-3-14B-Reasoning-2512 ', result.stdout)
        self.assertIn('--max_tokens 8192', result.stdout)
        self.assertNotIn('predict_dev.json', result.stdout)


if __name__ == '__main__':
    unittest.main()
