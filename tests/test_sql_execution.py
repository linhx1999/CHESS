"""Regression checks for SQL cancellation and worker shutdown."""
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
from database_utils.execution import execute_sql

LONG_QUERY = '''WITH RECURSIVE n(x) AS (
    VALUES(1) UNION ALL SELECT x + 1 FROM n WHERE x < 1000000000
) SELECT sum(x) FROM n'''


class SQLExecutionTests(unittest.TestCase):
    def test_fetch_modes_and_errors(self):
        query = 'SELECT 1 UNION ALL SELECT 2'
        self.assertEqual(execute_sql(':memory:', query), [(1,), (2,)])
        self.assertEqual(execute_sql(':memory:', query, fetch='one'), (1,))
        self.assertEqual(execute_sql(':memory:', query, fetch=1), [(1,)])
        self.assertIn(execute_sql(':memory:', query, fetch='random'), [(1,), (2,)])
        with self.assertRaises(sqlite3.OperationalError):
            execute_sql(':memory:', 'SELECT * FROM missing_table')
        with self.assertRaises(ValueError):
            execute_sql(':memory:', query, fetch='invalid')

    def test_timeout_cancels_query_without_leaving_threads(self):
        before = set(threading.enumerate())
        start = time.monotonic()
        with self.assertRaises(TimeoutError):
            execute_sql(':memory:', LONG_QUERY, timeout=0.05)
        self.assertLess(time.monotonic() - start, 2)
        self.assertEqual(set(threading.enumerate()), before)
        self.assertEqual(execute_sql(':memory:', 'SELECT 42'), [(42,)])

    def test_database_lock_respects_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'locked.sqlite')
            conn = sqlite3.connect(path)
            try:
                conn.execute('CREATE TABLE t(x)')
                conn.commit()
                conn.execute('BEGIN EXCLUSIVE')
                start = time.monotonic()
                with self.assertRaises(TimeoutError):
                    execute_sql(path, 'SELECT * FROM t', timeout=0.05)
                self.assertLess(time.monotonic() - start, 2)
            finally:
                conn.close()
            self.assertEqual(execute_sql(path, 'SELECT * FROM t'), [])

    def test_pool_exits_after_timed_out_query(self):
        # Isolate the pool and bound the whole process, so a shutdown regression
        # fails instead of hanging the test runner.
        code = f'''
from multiprocessing import Pool
from database_utils.execution import execute_sql
pool = Pool(2)
result = pool.apply_async(execute_sql, (':memory:', {LONG_QUERY!r}), {{'timeout': 0.05}})
try:
    result.get(timeout=5)
except TimeoutError:
    pass
else:
    raise AssertionError('Expected SQL timeout')
pool.close()
pool.join()
print('pool exited')
'''
        result = subprocess.run(
            [sys.executable, '-c', code], capture_output=True, text=True,
            timeout=10, env={**os.environ, 'PYTHONPATH': str(SRC)},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('pool exited', result.stdout)


if __name__ == '__main__':
    unittest.main()
