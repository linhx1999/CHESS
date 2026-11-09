"""Resume classification and process/thread-safe API failure signalling."""
import json
from pathlib import Path
from threading import Event

LEGACY_TOOLS = {
    "extract_keywords", "retrieve_entity", "retrieve_context", "select_tables",
    "select_columns", "filter_column", "generate_candidate", "revise",
    "execution_accuracy",
}
_failed = Event()
_stop = Event()


def init_worker(stop):
    global _stop
    _stop = stop


def reset_task():
    _failed.clear()


def record_failure(error):
    _failed.set()
    text = str(error).lower()
    if (getattr(error, "status_code", None) in (401, 402, 403)
            or any(code in text for code in (
                "accessdenied", "insufficient_quota", "insufficient_balance",
                "arrearage", "credit balance", "unpaid"))):
        _stop.set()


def check_stop():
    if _stop.is_set():
        raise RuntimeError("Run stopped after API authorization/billing failure")


def task_failed():
    return _failed.is_set()


def completed_sql(history):
    """New runs need an explicit success marker; legacy runs need all nine tools."""
    if not isinstance(history, list) or not history:
        return None
    if any(not isinstance(s, dict) or s.get("status") == "error" for s in history):
        return None
    markers = [s for s in history if s.get("tool_name") == "run_status"]
    if markers:
        if markers[-1].get("status") != "success":
            return None
    elif not LEGACY_TOOLS.issubset({
        s.get("tool_name") for s in history if s.get("status") == "success"
    }):
        return None
    for step in reversed(history):
        final = step.get("final_SQL")
        sql = final.get("PREDICTED_SQL") if isinstance(final, dict) else None
        if isinstance(sql, str) and sql.strip():
            return sql
    return None


def read_history(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return []


def restore_arguments(args):
    """Automatically resume an existing run using its saved configuration."""
    args.resume = False
    if not args.output_dir:
        return
    saved_path = Path(args.output_dir) / "-args.json"
    if not saved_path.is_file():
        return
    saved = json.loads(saved_path.read_text())
    args.resume = True
    for name, value in saved.items():
        if name not in {"output_dir", "resume", "num_workers", "log_level"}:
            setattr(args, name, value)
