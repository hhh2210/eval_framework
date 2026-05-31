import importlib.util
import json
import os
import re
import sys
import threading
import traceback
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Tuple

from tqdm import tqdm


@dataclass
class JsonlRecord:
    data: dict


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_jsonl(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str, records: Iterable[dict], mode: str = "w") -> None:
    with open(path, mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(path: str, record: dict) -> None:
    write_jsonl(path, [record], mode="a")


class ThreadSafeJsonlWriter:
    """Thread-safe JSONL writer for concurrent append operations."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        """Append a single record to the file in a thread-safe manner."""
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_many(self, records: list[dict]) -> None:
        """Append multiple records to the file in a thread-safe manner."""
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_module_from_path(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def strip_json_fence(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"```$", "", cleaned)
    return cleaned.strip()


def strip_think_tags(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def safe_json_loads(text: str) -> dict:
    cleaned = strip_json_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
        return {}


def maybe_download_nltk_resources(
    skip: bool,
    resources: list[tuple[str, str]],
) -> list[str]:
    try:
        import nltk
    except ImportError:
        return [package for _, package in resources]

    missing: list[str] = []
    for resource_path, package in resources:
        try:
            nltk.data.find(resource_path)
            continue
        except LookupError:
            if not skip:
                try:
                    nltk.download(package, quiet=True)
                except Exception:
                    pass
        try:
            nltk.data.find(resource_path)
        except LookupError:
            missing.append(package)
    return missing


def maybe_download_nltk(skip: bool) -> None:
    maybe_download_nltk_resources(
        skip,
        [
            ("tokenizers/punkt", "punkt"),
            ("tokenizers/punkt_tab/english", "punkt_tab"),
        ],
    )


def map_with_progress(
    f: Callable,
    xs: list[Any],
    num_threads: int = 128,
    pbar: bool = True,
    desc: str | None = None,
    writer: Optional[ThreadSafeJsonlWriter] = None,
) -> list[Any]:
    """
    Apply f to each element of xs using ThreadPool with tqdm progress bar.
    Based on OpenAI simple-evals pattern.

    Uses imap_unordered for better progress bar responsiveness (results returned
    as soon as they complete, not blocked by slow earlier tasks).

    Args:
        f: Function to apply to each element
        xs: List of inputs
        num_threads: Number of parallel threads
        pbar: Whether to show progress bar
        desc: Description for progress bar
        writer: Optional ThreadSafeJsonlWriter for immediate result persistence.
                If provided, each result is written immediately after completion.

    Returns:
        List of results (in completion order, not input order)
    """
    if not xs:
        return []

    pbar_fn = tqdm if pbar else lambda x, *args, **kwargs: x
    results = []

    if os.getenv("DEBUG"):
        # Sequential execution for debugging
        for x in pbar_fn(xs, total=len(xs), desc=desc):
            result = f(x)
            # If the worker function signals "skip" via None, do not write or record it.
            if result is None:
                continue
            if writer is not None:
                writer.write(result)
            results.append(result)
    else:
        with ThreadPool(min(num_threads, len(xs))) as pool:
            for result in pbar_fn(pool.imap_unordered(f, xs), total=len(xs), desc=desc):
                # If the worker function signals "skip" via None, do not write or record it.
                if result is None:
                    continue
                if writer is not None:
                    writer.write(result)
                results.append(result)

    return results


class ProgressCheckpoint:
    """Utility to track progress and provide checkpoint info on failure."""

    def __init__(self, phase: str, total: int):
        self.phase = phase
        self.total = total
        self.completed = 0
        self.failed = 0
        self._lock = threading.Lock()

    def increment(self, success: bool = True) -> None:
        with self._lock:
            if success:
                self.completed += 1
            else:
                self.failed += 1

    def summary(self) -> str:
        return f"[{self.phase}] Completed: {self.completed}/{self.total}, Failed: {self.failed}"


def map_with_progress_safe(
    f: Callable,
    xs: list[Any],
    num_threads: int = 128,
    pbar: bool = True,
    desc: str | None = None,
    writer: Optional[ThreadSafeJsonlWriter] = None,
    error_writer: Optional[ThreadSafeJsonlWriter] = None,
    on_error: str = "continue",  # "continue", "raise", "abort"
) -> Tuple[list[Any], list[dict]]:
    """
    Apply f to each element of xs with robust error handling.

    Unlike map_with_progress, this version:
    - Catches exceptions per-item instead of failing the entire batch
    - Optionally logs errors to a separate file
    - Returns both results and errors

    Args:
        f: Function to apply to each element
        xs: List of inputs
        num_threads: Number of parallel threads
        pbar: Whether to show progress bar
        desc: Description for progress bar
        writer: Optional ThreadSafeJsonlWriter for immediate result persistence
        error_writer: Optional ThreadSafeJsonlWriter for error logging
        on_error: How to handle errors:
            - "continue": Log error and continue processing
            - "raise": Re-raise the first exception after processing all items
            - "abort": Stop immediately on first error (still saves completed work)

    Returns:
        Tuple of (results, errors) where errors is a list of dicts with error info
    """
    if not xs:
        return [], []

    pbar_fn = tqdm if pbar else lambda x, *args, **kwargs: x
    results = []
    errors = []
    first_exception = None
    abort_flag = threading.Event()

    def safe_f(x):
        if abort_flag.is_set():
            return None, None, True  # Signal aborted
        try:
            result = f(x)
            return result, None, False
        except Exception as e:
            error_info = {
                "input": str(x)[:500],  # Truncate long inputs
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
            }
            return None, error_info, False

    if os.getenv("DEBUG"):
        # Sequential execution for debugging
        for x in pbar_fn(xs, total=len(xs), desc=desc):
            if abort_flag.is_set():
                break
            result, error, aborted = safe_f(x)
            if aborted:
                break
            if error is not None:
                errors.append(error)
                if error_writer is not None:
                    error_writer.write(error)
                if on_error == "abort":
                    abort_flag.set()
                    break
                elif on_error == "raise" and first_exception is None:
                    first_exception = Exception(error["error_message"])
            elif result is not None:
                if writer is not None:
                    writer.write(result)
                results.append(result)
    else:
        with ThreadPool(min(num_threads, len(xs))) as pool:
            for result, error, aborted in pbar_fn(
                pool.imap_unordered(safe_f, xs), total=len(xs), desc=desc
            ):
                if aborted:
                    continue
                if error is not None:
                    errors.append(error)
                    if error_writer is not None:
                        error_writer.write(error)
                    if on_error == "abort":
                        abort_flag.set()
                        # Note: already running tasks will complete
                    elif on_error == "raise" and first_exception is None:
                        first_exception = Exception(error["error_message"])
                elif result is not None:
                    if writer is not None:
                        writer.write(result)
                    results.append(result)

    if errors:
        print(
            f"\n⚠️  {len(errors)} errors occurred during {desc or 'processing'}.",
            file=sys.stderr,
        )
        print(f"   Successfully processed: {len(results)}/{len(xs)}", file=sys.stderr)

    if on_error == "raise" and first_exception is not None:
        raise first_exception

    return results, errors
