#!/usr/bin/env python
"""Run the test suite one pytest process per test file, in parallel.

One process per file means no test can poison another file through leaked
widgets, threads, monkeypatches, or module state, so an order-dependent hang
or crash costs one file instead of the whole run. Each file also gets a hard
wall-clock limit enforced from outside: a test stuck in a native modal loop
(``QDialog::exec``) cannot be interrupted by pytest-timeout's signal method,
and its thread method ``os._exit``s the process, so the outer kill is the
backstop for both.

    python tools/run_tests_per_file.py                    # whole suite
    python tools/run_tests_per_file.py -j 4 tests/identity
    python tools/run_tests_per_file.py -- -k cache -x     # extra pytest args

Writes per-file logs plus ``failures.txt`` (sorted ``FAILED``/``ERROR`` node
ids, plus ``HUNG``/``CRASHED`` file entries) to ``--log-dir``, and exits
non-zero when anything failed. Files run longest-first using the durations
recorded by the previous run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_DIR = REPO_ROOT / ".pytest-per-file"
_RESULT_LINE = re.compile(r"^(FAILED|ERROR) (\S+)")
_COUNT = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed)")
# pytest exit codes: 0 ok, 1 tests failed, 5 nothing collected.
_OK_CODES = {0, 5}


@dataclass
class FileResult:
    path: str
    returncode: int | None
    seconds: float
    failures: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failures


def _discover(targets: list[str]) -> list[str]:
    found: set[Path] = set()
    for target in targets:
        path = Path(target).resolve()
        if path.is_file():
            found.add(path)
            continue
        for pattern in ("test_*.py", "*_test.py"):
            found.update(path.rglob(pattern))
    return sorted(
        p.relative_to(REPO_ROOT).as_posix() if p.is_relative_to(REPO_ROOT) else str(p)
        for p in found
    )


def _slug(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", path.removesuffix(".py"))


def _parse(
    path: str, returncode: int | None, output: str, seconds: float
) -> FileResult:
    result = FileResult(path, returncode, seconds)
    lines = output.splitlines()
    for line in lines:
        match = _RESULT_LINE.match(line)
        if match:
            result.failures.append(f"{match.group(1)} {match.group(2)}")
    for line in reversed(lines):
        # -q prints a bare "3 passed, 1 failed in 1.2s"; -v wraps it in "=".
        if " in " in line and _COUNT.search(line):
            for number, kind in _COUNT.findall(line):
                kind = "errors" if kind.startswith("error") else kind
                result.counts[kind] = int(number)
            break
    if returncode is None:
        result.failures.append(f"HUNG {path} (killed after {seconds:.0f}s)")
    elif returncode < 0:
        try:
            name = signal.Signals(-returncode).name
        except ValueError:
            name = f"signal {-returncode}"
        result.failures.append(f"CRASHED {path} ({name})")
    elif returncode not in _OK_CODES and not result.failures:
        # e.g. pytest-timeout's thread method os._exit(1)s before any summary.
        hint = "pytest-timeout" if "+ Timeout +" in output else f"exit {returncode}"
        result.failures.append(f"ERROR {path} ({hint}; see log)")
    return result


class _Runner:
    def __init__(self, args: argparse.Namespace, pytest_args: list[str]):
        self.args = args
        self.pytest_args = pytest_args
        self.log_dir: Path = args.log_dir
        self._live: dict[int, subprocess.Popen] = {}
        self._lock = threading.Lock()
        self._stopping = False

    def run_file(self, path: str) -> FileResult:
        slug = _slug(path)
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            path,
            "-q",
            "-rfE",
            "-p",
            "no:cacheprovider",
            *self.pytest_args,
        ]
        env = dict(os.environ)
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
        env.setdefault("PYTHONFAULTHANDLER", "1")
        start = time.monotonic()
        log_path = self.log_dir / "logs" / f"{slug}.log"
        with open(log_path, "w", encoding="utf-8", errors="replace") as log:
            with self._lock:
                if self._stopping:
                    return FileResult(
                        path, 0, 0.0, ["ERROR " + path + " (not run: interrupted)"]
                    )
                proc = subprocess.Popen(
                    cmd,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                self._live[proc.pid] = proc
            try:
                returncode = proc.wait(timeout=self.args.file_timeout)
            except subprocess.TimeoutExpired:
                _kill_group(proc)
                returncode = None
            finally:
                with self._lock:
                    self._live.pop(proc.pid, None)
        seconds = time.monotonic() - start
        output = log_path.read_text(encoding="utf-8", errors="replace")
        return _parse(path, returncode, output, seconds)

    def kill_all(self) -> None:
        with self._lock:
            self._stopping = True
            procs = list(self._live.values())
        for proc in procs:
            _kill_group(proc)


def _kill_group(proc: subprocess.Popen) -> None:
    # The file's whole session: sidecar children and resource trackers too.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _default_jobs() -> int:
    return max(1, min(8, (os.cpu_count() or 2) // 2))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    pytest_args: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, pytest_args = argv[:split], argv[split + 1 :]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "targets",
        nargs="*",
        default=["tests"],
        help="test files or directories (default: tests)",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=int(os.environ.get("PYTEST_JOBS", _default_jobs())),
        help="parallel processes (default: $PYTEST_JOBS or min(8, cpus/2))",
    )
    parser.add_argument(
        "--file-timeout",
        type=float,
        default=float(os.environ.get("PYTEST_FILE_TIMEOUT", 1800)),
        help="hard wall-clock seconds per file before its process "
        "group is killed (default: 1800)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help=f"logs, failures.txt, durations.json (default: {DEFAULT_LOG_DIR.name}/)",
    )
    args = parser.parse_args(argv)
    args.log_dir = args.log_dir.resolve()
    (args.log_dir / "logs").mkdir(parents=True, exist_ok=True)

    os.chdir(REPO_ROOT)
    files = _discover(args.targets)
    if not files:
        print("no test files found", file=sys.stderr)
        return 5
    durations_path = args.log_dir / "durations.json"
    try:
        previous = json.loads(durations_path.read_text())
    except (OSError, ValueError):
        previous = {}
    # Longest first so one slow file does not start last and set the wall time.
    files.sort(key=lambda f: -float(previous.get(f, 60.0)))

    runner = _Runner(args, pytest_args)
    results: list[FileResult] = []
    started = time.monotonic()
    print(
        f"{len(files)} files, {args.jobs} jobs, {args.file_timeout:.0f}s/file limit; "
        f"logs in {args.log_dir}",
        flush=True,
    )
    pool = ThreadPoolExecutor(max_workers=args.jobs)
    try:
        futures = {pool.submit(runner.run_file, f): f for f in files}
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            status = "ok  " if result.ok else "FAIL"
            print(
                f"[{done:>{len(str(len(files)))}}/{len(files)}] {status} "
                f"{result.path} ({result.seconds:.1f}s)",
                flush=True,
            )
            for failure in result.failures:
                print(f"      {failure}", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted: killing running files", file=sys.stderr, flush=True)
        runner.kill_all()
        pool.shutdown(wait=True, cancel_futures=True)
        return 130
    pool.shutdown(wait=True)

    previous.update(
        {r.path: round(r.seconds, 1) for r in results if r.returncode is not None}
    )
    durations_path.write_text(json.dumps(previous, indent=0, sort_keys=True))
    failures = sorted(f for r in results for f in r.failures)
    (args.log_dir / "failures.txt").write_text("".join(f + "\n" for f in failures))

    totals: dict[str, int] = {}
    for result in results:
        for kind, number in result.counts.items():
            totals[kind] = totals.get(kind, 0) + number
    summary = ", ".join(f"{n} {k}" for k, n in sorted(totals.items()))
    print(
        f"\n{summary or 'no tests ran'} across {len(results)} files "
        f"in {time.monotonic() - started:.0f}s"
    )
    if failures:
        print(f"{len(failures)} failure(s) -> {args.log_dir / 'failures.txt'}")
        for failure in failures:
            print(f"  {failure}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
