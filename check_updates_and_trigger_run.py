"""Cron entry point: run every problem's pending models, guarded by a lock file.

A model is pending when it has no backtest under the problem's specification, so
registering a new model version in chap-core is what causes a run. The lock stops
overlapping runs when cron fires while a previous run is still waiting on jobs.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import cyclopts

from chap_client import ChapClient
from run_benchmarks import DEFAULT_CONFIG_FOLDER, BenchmarkRunner, load_problems

logger = logging.getLogger(__name__)

LOCK_FILE = Path("/tmp/check_updates_and_trigger_run.lock")
STALE_LOCK_SECONDS = 6 * 3600


def run_pending(config_folder: Path) -> bool:
    """Run pending models for every problem. Returns False if any run failed."""
    client = ChapClient.from_env()
    if not client.is_healthy():
        logger.error("chap is not reachable at %s", client.base_url)
        return False
    runner = BenchmarkRunner(client)
    ok = True
    for problem in load_problems(config_folder / "problem_specifications.yaml"):
        for result in runner.run(problem):
            if result.error:
                ok = False
                logger.error("%s/%s failed:\n%s", result.problem, result.model, result.error)
            else:
                logger.info("%s/%s -> backtest %s", result.problem, result.model, result.backtest_id)
    return ok


def main(config_folder: Path = DEFAULT_CONFIG_FOLDER):
    """Run pending benchmarks unless another run holds the lock."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if LOCK_FILE.exists():
        age = time.time() - LOCK_FILE.stat().st_mtime
        if age < STALE_LOCK_SECONDS:
            logger.info("Another run holds the lock (%d seconds old), exiting", age)
            return
        logger.warning("Removing stale lock (%d seconds old)", age)
        LOCK_FILE.unlink()
    LOCK_FILE.touch()
    try:
        ok = run_pending(config_folder)
    finally:
        LOCK_FILE.unlink(missing_ok=True)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    cyclopts.run(main)
