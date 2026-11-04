"""Cron entry point: run every problem's pending models, guarded by a lock file.

A model is pending when it has no backtest under the problem's specification, so
registering a new model version in chap-core is what causes a run. The lock stops
overlapping runs when cron fires while a previous run is still waiting on jobs.

When a run produces new backtests, the model marketplace is told to redeploy, so
its leaderboard picks them up within minutes instead of on its own schedule.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import cyclopts
import requests

from chap_client import ChapClient
from run_benchmarks import DEFAULT_CONFIG_FOLDER, BenchmarkRunner, load_problems

logger = logging.getLogger(__name__)

LOCK_FILE = Path("/tmp/check_updates_and_trigger_run.lock")
STALE_LOCK_SECONDS = 6 * 3600

MARKETPLACE_DISPATCH_URL = "https://api.github.com/repos/dhis2-chap/chap-marketplace/dispatches"
MARKETPLACE_TOKEN_ENV_VAR = "MARKETPLACE_DISPATCH_TOKEN"


def notify_marketplace(token: str | None, post=requests.post) -> None:
    """Ask the model marketplace to redeploy with the new results.

    Sends a `benchmarks-updated` repository_dispatch, which starts the marketplace's
    deploy workflow; that workflow reads the results from chap itself. Does nothing
    without a token, and a failure only logs: the results are stored either way, and
    the marketplace's scheduled deploy picks them up later.
    """
    if not token:
        logger.info("%s is not set, not notifying the marketplace", MARKETPLACE_TOKEN_ENV_VAR)
        return
    try:
        response = post(
            MARKETPLACE_DISPATCH_URL,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
            json={"event_type": "benchmarks-updated"},
            timeout=10,
        )
    except requests.RequestException as error:
        logger.warning("Could not notify the marketplace: %s", error)
        return
    if response.status_code != 204:
        logger.warning("Could not notify the marketplace: HTTP %s %s", response.status_code, response.text)
    else:
        logger.info("Notified the marketplace of new results")


def run_pending(config_folder: Path) -> tuple[bool, int]:
    """Run pending models for every problem.

    Returns whether every run succeeded, and how many new backtests were stored.
    """
    client = ChapClient.from_env()
    if not client.is_healthy():
        logger.error("chap is not reachable at %s", client.base_url)
        return False, 0
    runner = BenchmarkRunner(client)
    ok = True
    new_backtests = 0
    for problem in load_problems(config_folder / "problem_specifications.yaml"):
        for result in runner.run(problem):
            if result.error:
                ok = False
                logger.error("%s/%s failed:\n%s", result.problem, result.model, result.error)
            else:
                new_backtests += 1
                logger.info("%s/%s -> backtest %s", result.problem, result.model, result.backtest_id)
    return ok, new_backtests


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
        ok, new_backtests = run_pending(config_folder)
    finally:
        LOCK_FILE.unlink(missing_ok=True)
    if new_backtests:
        notify_marketplace(os.environ.get(MARKETPLACE_TOKEN_ENV_VAR))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    cyclopts.run(main)
