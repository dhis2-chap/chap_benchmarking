"""Run the standing benchmarks against a chap-core instance and read the results back.

A benchmark problem is a human-chosen name for a dataset plus the backtest
parameters plus the configured models to run. chap-core deduplicates the
(dataset, parameters) tuple into a `BacktestSpecification`, so every backtest a
problem produces lands under one specification and is comparable by
construction. Results live in chap-core's database; this script keeps nothing.

Model identity is the configured model row in chap-core, which is immutable per
name, version and configuration. A model is pending for a problem when it has no
backtest under the problem's specification, so registering a new model version in
chap-core is what triggers a run for it.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cyclopts
import pydantic
import yaml

from chap_client import ChapClient, ChapClientError

logger = logging.getLogger(__name__)


class BacktestParams(pydantic.BaseModel):
    """Mirror of chap-core's BacktestParams. Names must match so a problem maps to one specification."""

    n_periods: int
    n_splits: int
    stride: int = 1
    n_retrain: int = 1
    future_weather_provider: str | None = None

    def as_request(self) -> dict:
        """Parameters to send, omitting an unset weather provider so chap-core applies its default."""
        return self.model_dump(exclude_none=True)


class Problem(pydantic.BaseModel):
    name: str
    dataset_name: str
    backtest_params: BacktestParams
    models: list[str]


class RunResult(pydantic.BaseModel):
    problem: str
    model: str
    job_id: str | None = None
    backtest_id: int | None = None
    error: str | None = None


def load_problems(path: Path) -> list[Problem]:
    with open(path) as f:
        return pydantic.TypeAdapter(list[Problem]).validate_python(yaml.safe_load(f))


class BenchmarkRunner:
    def __init__(self, client: ChapClient, timeout: float = 3600, poll_interval: float = 30):
        self.client = client
        self.timeout = timeout
        self.poll_interval = poll_interval

    def dataset(self, problem: Problem) -> dict:
        datasets = self.client.list_datasets()
        for dataset in datasets:
            if dataset["name"] == problem.dataset_name:
                return dataset
        raise ValueError(f"Dataset {problem.dataset_name!r} not found in chap. Available: {[d['name'] for d in datasets]}")

    def configured_models(self, problem: Problem) -> list[dict]:
        by_name = {model["name"]: model for model in self.client.list_configured_models()}
        missing = [name for name in problem.models if name not in by_name]
        if missing:
            raise ValueError(f"Configured models {missing} not found in chap. Available: {sorted(by_name)}")
        return [by_name[name] for name in problem.models]

    def find_specification(self, problem: Problem) -> dict | None:
        """The specification summary for this problem, or None if nothing has run under it yet."""
        dataset = self.dataset(problem)
        matches = self.client.list_specifications(dataset_id=dataset["id"], **problem.backtest_params.as_request())
        if len(matches) > 1:
            raise ValueError(
                f"Problem {problem.name!r} matches {len(matches)} specifications; "
                "set future_weather_provider in its backtest_params to pick one"
            )
        return matches[0] if matches else None

    def results(self, problem: Problem) -> dict | None:
        """The specification with every backtest under it, or None if nothing has run yet."""
        summary = self.find_specification(problem)
        return self.client.get_specification(summary["id"]) if summary else None

    def pending_models(self, problem: Problem) -> list[dict]:
        """Configured models of the problem with no backtest under its specification."""
        models = self.configured_models(problem)
        specification = self.results(problem)
        if specification is None:
            return models
        done = {backtest["configuredModel"]["id"] for backtest in specification["backtests"]}
        return [model for model in models if model["id"] not in done]

    def run(self, problem: Problem, force: bool = False) -> list[RunResult]:
        """Submit one backtest per pending model (every model with force) and wait for them all.

        One model failing does not stop the others; failures are returned, not raised.
        """
        dataset = self.dataset(problem)
        models = self.configured_models(problem) if force else self.pending_models(problem)
        if not models:
            logger.info("Problem %s: nothing to run", problem.name)
            return []
        results = []
        for model in models:
            result = RunResult(problem=problem.name, model=model["name"])
            try:
                result.job_id = self.client.create_backtest(
                    name=f"{problem.name}/{model['name']}",
                    dataset_id=dataset["id"],
                    model_id=model["id"],
                    backtest_params=problem.backtest_params.as_request(),
                )
                logger.info("Problem %s: submitted %s as job %s", problem.name, model["name"], result.job_id)
            except ChapClientError as e:
                result.error = str(e)
                logger.error("Problem %s: could not submit %s: %s", problem.name, model["name"], e)
            results.append(result)
        for result in results:
            if result.job_id is None:
                continue
            try:
                result.backtest_id = self.client.wait_for_job(result.job_id, self.timeout, self.poll_interval)
                logger.info("Problem %s: %s finished as backtest %s", problem.name, result.model, result.backtest_id)
            except ChapClientError as e:
                result.error = getattr(e, "logs", None) or str(e)
                logger.error("Problem %s: %s failed: %s", problem.name, result.model, e)
        return results


def format_results(specification: dict) -> str:
    """One line per backtest: model, version, source digest, chap version, created and aggregate metrics."""
    lines = [
        f"specification {specification['id']} on dataset {specification['dataset']['name']}: "
        f"n_periods={specification.get('nPeriods')} n_splits={specification.get('nSplits')} "
        f"stride={specification.get('stride')} n_retrain={specification.get('nRetrain')} "
        f"future_weather_provider={specification.get('futureWeatherProvider')}"
    ]
    for backtest in specification["backtests"]:
        model = backtest["configuredModel"]
        template = model.get("modelTemplate") or {}
        digest = (template.get("sourceDigest") or "-")[:12]
        metrics = " ".join(f"{k}={v:.4g}" for k, v in sorted(backtest.get("aggregateMetrics", {}).items()))
        lines.append(
            f"  {model['name']}  version={template.get('version') or '-'}  digest={digest}  "
            f"chap={backtest.get('chapVersion') or '-'}  created={backtest.get('created')}  {metrics}"
        )
    return "\n".join(lines)


app = cyclopts.App(help="Run the standing benchmarks against chap-core. Reads CHAP_URL and CHAP_API_TOKEN.")

DEFAULT_CONFIG_FOLDER = Path(os.environ.get("BENCHMARK_CONFIG_FOLDER", "./config"))


def _problems(config_folder: Path, name: str | None) -> list[Problem]:
    problems = load_problems(config_folder / "problem_specifications.yaml")
    if name is None:
        return problems
    selected = [p for p in problems if p.name == name]
    if not selected:
        raise ValueError(f"No problem named {name!r}. Available: {[p.name for p in problems]}")
    return selected


@app.command
def run(problem: str | None = None, config_folder: Path = DEFAULT_CONFIG_FOLDER, force: bool = False):
    """Run every model that has no result yet, for one problem or all of them."""
    runner = BenchmarkRunner(ChapClient.from_env())
    failed = []
    for p in _problems(config_folder, problem):
        for result in runner.run(p, force=force):
            if result.error:
                failed.append(result)
                print(f"FAILED {result.problem}/{result.model}:\n{result.error}")
            else:
                print(f"OK {result.problem}/{result.model} -> backtest {result.backtest_id}")
    if failed:
        raise SystemExit(1)


@app.command
def status(config_folder: Path = DEFAULT_CONFIG_FOLDER):
    """Show each problem's specification, how many backtests it has and which models are pending."""
    runner = BenchmarkRunner(ChapClient.from_env())
    for p in _problems(config_folder, None):
        summary = runner.find_specification(p)
        pending = [m["name"] for m in runner.pending_models(p)]
        spec = f"specification {summary['id']} with {summary['backtestCount']} backtests" if summary else "no runs yet"
        print(f"{p.name}: {spec}; pending: {pending or 'none'}")


@app.command
def results(problem: str, config_folder: Path = DEFAULT_CONFIG_FOLDER):
    """Print the backtests under a problem's specification with their aggregate metrics."""
    runner = BenchmarkRunner(ChapClient.from_env())
    (p,) = _problems(config_folder, problem)
    specification = runner.results(p)
    print(format_results(specification) if specification else f"{p.name}: no runs yet")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app()


if __name__ == "__main__":
    main()
