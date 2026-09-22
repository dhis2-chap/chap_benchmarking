from dataclasses import dataclass, field
from pathlib import Path

import pytest

from chap_client import ChapClient
from run_benchmarks import BacktestParams, BenchmarkRunner, Problem


@dataclass
class Call:
    method: str
    path: str
    params: dict | None
    json: dict | None


@dataclass
class FakeResponse:
    status_code: int
    payload: object

    def json(self):
        return self.payload

    @property
    def text(self):
        return str(self.payload)


@dataclass
class FakeSession:
    """Stands in for requests.Session: routes (method, path) to a canned (status, payload)."""

    routes: dict
    headers: dict = field(default_factory=dict)
    calls: list = field(default_factory=list)

    def request(self, method, url, params=None, json=None, timeout=None):
        path = url.split("/v1", 1)[1]
        self.calls.append(Call(method, path, params, json))
        status, payload = self.routes.get((method, path), (404, {"detail": "Not Found"}))
        return FakeResponse(status, payload)


@pytest.fixture
def example_config_dir():
    return Path(__file__).parent.parent / "example_config"


@pytest.fixture
def dataset():
    return {"id": 7, "name": "rwanda_evaluation_set", "type": "evaluation"}


@pytest.fixture
def configured_models():
    return [
        {"id": 3, "name": "naive_model", "version": "v1", "sourceDigest": "abc1234def"},
        {"id": 5, "name": "chap_ewars_monthly", "version": "stable", "sourceDigest": "9876543210"},
    ]


@pytest.fixture
def backtest_params():
    return {"n_periods": 3, "n_splits": 2, "stride": 1, "n_retrain": 1}


@pytest.fixture
def problem(dataset, backtest_params):
    return Problem(
        name="rwanda_monthly",
        dataset_name=dataset["name"],
        backtest_params=BacktestParams(**backtest_params),
        models=["naive_model", "chap_ewars_monthly"],
    )


@pytest.fixture
def specification_summary(dataset):
    return {
        "id": 11,
        "dataset": dataset,
        "nPeriods": 3,
        "nSplits": 2,
        "stride": 1,
        "nRetrain": 1,
        "futureWeatherProvider": "climatology",
        "orgUnitCount": 30,
        "backtestCount": 1,
    }


@pytest.fixture
def specification_read(specification_summary, configured_models):
    naive = configured_models[0]
    backtest = {
        "id": 42,
        "name": "rwanda_monthly/naive_model",
        "datasetId": 7,
        "modelId": "naive_model",
        "specificationId": 11,
        "created": "2026-09-22T10:00:00",
        "chapVersion": "2.4.0",
        "dataset": specification_summary["dataset"],
        "aggregateMetrics": {"crps": 1.5, "mae": 2.25},
        "configuredModel": {
            "id": naive["id"],
            "name": naive["name"],
            "configurationDigest": "cfg1",
            "modelTemplate": {"name": "naive_model", "version": "v1", "sourceDigest": "abc1234def"},
        },
    }
    return {
        **{k: v for k, v in specification_summary.items() if k not in ("orgUnitCount", "backtestCount")},
        "orgUnits": ["a", "b"],
        "backtests": [backtest],
    }


@pytest.fixture
def fake_session(dataset, configured_models, specification_summary, specification_read):
    routes = {
        ("GET", "/health"): (200, {"status": "success"}),
        ("GET", "/crud/datasets"): (200, [dataset]),
        ("GET", "/crud/configured-models"): (200, configured_models),
        ("GET", "/crud/backtest-specifications"): (200, [specification_summary]),
        ("GET", "/crud/backtest-specifications/11"): (200, specification_read),
        ("GET", "/jobs/job-ok"): (200, "SUCCESS"),
        ("GET", "/jobs/job-ok/database_result"): (200, {"id": 42}),
        ("GET", "/jobs/job-bad"): (200, "FAILURE"),
        ("GET", "/jobs/job-bad/logs"): (200, "Traceback: boom"),
    }
    session = FakeSession(routes)

    def create_backtest(method, url, params=None, json=None, timeout=None):
        job_id = "job-ok" if json["modelId"] == 3 else "job-bad"
        return FakeResponse(200, {"id": job_id})

    original = session.request

    def request(method, url, params=None, json=None, timeout=None):
        if method == "POST" and url.endswith("/analytics/create-backtest"):
            session.calls.append(Call(method, "/analytics/create-backtest", params, json))
            return create_backtest(method, url, params, json, timeout)
        return original(method, url, params, json, timeout)

    session.request = request
    return session


@pytest.fixture
def client(fake_session):
    return ChapClient("http://chap", session=fake_session)


@pytest.fixture
def runner(client):
    return BenchmarkRunner(client, timeout=5, poll_interval=0)
