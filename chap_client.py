"""HTTP client for the chap-core REST API, covering only what the benchmarks need.

A thin client over the REST API, never the database, so the two cannot drift.
Field names on the wire are camelCase; this module accepts snake_case parameter
names and converts them.
"""

from __future__ import annotations

import logging
import os
import time

import requests

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://localhost:8000"
URL_ENV_VAR = "CHAP_URL"
TOKEN_ENV_VAR = "CHAP_API_TOKEN"

SUCCESS_STATUSES = {"SUCCESS", "COMPLETED"}
FAILED_STATUSES = {"FAILURE", "FAILED", "ERROR", "REVOKED"}
RUNNING_STATUSES = {"PENDING", "RECEIVED", "STARTED", "RETRY"}


class ChapClientError(RuntimeError):
    """A request to chap-core failed."""


class JobFailed(ChapClientError):
    """A background job ended in a failed state. Carries the job's captured logs."""

    def __init__(self, job_id: str, status: str, logs: str):
        super().__init__(f"Job {job_id} ended with status {status}")
        self.job_id = job_id
        self.status = status
        self.logs = logs


class JobTimeout(ChapClientError):
    """A background job did not finish within the allowed time."""


def to_camel(name: str) -> str:
    head, *tail = name.split("_")
    return head + "".join(part.capitalize() for part in tail)


def camel_keys(values: dict) -> dict:
    """Convert snake_case keys to camelCase and drop None values."""
    return {to_camel(key): value for key, value in values.items() if value is not None}


class ChapClient:
    def __init__(self, base_url: str = DEFAULT_URL, token: str | None = None, session=None):
        self.root_url = base_url.rstrip("/")
        self.base_url = self.root_url + "/v1"
        self.session = session or requests.Session()
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    @classmethod
    def from_env(cls) -> ChapClient:
        """Build a client from CHAP_URL and CHAP_API_TOKEN, falling back to localhost without a token."""
        return cls(base_url=os.environ.get(URL_ENV_VAR, DEFAULT_URL), token=os.environ.get(TOKEN_ENV_VAR))

    def _request(self, method: str, path: str, url: str | None = None, **kwargs):
        url = url or f"{self.base_url}{path}"
        try:
            response = self.session.request(method, url, timeout=120, **kwargs)
        except requests.exceptions.RequestException as e:
            raise ChapClientError(f"{method} {url} failed: {e}") from e
        if response.status_code >= 400:
            raise ChapClientError(f"{method} {url} failed: {response.status_code} {response.text[:500]}")
        return response.json()

    def is_healthy(self) -> bool:
        """True if chap answers on its health endpoint, which lives at the root rather than under /v1."""
        try:
            self._request("GET", "/health", url=f"{self.root_url}/health")
            return True
        except ChapClientError:
            return False

    # datasets and models

    def list_datasets(self) -> list[dict]:
        return self._request("GET", "/crud/datasets")

    def list_configured_models(self) -> list[dict]:
        return self._request("GET", "/crud/configured-models")

    def make_dataset(self, request: dict) -> str:
        """Submit a dataset import and return the job id."""
        return self._request("POST", "/analytics/make-dataset", json=request)["id"]

    # specifications and backtests

    def list_specifications(self, dataset_id: int | None = None, **backtest_params) -> list[dict]:
        """List specification summaries, filtered by dataset id and any BacktestParams field."""
        params = camel_keys({"dataset_id": dataset_id, **backtest_params})
        return self._request("GET", "/crud/backtest-specifications", params=params)

    def get_specification(self, specification_id: int) -> dict:
        """One specification with every backtest under it, newest first."""
        return self._request("GET", f"/crud/backtest-specifications/{specification_id}")

    def create_backtest(self, name: str, dataset_id: int, model_id: int | str, backtest_params: dict) -> str:
        """Submit one backtest and return the job id."""
        body = camel_keys({"name": name, "dataset_id": dataset_id, "model_id": model_id, **backtest_params})
        return self._request("POST", "/analytics/create-backtest", json=body)["id"]

    def get_backtest(self, backtest_id: int) -> dict:
        return self._request("GET", f"/crud/backtests/{backtest_id}")

    # jobs

    def list_jobs(self) -> list[dict]:
        """Every job chap still tracks, with its name and status."""
        return self._request("GET", "/jobs")

    def job_status(self, job_id: str) -> str:
        status = self._request("GET", f"/jobs/{job_id}")
        return str(status).strip('"').upper()

    def job_logs(self, job_id: str) -> str:
        try:
            return str(self._request("GET", f"/jobs/{job_id}/logs"))
        except ChapClientError as e:
            return f"<logs unavailable: {e}>"

    def job_result_id(self, job_id: str) -> int:
        """The database id a finished job produced."""
        return self._request("GET", f"/jobs/{job_id}/database_result")["id"]

    def wait_for_job(self, job_id: str, timeout: float = 3600, poll_interval: float = 30) -> int:
        """Poll until the job finishes and return the database id it produced.

        Raises JobFailed with the job's logs if it fails, JobTimeout if it does not finish in time.
        """
        deadline = time.monotonic() + timeout
        while True:
            status = self.job_status(job_id)
            if status in SUCCESS_STATUSES:
                return self.job_result_id(job_id)
            if status in FAILED_STATUSES:
                raise JobFailed(job_id, status, self.job_logs(job_id))
            if time.monotonic() >= deadline:
                raise JobTimeout(f"Job {job_id} did not finish within {timeout} seconds (last status {status})")
            logger.info("Job %s status: %s", job_id, status)
            time.sleep(poll_interval)
