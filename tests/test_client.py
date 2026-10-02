import pytest

from chap_client import ChapClient, ChapClientError, JobFailed, camel_keys


def test_camel_keys_converts_names_and_drops_none():
    assert camel_keys({"n_periods": 3, "future_weather_provider": None, "stride": 1}) == {"nPeriods": 3, "stride": 1}


def test_token_is_sent_as_bearer_header(fake_session):
    ChapClient("http://chap", token="secret", session=fake_session)
    assert fake_session.headers["Authorization"] == "Bearer secret"


def test_is_healthy_uses_root_health_endpoint(client, fake_session):
    assert client.is_healthy()
    assert fake_session.calls[-1].path == "/health"


def test_is_healthy_is_false_when_chap_does_not_answer(client, fake_session):
    del fake_session.routes[("GET", "/health")]
    assert not client.is_healthy()


def test_list_specifications_sends_camel_case_query(client, fake_session, specification_summary):
    result = client.list_specifications(dataset_id=7, n_periods=3, n_splits=2, future_weather_provider=None)
    assert result == [specification_summary]
    call = fake_session.calls[-1]
    assert call.method == "GET"
    assert call.params == {"datasetId": 7, "nPeriods": 3, "nSplits": 2}


def test_create_backtests_posts_camel_case_body_and_returns_specification_and_jobs(client, fake_session, backtest_params):
    response = client.create_backtests("rwanda_monthly", dataset_id=7, model_ids=[3, 5], backtest_params=backtest_params)
    assert response["specificationId"] == 11
    assert [job["jobId"] for job in response["jobs"]] == ["job-ok", "job-bad"]
    body = fake_session.calls[-1].json
    assert body == {"name": "rwanda_monthly", "datasetId": 7, "modelIds": [3, 5], "nPeriods": 3, "nSplits": 2, "stride": 1, "nRetrain": 1}


def test_wait_for_job_returns_database_id_on_success(client):
    assert client.wait_for_job("job-ok", poll_interval=0) == 42


def test_wait_for_job_raises_with_logs_on_failure(client):
    with pytest.raises(JobFailed) as excinfo:
        client.wait_for_job("job-bad", poll_interval=0)
    assert excinfo.value.status == "FAILURE"
    assert "boom" in excinfo.value.logs


def test_http_error_raises_client_error(client):
    with pytest.raises(ChapClientError, match="404"):
        client.get_specification(999)
