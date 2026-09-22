import pytest

from run_benchmarks import BenchmarkRunner, Problem, format_results, load_problems


def test_load_problems_from_example_config(example_config_dir):
    problems = load_problems(example_config_dir / "problem_specifications.yaml")
    assert [p.name for p in problems] == ["rwanda_monthly"]
    problem = problems[0]
    assert problem.dataset_name == "rwanda_evaluation_set"
    assert problem.backtest_params.as_request() == {"n_periods": 3, "n_splits": 2, "stride": 1, "n_retrain": 1}
    assert problem.models == ["naive_model", "chap_ewars_monthly"]


def test_find_specification_filters_on_dataset_and_full_parameter_tuple(runner, problem, fake_session, specification_summary):
    assert runner.find_specification(problem) == specification_summary
    assert fake_session.calls[-1].params == {"datasetId": 7, "nPeriods": 3, "nSplits": 2, "stride": 1, "nRetrain": 1}


def test_find_specification_returns_none_when_nothing_has_run(runner, problem, fake_session):
    fake_session.routes[("GET", "/crud/backtest-specifications")] = (200, [])
    assert runner.find_specification(problem) is None


def test_find_specification_rejects_ambiguous_match(runner, problem, fake_session, specification_summary):
    fake_session.routes[("GET", "/crud/backtest-specifications")] = (
        200,
        [specification_summary, {**specification_summary, "id": 12}],
    )
    with pytest.raises(ValueError, match="future_weather_provider"):
        runner.find_specification(problem)


def test_unknown_model_name_is_an_error(runner, problem):
    broken = Problem(**{**problem.model_dump(), "models": ["no_such_model"]})
    with pytest.raises(ValueError, match="no_such_model"):
        runner.configured_models(broken)


def test_unknown_dataset_name_is_an_error(runner, problem):
    broken = Problem(**{**problem.model_dump(), "dataset_name": "no_such_dataset"})
    with pytest.raises(ValueError, match="no_such_dataset"):
        runner.dataset(broken)


def test_pending_models_skips_models_with_a_backtest_under_the_specification(runner, problem):
    assert [m["name"] for m in runner.pending_models(problem)] == ["chap_ewars_monthly"]


def test_pending_models_is_every_model_when_nothing_has_run(runner, problem, fake_session):
    fake_session.routes[("GET", "/crud/backtest-specifications")] = (200, [])
    assert [m["name"] for m in runner.pending_models(problem)] == ["naive_model", "chap_ewars_monthly"]


def test_pending_models_holds_back_models_whose_last_job_failed(runner, problem, fake_session):
    fake_session.routes[("GET", "/jobs")] = (
        200,
        [{"id": "j1", "name": "rwanda_monthly/chap_ewars_monthly", "status": "FAILURE", "type": "create_backtest"}],
    )
    assert runner.pending_models(problem) == []
    assert runner.failed_models(problem) == ["chap_ewars_monthly"]
    assert [r.model for r in runner.run(problem)] == []
    assert [r.model for r in runner.run(problem, force=True)] == ["naive_model", "chap_ewars_monthly"]


def test_pending_models_skips_models_with_a_running_job(runner, problem, fake_session):
    fake_session.routes[("GET", "/jobs")] = (
        200,
        [
            {"id": "j1", "name": "rwanda_monthly/chap_ewars_monthly", "status": "FAILURE", "type": "create_backtest"},
            {"id": "j2", "name": "rwanda_monthly/chap_ewars_monthly", "status": "STARTED", "type": "create_backtest"},
            {"id": "j3", "name": "other_problem/chap_ewars_monthly", "status": "FAILURE", "type": "create_backtest"},
        ],
    )
    assert runner.pending_models(problem) == []
    assert runner.failed_models(problem) == ["chap_ewars_monthly"]


def test_run_only_submits_pending_models(runner, problem, fake_session):
    results = runner.run(problem)
    submitted = [c.json["modelId"] for c in fake_session.calls if c.path == "/analytics/create-backtest"]
    assert submitted == [5]
    assert [r.model for r in results] == ["chap_ewars_monthly"]


def test_run_with_force_submits_every_model_and_one_failure_does_not_stop_the_rest(runner, problem, fake_session):
    results = runner.run(problem, force=True)
    submitted = [c.json for c in fake_session.calls if c.path == "/analytics/create-backtest"]
    assert [body["modelId"] for body in submitted] == [3, 5]
    assert submitted[0]["name"] == "rwanda_monthly/naive_model"
    assert submitted[0]["nPeriods"] == 3
    by_model = {r.model: r for r in results}
    assert by_model["naive_model"].backtest_id == 42
    assert by_model["naive_model"].error is None
    assert by_model["chap_ewars_monthly"].backtest_id is None
    assert "boom" in by_model["chap_ewars_monthly"].error


def test_format_results_lists_model_version_digest_and_metrics(specification_read):
    table = format_results(specification_read)
    assert "naive_model" in table
    assert "version=v1" in table
    assert "digest=abc1234def" in table
    assert "crps=1.5" in table
