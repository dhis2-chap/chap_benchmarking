# CHAP benchmarking

Runs the standing model benchmarks against a chap-core instance and keeps
nothing itself: results live in chap-core's database and are read back through
its REST API.

## How it fits together

A benchmark problem (`problem_specifications.yaml`) is a human-chosen name for a
dataset plus the backtest parameters plus the configured models to run. chap-core
deduplicates the (dataset, parameters) tuple into one `BacktestSpecification`,
so every backtest a problem produces lands under one specification and is
comparable by construction. This repo owns what to run and when; chap-core owns
the results and knows nothing about problem names.

A model is identified by its configured model row in chap-core, which is
immutable per name, version and configuration. A model is pending for a problem
when it has no backtest under the problem's specification. Registering a new
model version in chap-core (a new version label pinned to a commit in its
`config/configured_models/*.yaml`, then a restart) is therefore what triggers a
run for it.

Files:

- `chap_client.py`: thin HTTP client for the chap-core endpoints the benchmarks need
- `run_benchmarks.py`: `run`, `status` and `results` commands
- `check_updates_and_trigger_run.py`: cron entry point, runs pending models under a lock
- `seed_datasets.py`: import the benchmark datasets into chap

## Running locally

1. `git clone git@github.com:dhis2-chap/chap_benchmarking.git && cd chap_benchmarking && uv sync`
2. Have chap-core running (default `http://localhost:8000`). Point elsewhere with `CHAP_URL`; if the instance is token-gated set `CHAP_API_TOKEN`.
3. `cp -r example_config config` and edit `config/problem_specifications.yaml`.
4. Seed datasets: `uv run seed_datasets.py seed config/dataset_seeds.yaml`
5. `uv run run_benchmarks.py status` shows each problem's specification and pending models.
6. `uv run run_benchmarks.py run` runs every pending model. `--problem NAME` limits to one problem, `--force` reruns every model.
7. `uv run run_benchmarks.py results NAME` prints the backtests under a problem with version, source digest and aggregate metrics.

## Server

The benchmarking server keeps this repo in `/data/chap_benchmarking` with the live
configuration in `/data/chap_benchmarking/config/` and `CHAP_URL` /
`CHAP_API_TOKEN` in `/data/chap_benchmarking/.env`. `deploy.sh` runs on push to
main (see `.github/workflows/deploy.yml`) and installs a cron job that runs
`check_updates_and_trigger_run.py` every 15 minutes.

To trigger a run by hand:

```bash
cd /data/chap_benchmarking
set -a; . ./.env; set +a
.venv/bin/python check_updates_and_trigger_run.py
```

Results are rendered by the model marketplace, which fetches
`GET /v1/crud/backtest-specifications/{id}` from the server's chap instance at
build time.
