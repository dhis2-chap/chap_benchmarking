# CHAP benchmarking

Runs the standing model benchmarks against a chap-core instance and keeps
nothing itself: results live in chap-core's database and are read back through
its REST API.

## How it fits together

A benchmark problem (`problem_specifications.yaml`) is a human-chosen name for a
dataset plus the backtest parameters. chap-core deduplicates the (dataset,
parameters) tuple into one `BacktestSpecification`, so every backtest a problem
produces lands under one specification and is comparable by construction. This
repo owns what to run and when; chap-core owns the results and knows nothing
about problem names.

The models are not listed here. The [model marketplace](https://github.com/dhis2-chap/model-marketplace)
is the source of truth: `chap-admin install-all` and `chap-admin update-all`
(from chap-core) keep the server's chap in step with it, and the runner
benchmarks every live configured model whose period type and covariates the
problem's dataset satisfies (covariates chap generates itself, prefixed `gen:`,
are not required of the dataset). A model is identified by its configured model
row in chap-core, which is immutable per name, version and configuration, and is
pending for a problem when it has no backtest under the problem's specification.
Installing a model, or updating it to a new stable version, is therefore what
triggers a run. A model whose last job failed is held back until `run --force`.
The pending models of a problem are submitted in one call to
`POST /v1/analytics/create-backtests`.

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
5. `uv run run_benchmarks.py status` shows each problem's specification, the models its dataset can run, and which are pending or failed.
6. `uv run run_benchmarks.py run` runs every pending model. `--problem NAME` limits to one problem, `--force` reruns every model.
7. `uv run run_benchmarks.py results NAME` prints the backtests under a problem with version, source digest and aggregate metrics.

## Server

The benchmarking server (`chap-benchmarking.dhis2.org`, runbook in the private
[climate-sre](https://github.com/dhis2-chap/climate-sre) repository under
`hosts/chap-benchmarking/`) keeps this repo in `/opt/dhis2/chap_benchmarking` with
the live configuration in `config/` and `CHAP_URL` / `CHAP_API_TOKEN` in `.env`,
both untracked. `deploy.sh` runs on push to main (see
`.github/workflows/deploy.yml`) and installs a cron job that runs
`check_updates_and_trigger_run.py` every 15 minutes. A separate cron job on the
host runs `chap-admin install-all` and `update-all` against the server's chap, so
new marketplace models and versions are picked up by the next run.

To trigger a run by hand:

```bash
cd /opt/dhis2/chap_benchmarking
set -a; . ./.env; set +a
.venv/bin/python check_updates_and_trigger_run.py
```

Results are read by the model marketplace from
`GET /v1/crud/backtest-specifications/{id}` on the server's chap instance,
`https://chap-benchmarking.dhis2.org`, with the API token.
