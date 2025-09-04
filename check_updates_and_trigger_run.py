"""
For each model, check if any of the models have been updated on github.

If any is updated, trigger a benchmark run with that model.
"""

from pathlib import Path
from github_tools import get_last_commit_hash
from run_benchmarks import BenchmarkRunner, _read_log_entries, run_benchmarks
import pandas
import cyclopts
import os
import sys
import time


def check_for_updates(config_folder: Path, log_file: Path):
    """Check for model updates and trigger benchmarks if needed (core logic without locking)."""
    runner = BenchmarkRunner()
    mapping_filename = config_folder / 'dataset_model_maps.yaml'
    problem_spec_filename = config_folder / 'problem_specifications.yaml'
    out_file = log_file
    models = runner.get_models(mapping_filename, template_name=None)
    
    print(f"\nChecking {len(models)} models for updates:\n")
    
    last_commits = {}
    
    for model_name, model in models.items():
        source_url = model['sourceUrl']
        try:
            commit_hash = get_last_commit_hash(source_url)
            last_commits[model_name] = commit_hash
            print(f"✓ {model_name}: Last commit {commit_hash})")
        except Exception as e:
            print(f"✗ {model_name}: Failed - {e}")

    log_entries = _read_log_entries(log_file)
    full_entries = [e.model_dump() for e in log_entries]
    df = pandas.DataFrame(full_entries)
    df_latest = df.sort_values('timestamp').groupby('model_slug').tail(1)

    for model_name, last_commit in last_commits.items():
        if last_commit is None:
            print(f"✗ {model_name}: No commit info, skipping")
            continue
        else:
            latest_entry = df_latest[df_latest['model_slug'] == model_name]
            if latest_entry.empty:
                print(f"⚠ {model_name}: No previous run found, triggering benchmark")
                run_benchmarks(mapping_filename, problem_spec_filename, out_file, template_name=model_name)
            else:
                latest_commit = latest_entry.iloc[0]['model_commit_hash']
                if latest_commit != last_commit:
                    print(f"⚠ {model_name}: Updated from {latest_commit} to {last_commit}, triggering benchmark")
                    run_benchmarks(mapping_filename, problem_spec_filename, out_file, template_name=model_name)
                else:
                    print(f"✓ {model_name}: No update (last commit {latest_commit})")


def main(config_folder: Path=Path('./example_config/'), log_file: Path=Path('benchmark_log.csv')):
    """
    Checks for updates to models and reruns benchmarks if needed.
    This function is typically meant to be run from a server often, therefore it is using lock logic to avoid simultaneous runs.
    """

    # Lock file path
    lock_file = Path('/tmp/check_updates_and_trigger_run.lock')
    
    # Check if lock file exists
    if lock_file.exists():
        # Check if lock file is stale (older than 60 minutes)
        lock_age = time.time() - lock_file.stat().st_mtime
        if lock_age < 3600:  # 60 minutes
            print(f"Script is already running (lock file exists and is {int(lock_age)} seconds old). Exiting.")
            sys.exit(0)
        else:
            print(f"Removing stale lock file (age: {int(lock_age)} seconds)")
            lock_file.unlink()
    
    # Create lock file
    try:
        lock_file.touch()
        
        # Run the main update checking logic
        check_for_updates(config_folder, log_file)
    
    finally:
        # Always remove lock file when done
        if lock_file.exists():
            lock_file.unlink()
            print("\nLock file removed.")


if __name__ == "__main__":
    cyclopts.run(main)

