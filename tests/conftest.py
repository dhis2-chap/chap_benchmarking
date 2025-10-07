from pathlib import Path

import pytest


@pytest.fixture
def data_path():
    return Path(__file__).parent / "data"

@pytest.fixture
def benchmark_log_file(data_path):
    return data_path / "benchmark_log.csv"
