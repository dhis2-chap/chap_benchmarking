#!/usr/bin/env python3
"""
Benchmark Pipeline Script

This script runs benchmarks for given models by:
1. Git pulling the model repository
2. Discovering model configurations and adding them to the database
3. Running benchmarks for each problem-spec and model-config combination
4. Logging results to the database

Assumes CHAP core server is running on localhost.
"""
import csv
import datetime
import logging
import pydantic
import cyclopts
import glob
import yaml
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import pandas
import altair as alt
alt.renderers.enable('browser')
LOG_FILENAME = 'logged_runs.csv'

logging.basicConfig(level=logging.INFO, )

# CHAP API base URL
CHAP_API_BASE = "http://localhost:8000/v1"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataSetModelMapping(pydantic.BaseModel):
    dataset_name: str
    model_slug: str


class BackTestSetting(pydantic.BaseModel):
    stride: int = 1
    n_periods: int = 3
    prediction_length: int = 6


class ProblemSpec(pydantic.BaseModel):
    name: str
    dataset_name: str
    backtest_settings: BackTestSetting
    metrics: list[str]

class LoggedRun(pydantic.BaseModel):
    timestamp: datetime.datetime
    backtest_id: int #TODO: Add all metrics
    model_slug: str
    problem_spec_name: str
    metric_name: str
    metric_value: float
    chap_version: str = ''
    model_commit_hash: str = ''


class ChapAPIClient:
    """Client for interacting with CHAP API"""

    def __init__(self, base_url: str = CHAP_API_BASE):
        self.base_url = base_url
        self.session = requests.Session()

    def health_check(self) -> bool:
        """Check if CHAP server is running"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def wait_for_job_completion(self, job_id: str, timeout: int = 3600) -> bool:
        """Wait for a job to complete"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.get_job_status(job_id)

            if status is None:
                logger.error(f"Failed to get status for job {job_id}")
                return False

            if status.lower() in ['completed', 'success']:
                logger.info(f"Job {job_id} completed successfully")
                return self.get_db_id(job_id)
            elif status.lower() in ['failed', 'error', 'failure']:
                logger.error(f"Job {job_id} failed")
                logs = self.session.get(f"{self.base_url}/jobs/{job_id}/logs").text
                print(logs)
                raise Exception(f"Job {job_id} failed with status: {status}")

            logger.info(f"Job {job_id} status: {status}")
            time.sleep(30)  # Wait 30 seconds before checking again

        logger.error(f"Job {job_id} timed out after {timeout} seconds")
        return False

    def add_configured_model(self, model_config: Dict) -> Optional[int]:
        """Add a configured model to the database"""
        try:
            response = self.session.post(
                f"{self.base_url}/crud/configured-models",
                json=model_config
            )
            if response.status_code == 200:
                result = response.json()
                return result.get('id')
            else:
                logger.error(f"Failed to add model config: {response.status_code} - {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error adding model config: {e}")
            return None

    def create_backtest(self, backtest_config: Dict) -> Optional[str]:
        """Create a backtest job"""
        response = self.session.post(
            f"{self.base_url}/analytics/create-backtest",
            json=backtest_config
        )
        if response.status_code != 200:
            logger.error(f"Failed to create backtest: {response.status_code} - {response.text}")
            raise Exception(f"Failed to create backtest: {response.status_code} - {response.text}")
        result = response.json()
        job_id =  result.get('id')
        db_id = self.wait_for_job_completion(job_id, timeout=3600)  # Wait for the job to complete
        backtest = self.get('backtests', db_id)
        return job_id

    def get_job_status(self, job_id: str) -> Optional[str]:
        """Get job status"""
        try:
            response = self.session.get(f"{self.base_url}/jobs/{job_id}")
            if response.status_code == 200:
                return response.text.strip('"')
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting job status: {e}")
            return None

    def get_datasets(self) -> List[Dict]:
        """Get available datasets"""
        try:
            response = self.session.get(f"{self.base_url}/crud/datasets")
            if response.status_code == 200:
                return response.json()
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting datasets: {e}")
            return []

    def get_model_templates(self) -> List[Dict]:
        """Get available model templates"""
        return self.list_entries('model-templates')

    def list_entries(self, name):
        try:
            response = self.session.get(f"{self.base_url}/crud/{name}")
            if response.status_code == 200:
                return response.json()
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting model templates: {e}")
            return []

    def get(self, name, db_id):
        response = self.session.get(f"{self.base_url}/crud/{name}/{db_id}")
        if response.status_code != 200:
            logger.error(f"Failed to get {name} with ID {db_id}: {response.status_code} - {response.text}")
            raise Exception(f"Failed to get {name} with ID {db_id}: {response.status_code} - {response.text}")
        return response.json()

    def get_db_id(self, job_id):
        """Get database ID from job ID"""
        response = self.session.get(f"{self.base_url}/jobs/{job_id}/database_result")
        assert response.status_code == 200, f"Failed to get job {job_id}: {response.status_code} - {response.text}"
        return response.json().get('id')



class BenchmarkRunner:
    """Main benchmark runner class"""

    def __init__(self, models_dir: str = "models", problem_config_file: str = "problem_config_mapping.yaml"):
        self.models_dir = Path(models_dir)
        self.problem_config_file = Path(problem_config_file)
        self.api_client = ChapAPIClient()

    def run_from_mapping(self, config_filename, template_name, problem_specs) -> ProblemSpec:
        config = yaml.safe_load(open(config_filename, 'r'))
        global_models = config.get('global_models', [])
        if template_name:
            global_models = [m for m in global_models if m.startswith(template_name)]
        configured_models = self.api_client.list_entries('configured-models')
        model_mapping = {model['name']: model for model in configured_models if model['name'] in global_models}
        data_sets = self.api_client.get_datasets()
        logged_runs = []
        for problem_spec in problem_specs:
            logger.info(f"Processing problem spec: {problem_spec.name}")
            dataset_name = problem_spec.dataset_name
            dataset = next((ds for ds in data_sets if ds['name'] == dataset_name), None)
            if not dataset:
                raise ValueError(f"Dataset {dataset_name} not found in CHAP datasets: {[ds['name'] for ds in data_sets]}")
            logger.info(f"Running benchmarks for dataset: {dataset_name}")
            for model_name, models_conf in model_mapping.items():
                logger.info(f"    Running benchmark for model: {model_name}")
                settings_dict = {'stride': problem_spec.backtest_settings.stride,
                                 'nSplits': problem_spec.backtest_settings.n_periods,
                                 'nPeriods': problem_spec.backtest_settings.prediction_length}
                job_id = self.api_client.create_backtest(
                    settings_dict | {'modelId': model_name, 'datasetId': dataset['id'],
                                     'name': f"{model_name}_{dataset_name}_benchmark"})
                assert job_id is not None
                self.api_client.wait_for_job_completion(job_id)
                backtest_id = self.api_client.get_db_id(job_id)
                backtest = self.api_client.get('backtests', backtest_id)
                for metric_name, metric_value in backtest['aggregateMetrics'].items():
                    logged_runs.append(
                        LoggedRun(timestamp=datetime.datetime.now(),
                        backtest_id=backtest_id,
                        model_slug=model_name,
                        model_commit_hash='',
                        problem_spec_name=problem_spec.name,
                        metric_name=metric_name,
                        metric_value=metric_value))
        return logged_runs

    def discover_model_configs(self, model_slug: str) -> List[Dict]:
        """Discover model configurations in the model directory"""
        model_path = self.models_dir / model_slug
        configs_path = model_path / "configs"

        if not configs_path.exists():
            logger.warning(f"No configs directory found in {model_path}")
            return []

        configs = []
        config_files = glob.glob(str(configs_path / "*.yaml")) + glob.glob(str(configs_path / "*.yml"))

        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)

                config_name = Path(config_file).stem

                # Create model configuration for CHAP API
                model_config = {
                    "name": f"{model_slug}_{config_name}",
                    "modelTemplateId": 1,  # Default template ID, should be configurable
                    "userOptionValues": config_data,
                    "additionalContinuousCovariates": []
                }

                configs.append({
                    "config_name": config_name,
                    "config_data": config_data,
                    "api_config": model_config
                })

            except Exception as e:
                logger.error(f"Error reading config file {config_file}: {e}")
                continue

        logger.info(f"Discovered {len(configs)} configurations for {model_slug}")
        return configs

    def load_problem_config_mapping(self) -> Dict:
        """Load problem-config mapping from YAML file"""
        if not self.problem_config_file.exists():
            logger.error(f"Problem config file {self.problem_config_file} does not exist")
            return {}

        try:
            with open(self.problem_config_file, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading problem config mapping: {e}")
            return {}


    def run_benchmark_for_config(self, model_slug: str, config_name: str,
                                 model_id: int, problem_specs: List[str]) -> List[bool]:
        """Run benchmark for a specific model configuration against problem specs"""
        results = []
        datasets = self.api_client.get_datasets()

        if not datasets:
            logger.error("No datasets available for benchmarking")
            return []

        for problem_spec in problem_specs:
            # Find matching dataset
            dataset = None
            for ds in datasets:
                if problem_spec in ds.get('name', '').lower():
                    dataset = ds
                    break

            if not dataset:
                logger.warning(f"No dataset found for problem spec: {problem_spec}")
                results.append(False)
                continue

            # Create backtest configuration
            backtest_config = {
                "nPeriods": 6,  # Default, should be configurable
                "nSplits": 3,  # Default, should be configurable
                "stride": 1,  # Default, should be configurable
                "name": f"{model_slug}_{config_name}_{problem_spec}_benchmark",
                "modelId": str(model_id),
                "datasetId": dataset['id']
            }

            logger.info(f"Running benchmark: {model_slug}/{config_name} on {problem_spec}")

            # Create and run backtest
            job_id = self.api_client.create_backtest(backtest_config)

            if job_id:
                success = self.api_client.wait_for_job_completion(job_id)
                results.append(success)

                if success:
                    logger.info(f"Benchmark completed: {model_slug}/{config_name} on {problem_spec}")
                else:
                    logger.error(f"Benchmark failed: {model_slug}/{config_name} on {problem_spec}")
            else:
                logger.error(f"Failed to start benchmark: {model_slug}/{config_name} on {problem_spec}")
                results.append(False)

        return results

    def run_benchmarks_for_model(self, model_slug: str) -> bool:
        """Run all benchmarks for a given model"""
        logger.info(f"Starting benchmarks for model: {model_slug}")

        # Check CHAP server health
        if not self.api_client.health_check():
            logger.error("CHAP server is not running or not accessible")
            return False

        # Git pull the model
        if not self.git_pull_model(model_slug):
            logger.error(f"Failed to update model {model_slug}")
            return False

        # Discover model configurations
        configs = self.discover_model_configs(model_slug)
        if not configs:
            logger.error(f"No configurations found for model {model_slug}")
            return False

        # Load problem configuration mapping
        problem_mapping = self.load_problem_config_mapping()
        problem_specs = problem_mapping.get(model_slug, [])

        if not problem_specs:
            logger.warning(f"No problem specs defined for model {model_slug}")
            # Use all available datasets as fallback
            datasets = self.api_client.get_datasets()
            problem_specs = [ds['name'] for ds in datasets]

        # Process each configuration
        overall_success = True

        for config in configs:
            logger.info(f"Processing configuration: {config['config_name']}")

            # Add configuration to database
            model_id = self.api_client.add_configured_model(config['api_config'])

            if not model_id:
                logger.error(f"Failed to add configuration {config['config_name']} to database")
                overall_success = False
                continue

            # Run benchmarks for this configuration
            results = self.run_benchmark_for_config(
                model_slug,
                config['config_name'],
                model_id,
                problem_specs
            )

            if not all(results):
                logger.error(f"Some benchmarks failed for configuration {config['config_name']}")
                overall_success = False

        return overall_success

    def plot_logs(self, log_entries: list[LoggedRun]):
        full_entries = [e.model_dump() for e in log_entries]
        # for entry in log_entries:
        #     db_entry = self.api_client.get('backtests', entry.backtest_id)
        #     metrics = db_entry['aggregateMetrics']
        #     for metric_name, value in metrics.items():
        #         full_entries.append(entry.model_dump() | {'metric_name': metric_name, 'metric_value': value})
        df = pandas.DataFrame(full_entries)
        metrics = sorted(df['metric_name'].unique().tolist())
        dropdown = alt.binding_select(options=metrics, name='Metric: ')
        met = alt.param('met', bind=dropdown, value=metrics[0])

        problems = sorted(df['problem_spec_name'].unique().tolist())
        dropdown = alt.binding_select(options=problems, name='Problem: ')
        prob = alt.param('prob', bind=dropdown, value=problems[0])

        chart = alt.Chart(df).mark_line(interpolate='step-before').encode(
            x='timestamp',
            y='metric_value',
            color='model_slug',
        ).add_params(met).add_params(prob).transform_filter(alt.datum.metric_name == met).transform_filter(alt.datum.problem_spec_name == prob)
        # save plot to file
        chart.save('benchmark_plot.html')

        # chart.show()


def test_plot_logfile():
    log_filename = LOG_FILENAME

    log_entries = _read_log_entries(log_filename)
    runner = BenchmarkRunner()
    runner.plot_logs(log_entries)


def _read_log_entries(log_filename):
    logger.info(f"Reading log entries from {log_filename}")
    with open(log_filename, 'r') as f:
        reader = csv.DictReader(f)
        log_entries = [LoggedRun(**row) for row in reader]
    return log_entries


def test_benchmark_runner():
    problem_spec_filename = 'problem_specifications.yaml'
    mapping_filename = 'dataset_model_maps.yaml'
    out_file = LOG_FILENAME
    logged_runs = run_benchmarks(mapping_filename, problem_spec_filename, out_file)

    assert False, logged_runs


def run_benchmarks(mapping_filename, problem_spec_filename, out_file):
    runner = BenchmarkRunner()
    T = List[ProblemSpec]
    problem_specs = parse_yaml(problem_spec_filename, T)
    logged_runs = runner.run_from_mapping(mapping_filename, template_name=None, problem_specs=problem_specs)
    if not Path(out_file).exists():
        header = ','.join(LoggedRun.model_fields.keys())
        with open(out_file, 'w') as f:
            f.write(f'{header}\n')
    with open(out_file, 'a') as f:
        for run in logged_runs:
            line = ','.join(str(getattr(run, field)) for field in LoggedRun.model_fields.keys())
            f.write(f'{line}\n')

    return logged_runs


def parse_yaml(file_name, data_type):
    problem_specs_data = yaml.safe_load(open(file_name, 'r'))  # Load problem specs from YAML file
    problem_specs = pydantic.parse_obj_as(data_type, problem_specs_data)
    logger.info(f'Found {len(problem_specs)} problem specifications in {file_name}')
    return problem_specs



def main(config_folder: Path=Path('./example_config/'), log_file: Path=Path('benchmark_log.csv')):
    """Main entry point"""
    run_benchmarks(
        mapping_filename=config_folder / 'dataset_model_maps.yaml',
        problem_spec_filename=config_folder / 'problem_specifications.yaml',
        out_file=log_file
    )
    log_entries = _read_log_entries(log_file)
    runner = BenchmarkRunner()
    runner.plot_logs(log_entries)


if __name__ == "__main__":
    cyclopts.run(main)
