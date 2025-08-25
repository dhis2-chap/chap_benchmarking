#!/usr/bin/env python3
"""
Dataset Seeding Script

This script seeds the CHAP database with datasets using the REST API.

Supports multiple methods for dataset creation:
1. JSON dataset creation via /crud/datasets
2. CSV file upload via /crud/datasets/csvFile  
3. Advanced dataset creation via /analytics/make-dataset

Assumes CHAP core server is running on localhost.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Literal
import requests
import time
import yaml
import cyclopts

# Reuse the ChapAPIClient from run_benchmarks
from run_benchmarks import ChapAPIClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = cyclopts.App(name="seed-datasets", help="Seed CHAP database with datasets")


class DatasetSeeder:
    """Main dataset seeding class"""
    
    def __init__(self, base_url: str = "http://localhost:8000/v1"):
        self.api_client = ChapAPIClient(base_url)
    
    def create_dataset_from_json(self, dataset_config: Dict) -> Optional[str]:
        """Create dataset using JSON configuration via /crud/datasets"""
        print(dataset_config.keys())
        response = self.api_client.session.post(
            f"{self.api_client.base_url}/analytics/make-dataset",
            json=dataset_config
        )
        if response.status_code == 200:
            result = response.json()
            return result.get('id')
        else:
            logger.error(f"Failed to create dataset: {response.status_code} - {response.text[:500]}")
            raise Exception(f"Failed to create dataset: {response.status_code} - {response.text[:500]}")


    def make_dataset_with_fetching(self, dataset_config: Dict) -> Optional[str]:
        """Create dataset with data fetching via /analytics/make-dataset"""
        try:
            response = self.api_client.session.post(
                f"{self.api_client.base_url}/analytics/make-dataset",
                json=dataset_config
            )
            if response.status_code == 200:
                result = response.json()
                return result.get('id')
            else:
                logger.error(f"Failed to make dataset: {response.status_code} - {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making dataset: {e}")
            return None
    
    def wait_for_job_completion(self, job_id: str, timeout: int = 1800) -> bool:
        """Wait for an async job to complete"""
        if not job_id:
            return False
            
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.api_client.get_job_status(job_id)
            
            if status is None:
                logger.error(f"Failed to get status for job {job_id}")
                return False
            
            if status.lower() in ['completed', 'success']:
                logger.info(f"Job {job_id} completed successfully")
                return True
            elif status.lower() in ['failed', 'error']:
                logger.error(f"Job {job_id} failed")
                return False
            
            logger.info(f"Job {job_id} status: {status}")
            time.sleep(10)  # Wait 10 seconds before checking again
        
        logger.error(f"Job {job_id} timed out after {timeout} seconds")
        return False
    
    def load_dataset_config(self, config_file: str) -> Optional[Dict]:
        """Load dataset configuration from JSON or YAML file"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            logger.error(f"Config file not found: {config_file}")
            return None
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config file {config_file}: {e}")
            return None
    
    def seed_dataset(self, config_file: str, name: str) -> bool:
        """Seed a single dataset"""
        logger.info(f"Seeding dataset from {config_file} using with name '{name}'")

        # Load and validate basic dataset config
        config = self.load_dataset_config(config_file)
        config['name']= name
        print(config['name'])
        print(config.keys())
        if not self.api_client.health_check():
            logger.error("CHAP server is not running or not accessible")
            return False

        if not config:
            raise ValueError(f"Invalid dataset configuration in {config_file}")

        job_id = self.create_dataset_from_json(config)

        if job_id:
            # Wait for job completion if it's async
            success = self.wait_for_job_completion(job_id)
            if success:
                logger.info(f"Successfully created dataset via job: {job_id}")
            return success
        else:
            logger.error("Failed to create dataset from JSON")
            raise ValueError("Failed to create dataset from JSON")

    def list_existing_datasets(self) -> List[Dict]:
        """List all existing datasets in the database"""
        try:
            datasets = self.api_client.get_datasets()
            logger.info(f"Found {len(datasets)} existing datasets:")
            for ds in datasets:
                logger.info(f"  - {ds.get('name')} (ID: {ds.get('id')}, Type: {ds.get('type')})")
            return datasets
        except Exception as e:
            logger.error(f"Error listing datasets: {e}")
            return []


@app.command
def list_datasets(verbose: bool = False):
    """List all existing datasets in the database.
    
    Args:
        verbose: Enable verbose logging
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    seeder = DatasetSeeder()
    seeder.list_existing_datasets()

@app.command
def seed(seeding_yaml_filename: str):
    """Seed datasets using a YAML configuration file."""
    if not os.path.exists(seeding_yaml_filename):
        logger.error(f"Seeding YAML file not found: {seeding_yaml_filename}")
        sys.exit(1)

    with open(seeding_yaml_filename, 'r') as f:
        seeding_config = yaml.safe_load(f)
    print(seeding_config)
    seeder = DatasetSeeder()
    existing_names = {ds['name'] for ds in seeder.list_existing_datasets()}
    print(existing_names)
    for name, file_path in seeding_config.items():
        if name in existing_names:
            logger.info(f"Dataset '{name}' already exists, skipping.")
            continue

        if not os.path.exists(file_path):
            logger.error(f"Dataset file not found: {file_path}")
            continue

        success = seeder.seed_dataset(file_path, name)

        if success:
            logger.info(f"Successfully seeded dataset '{name}' from {file_path}")
        else:
            logger.error(f"Failed to seed dataset '{name}' from {file_path}")



if __name__ == "__main__":
    app()