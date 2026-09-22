"""Seed a chap-core instance with the datasets the benchmark problems run against.

The seeding file maps a dataset name to a JSON file holding a make-dataset
request body (geojson plus provided observations). Datasets that already exist
in chap are skipped, so the command is safe to rerun.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cyclopts
import yaml

from chap_client import ChapClient

logger = logging.getLogger(__name__)

app = cyclopts.App(name="seed-datasets", help="Seed chap with datasets. Reads CHAP_URL and CHAP_API_TOKEN.")


def seed_dataset(client: ChapClient, name: str, request_file: Path) -> int:
    """Import one dataset and return its id."""
    with open(request_file) as f:
        request = json.load(f)
    request["name"] = name
    job_id = client.make_dataset(request)
    logger.info("Dataset %s submitted as job %s", name, job_id)
    return client.wait_for_job(job_id, poll_interval=10)


@app.command
def list_datasets():
    """List the datasets chap holds."""
    for dataset in ChapClient.from_env().list_datasets():
        print(f"{dataset['id']}\t{dataset['name']}\t{dataset.get('type')}")


@app.command
def seed(seeding_yaml_filename: Path):
    """Seed every dataset in the YAML file that chap does not already have."""
    client = ChapClient.from_env()
    with open(seeding_yaml_filename) as f:
        seeds: dict[str, str] = yaml.safe_load(f)
    existing = {dataset["name"] for dataset in client.list_datasets()}
    for name, request_file in seeds.items():
        if name in existing:
            logger.info("Dataset %s already exists, skipping", name)
            continue
        dataset_id = seed_dataset(client, name, seeding_yaml_filename.parent / request_file)
        logger.info("Dataset %s created with id %s", name, dataset_id)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app()


if __name__ == "__main__":
    main()
