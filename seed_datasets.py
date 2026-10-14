"""Seed a chap-core instance with the datasets the benchmark problems run against.

The seeding file maps a dataset name to a JSON file holding a make-dataset
request body (geojson plus provided observations). Datasets that already exist
in chap are skipped, so the command is safe to rerun. `convert` builds such a
request body from a CSV and GeoJSON pair as published in dhis2/climate-health-data.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import cyclopts
import yaml

from chap_client import ChapClient

logger = logging.getLogger(__name__)

app = cyclopts.App(name="seed-datasets", help="Seed chap with datasets. Reads CHAP_URL and CHAP_API_TOKEN.")

# CSV columns that index a row rather than hold an observation.
INDEX_COLUMNS = ("time_period", "location", "location_name")


def make_dataset_request(rows: list[dict], geojson: dict) -> dict:
    """Build a make-dataset request body from CSV rows and the GeoJSON of their locations.

    Every column except the index columns becomes a feature, with an empty cell as null.
    Chap matches polygons on the feature id, so every location in the CSV must have one.
    The GeoJSON `crs` key is dropped: chap only reads the features.
    """
    feature_ids = {feature.get("id") for feature in geojson["features"]}
    missing = sorted({row["location"] for row in rows} - feature_ids)
    if missing:
        raise ValueError(f"Locations without a polygon in the GeoJSON: {missing}")
    provided = [
        {
            "featureName": column,
            "orgUnit": row["location"],
            "period": row["time_period"],
            "value": None if value == "" else float(value),
        }
        for row in rows
        for column, value in row.items()
        if column not in INDEX_COLUMNS
    ]
    return {
        "geojson": {key: value for key, value in geojson.items() if key != "crs"},
        "providedData": provided,
        "dataToBeFetched": [],
    }


@app.command
def convert(csv_file: Path, geojson_file: Path, output_file: Path):
    """Write the make-dataset request body for a CSV and GeoJSON pair, ready for the seeding file."""
    with open(csv_file, newline="") as f:
        rows = list(csv.DictReader(f))
    with open(geojson_file) as f:
        geojson = json.load(f)
    request = make_dataset_request(rows, geojson)
    with open(output_file, "w") as f:
        json.dump(request, f)
    logger.info("Wrote %s observations to %s", len(request["providedData"]), output_file)


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
