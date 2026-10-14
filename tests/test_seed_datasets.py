import csv
import json

import pytest

from seed_datasets import convert, make_dataset_request


def test_make_dataset_request_turns_every_observation_column_into_provided_data(csv_rows, geojson):
    request = make_dataset_request(csv_rows, geojson)
    assert request["dataToBeFetched"] == []
    assert request["geojson"] == {"type": "FeatureCollection", "features": geojson["features"]}
    assert request["providedData"] == [
        {"featureName": "rainfall", "orgUnit": "LA-XA", "period": "1998-01", "value": 2.58},
        {"featureName": "disease_cases", "orgUnit": "LA-XA", "period": "1998-01", "value": 12.0},
        {"featureName": "rainfall", "orgUnit": "LA-XA", "period": "1998-02", "value": 0.4},
        {"featureName": "disease_cases", "orgUnit": "LA-XA", "period": "1998-02", "value": None},
    ]


def test_make_dataset_request_rejects_a_location_without_a_polygon(csv_rows, geojson):
    rows = csv_rows + [{**csv_rows[0], "location": "LA-ZZ"}]
    with pytest.raises(ValueError, match="LA-ZZ"):
        make_dataset_request(rows, geojson)


def test_convert_writes_the_request_for_a_csv_and_geojson_pair(tmp_path, csv_rows, geojson):
    csv_file = tmp_path / "data.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    geojson_file = tmp_path / "data.geojson"
    geojson_file.write_text(json.dumps(geojson))
    output_file = tmp_path / "request.json"

    convert(csv_file, geojson_file, output_file)

    assert json.loads(output_file.read_text()) == make_dataset_request(csv_rows, geojson)
