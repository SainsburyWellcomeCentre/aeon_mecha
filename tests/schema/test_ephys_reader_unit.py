"""Unit tests for aeon/schema/ephys.py readers."""

import csv

import pytest

from aeon.schema.ephys import HarpSyncModel

pytestmark = pytest.mark.unit


@pytest.fixture
def synthetic_harp_csv(tmp_path):
    """Write a HarpSync CSV with three known sync samples and return its path."""
    csv_path = tmp_path / "NeuropixelsV2Beta_HarpSync_2024-06-04T11-00-00.csv"
    rows = [
        {"clock": 1000, "hub_clock": 0, "harp_time": 3000.5},
        {"clock": 2000, "hub_clock": 1, "harp_time": 3001.5},
        {"clock": 3000, "hub_clock": 2, "harp_time": 3002.5},
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clock", "hub_clock", "harp_time"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def test_read_returns_observed_harp_bounds_and_sample_count(synthetic_harp_csv):
    reader = HarpSyncModel.Reader(pattern="NeuropixelsV2Beta")
    df = reader.read(synthetic_harp_csv)

    assert len(df) == 1
    row = df.iloc[0]
    assert int(row["clock_start"]) == 1000
    assert int(row["clock_end"]) == 3000
    assert float(row["harp_start"]) == pytest.approx(3000.5)
    assert float(row["harp_end"]) == pytest.approx(3002.5)
    assert int(row["n_samples"]) == 3
    assert row["model"] is not None
    assert 0.0 <= float(row["r2"]) <= 1.0


def test_read_drops_na_rows_before_counting(synthetic_harp_csv, tmp_path):
    """Rows with NaN are dropped; n_samples reflects post-dropna count."""
    csv_path = tmp_path / "NeuropixelsV2Beta_HarpSync_2024-06-04T12-00-00.csv"
    with open(synthetic_harp_csv) as src, open(csv_path, "w") as dst:
        dst.write(src.read())
        # Append a row with empty harp_time → NaN after parse
        dst.write("4000,3,\n")

    reader = HarpSyncModel.Reader(pattern="NeuropixelsV2Beta")
    df = reader.read(csv_path)

    row = df.iloc[0]
    assert int(row["n_samples"]) == 3
    assert int(row["clock_end"]) == 3000  # last non-NaN row's clock
