"""Measure actual blob sizes in ephys pipeline tables.

Queries the test DB to determine which longblob columns are large enough
to benefit from external storage (blob@dj_store).

Usage:
  uv run python -m aeon.dj_pipeline.scripts.measure_blob_sizes
"""

import datajoint as dj

from aeon.dj_pipeline import get_schema_name

TABLES = [
    {
        "schema": get_schema_name("spike_sorting"),
        "table": "__sorted_spikes__unit",
        "label": "SortedSpikes.Unit",
        "columns": ["spike_indices", "spike_sites", "spike_depths"],
    },
    {
        "schema": get_schema_name("spike_sorting"),
        "table": "__synced_spikes__unit",
        "label": "SyncedSpikes.Unit",
        "columns": ["spike_times"],
    },
    {
        "schema": get_schema_name("spike_sorting"),
        "table": "__unit_matching__spikes",
        "label": "UnitMatching.Spikes",
        "columns": ["spike_times"],
    },
    {
        "schema": get_schema_name("spike_sorting"),
        "table": "__waveform__unit_waveform",
        "label": "Waveform.UnitWaveform",
        "columns": ["unit_waveform"],
    },
    {
        "schema": get_schema_name("spike_sorting"),
        "table": "__waveform__channel_waveform",
        "label": "Waveform.ChannelWaveform",
        "columns": ["channel_waveform"],
    },
]


def measure():
    conn = dj.conn()
    print("=" * 80)
    print("Blob Size Report")
    print(f"Host: {dj.config['database.host']}")
    print("=" * 80)

    for spec in TABLES:
        schema_name = spec["schema"]
        table_name = spec["table"]
        full_name = f"`{schema_name}`.`{table_name}`"

        # Check the table exists and has rows
        try:
            n_rows = conn.query(f"SELECT COUNT(*) AS n FROM {full_name}").fetchone()[0]
        except Exception as e:
            print(f"\n{spec['label']}  --  SKIPPED ({e})")
            continue

        if n_rows == 0:
            print(f"\n{spec['label']}  --  0 rows")
            continue

        print(f"\n{spec['label']}  ({n_rows} rows)")
        print("-" * 60)
        print(f"  {'Column':<20} {'Avg (KB)':>10} {'Max (KB)':>10} {'Total (MB)':>12}")

        for col in spec["columns"]:
            sql = (
                f"SELECT "
                f"  AVG(LENGTH(`{col}`)) AS avg_bytes, "
                f"  MAX(LENGTH(`{col}`)) AS max_bytes, "
                f"  SUM(LENGTH(`{col}`)) / 1024.0 / 1024.0 AS total_mb "
                f"FROM {full_name}"
            )
            row = conn.query(sql).fetchone()
            avg_kb = (row[0] or 0) / 1024
            max_kb = (row[1] or 0) / 1024
            total_mb = row[2] or 0
            flag = " << EXTERNALIZE" if avg_kb > 100 else ""
            print(f"  {col:<20} {avg_kb:>10.1f} {max_kb:>10.1f} {total_mb:>12.2f}{flag}")

    print("\n" + "=" * 80)
    print("Columns averaging >100 KB are flagged for blob@dj_store externalization.")
    print("=" * 80)


if __name__ == "__main__":
    measure()
