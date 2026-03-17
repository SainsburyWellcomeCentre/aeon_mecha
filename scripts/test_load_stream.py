"""Prototype: compare blob-based vs codec-based stream data storage.

Spins up a testcontainers MySQL, sets up the full pipeline, then:
1. Populates the existing FeederEncoder table (blob approach — current)
2. Creates a FeederEncoderCodec table:
   - individual columns store JSON summary stats (min, max, mean, dtype)
   - timestamps stores JSON time range + sampling rate
   - stream_df uses <aeon_stream> codec → returns full DataFrame on fetch
3. Fetches from both and asserts identical data via stream_df

Usage:
    uv run python scripts/test_load_stream.py
"""

import datetime
import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Spin up MySQL via testcontainers
# ---------------------------------------------------------------------------
logger.info("Starting MySQL container via testcontainers...")
from testcontainers.mysql import MySqlContainer

container = MySqlContainer(image="mysql:8.0", username="root", password="test_password", dbname="test_db")
container.start()

host = container.get_container_host_ip()
port = container.get_exposed_port(3306)
os.environ["DJ_HOST"] = host
os.environ["DJ_PORT"] = str(port)
os.environ["DJ_USER"] = "root"
os.environ["DJ_PASS"] = "test_password"
logger.info(f"MySQL container ready at {host}:{port}")

try:
    # ---------------------------------------------------------------------------
    # 2. Configure DataJoint BEFORE importing pipeline
    # ---------------------------------------------------------------------------
    import datajoint as dj

    TEST_DB_PREFIX = "test_aeon_"
    dj.config.safemode = False
    dj.config.database.host = host
    dj.config.database.port = int(port)
    dj.config.database.user = "root"
    dj.config.database.password = "test_password"
    dj.config.database.database_prefix = TEST_DB_PREFIX

    # =========================================================================
    # 3. Define the AeonStreamCodec BEFORE importing pipeline
    # =========================================================================

    class AeonStreamCodec(dj.Codec):
        """Codec for lazy-loading stream data from raw files.

        Only used for the `stream_df` column. On fetch, reconstructs the
        stream reader from stored metadata and calls io_api.load() to return
        the full DataFrame.

        Stored JSON format:
        {
            "stream_type": "Encoder",
            "experiment_name": "abcBehav0-aeon3",
            "device_name": "Feeder1",
            "chunk_start": "2025-11-18 10:13:15",
            "chunk_end": "2025-11-18 11:00:00",
            "epoch_start": "2025-11-18 10:13:15"
        }
        """

        name = "aeon_stream"

        def get_dtype(self, is_store: bool) -> str:
            return "json"

        _REQUIRED_KEYS = {
            "stream_type",
            "experiment_name",
            "device_name",
            "chunk_start",
            "chunk_end",
            "epoch_start",
        }

        def encode(self, value, *, key=None, store_name=None):
            if not isinstance(value, dict):
                raise TypeError(f"AeonStreamCodec expects a dict, got {type(value).__name__}")
            missing = self._REQUIRED_KEYS - value.keys()
            if missing:
                raise ValueError(f"AeonStreamCodec missing required keys: {missing}")
            return value

        def decode(self, stored, *, key=None):
            from swc.aeon.io import api as io_api

            from aeon.dj_pipeline import acquisition
            from aeon.dj_pipeline.utils.load_metadata import get_stream_reader_for_epoch

            data_dirs = acquisition.Experiment.get_data_directories(
                {"experiment_name": stored["experiment_name"]}
            )
            stream_reader = get_stream_reader_for_epoch(
                stored["experiment_name"],
                stored["device_name"],
                stored["stream_type"],
                stored["epoch_start"],
            )
            return io_api.load(
                root=data_dirs,
                reader=stream_reader,
                start=pd.Timestamp(stored["chunk_start"]),
                end=pd.Timestamp(stored["chunk_end"]),
            )

    logger.info("AeonStreamCodec registered")

    # =========================================================================
    # Helper: compute summary stats for a column
    # =========================================================================

    def column_stats(values):
        """Compute JSON-serializable summary stats for a numpy array."""
        stats = {"dtype": str(values.dtype), "count": int(len(values))}
        if np.issubdtype(values.dtype, np.number):
            stats["min"] = float(np.nanmin(values))
            stats["max"] = float(np.nanmax(values))
            stats["mean"] = round(float(np.nanmean(values)), 4)
        return stats

    def timestamp_stats(index):
        """Compute JSON-serializable summary stats for a datetime index."""
        stats = {
            "min": str(index.min()),
            "max": str(index.max()),
            "count": int(len(index)),
        }
        if len(index) > 1:
            diffs = np.diff(index.values).astype(float)  # nanoseconds
            median_diff_ns = float(np.median(diffs))
            stats["sampling_rate_hz"] = round(1e9 / median_diff_ns, 2) if median_diff_ns > 0 else None
            stats["median_dt_ms"] = round(median_diff_ns / 1e6, 4)
        return stats

    # Now import pipeline — schemas activate with test prefix
    import aeon.dj_pipeline as pipeline
    from aeon.dj_pipeline import get_schema_name

    # ---------------------------------------------------------------------------
    # 4. Golden dataset config + pipeline setup
    # ---------------------------------------------------------------------------
    GOLDEN_DATA_ROOT = Path.home() / "sciops-data/project_aeon/aeon/data"
    pipeline.repository_config = {"ceph_aeon": str(GOLDEN_DATA_ROOT)}

    cfg = {
        "experiment_name": "abcBehav0-aeon3",
        "experiment_path": "AEON3/abcBehav0",
        "epoch_dir": "2025-11-18T10-13-15",
        "devices_schema": "swc.aeon_exp.foragingABC.experiment:Experiment",
        "arena_name": "arena-aeon3",
        "lab": "SWC",
        "location": "room-0",
        "experiment_type": "foraging",
    }

    epoch_path = GOLDEN_DATA_ROOT / "raw" / cfg["experiment_path"] / cfg["epoch_dir"]
    assert epoch_path.exists(), f"Golden dataset not found: {epoch_path}"

    from aeon.dj_pipeline import acquisition, lab
    from aeon.dj_pipeline.utils import streams_maker
    from aeon.dj_pipeline.utils.load_metadata import (
        get_experiment_pydantic,
        get_stream_reader_for_epoch,
        populate_catalog_from_pydantic,
    )

    experiment_class = get_experiment_pydantic(cfg["devices_schema"])
    populate_catalog_from_pydantic(experiment_class)
    streams_module = streams_maker.main(create_tables=True)

    lab.Arena.insert1(
        {
            "arena_name": cfg["arena_name"],
            "arena_description": f"Arena for {cfg['experiment_name']}",
            "arena_shape": "circular",
            "arena_x_dim": 2.0,
            "arena_y_dim": 2.0,
            "arena_z_dim": 0.2,
        },
        skip_duplicates=True,
    )
    acquisition.DevicesSchema.insert1(
        {"devices_schema_name": cfg["devices_schema"]}, skip_duplicates=True
    )
    epoch_dt = datetime.datetime.strptime(cfg["epoch_dir"], "%Y-%m-%dT%H-%M-%S")
    acquisition.Experiment.insert1(
        {
            "experiment_name": cfg["experiment_name"],
            "experiment_start_time": epoch_dt,
            "experiment_description": f"Test: {cfg['experiment_name']}",
            "arena_name": cfg["arena_name"],
            "lab": cfg["lab"],
            "location": cfg["location"],
            "experiment_type": cfg["experiment_type"],
        },
        skip_duplicates=True,
    )
    acquisition.Experiment.DevicesSchema.insert1(
        {"experiment_name": cfg["experiment_name"], "devices_schema_name": cfg["devices_schema"]},
        skip_duplicates=True,
    )
    acquisition.Experiment.Directory.insert1(
        {
            "experiment_name": cfg["experiment_name"],
            "directory_type": "raw",
            "repository_name": "ceph_aeon",
            "directory_path": f"raw/{cfg['experiment_path']}",
        },
        skip_duplicates=True,
    )

    acquisition.Epoch.ingest_epochs(cfg["experiment_name"])
    acquisition.EpochConfig.populate()
    acquisition.Chunk.ingest_chunks(cfg["experiment_name"])
    logger.info("Pipeline setup complete")

    # ---------------------------------------------------------------------------
    # 5. Populate the EXISTING FeederEncoder table (blob approach)
    # ---------------------------------------------------------------------------
    feeder_encoder_table = None
    for attr_name in dir(streams_module):
        if "Feeder" in attr_name and "Encoder" in attr_name:
            feeder_encoder_table = getattr(streams_module, attr_name)
            break

    assert feeder_encoder_table is not None, "FeederEncoder table not found"
    logger.info(f"Populating blob-based {feeder_encoder_table.__name__}...")
    feeder_encoder_table.populate(max_calls=6, display_progress=True)

    # ---------------------------------------------------------------------------
    # 6. Create the CODEC-based table: JSON stats + stream_df codec
    # ---------------------------------------------------------------------------
    schema = dj.Schema(get_schema_name("streams"))

    # Get the reader columns for Encoder
    epoch_start = (acquisition.Epoch & {"experiment_name": cfg["experiment_name"]}).fetch1(
        "epoch_start"
    )
    encoder_reader = get_stream_reader_for_epoch(
        cfg["experiment_name"], "Feeder1", "Encoder", epoch_start
    )
    encoder_columns = [c for c in encoder_reader.columns if not c.startswith("_")]
    logger.info(f"Encoder columns: {encoder_columns}")

    # Build table definition:
    #   - sample_count: int (filterable)
    #   - timestamps: json (time range + sampling rate)
    #   - <data columns>: json (summary stats)
    #   - stream_df: <aeon_stream> (codec — full DataFrame on fetch)
    stream_type = "Encoder"

    codec_table_def = """ # Codec-based Feeder Encoder (prototype)
    -> streams_module.Feeder
    -> acquisition.Chunk
    ---
    sample_count: int32          # number of data points
    timestamps: json             # time range, sampling rate
    """
    for col in encoder_columns:
        clean_col = re.sub(r"\([^)]*\)", "", col)
        codec_table_def += f"{clean_col}: json             # summary stats (min, max, mean, dtype)\n    "
    codec_table_def += "stream_df: <aeon_stream>   # full DataFrame via codec\n    "

    @schema
    class FeederEncoderCodec(dj.Imported):
        definition = codec_table_def

        # Same key_source as the blob table: Chunk × Feeder with overlapping time
        @property
        def key_source(self):
            return (
                acquisition.Chunk
                * streams_module.Feeder.join(streams_module.Feeder.RemovalTime, left=True)
                & "chunk_start >= feeder_install_time"
                & 'chunk_start < IFNULL(feeder_removal_time, "2200-01-01")'
            )

        def make(self, key):
            from swc.aeon.io import api as io_api

            from aeon.dj_pipeline.utils.load_metadata import get_stream_reader_for_epoch

            # --- 1. Fetch chunk boundaries and epoch ---
            chunk_start, chunk_end, epoch_start = (acquisition.Chunk & key).fetch1(
                "chunk_start", "chunk_end", "epoch_start"
            )
            data_dirs = acquisition.Experiment.get_data_directories(key)
            device_name = key["device_name"]

            # --- 2. Get stream reader and load data ---
            stream_reader = get_stream_reader_for_epoch(
                key["experiment_name"], device_name, stream_type, epoch_start
            )
            stream_df = io_api.load(
                root=data_dirs,
                reader=stream_reader,
                start=pd.Timestamp(chunk_start),
                end=pd.Timestamp(chunk_end),
            )

            # --- 3. Compute summary stats for JSON columns ---
            row = {
                **key,
                "sample_count": len(stream_df),
                "timestamps": timestamp_stats(stream_df.index),
            }
            for col in stream_reader.columns:
                if col.startswith("_"):
                    continue
                clean_col = re.sub(r"\([^)]*\)", "", col)
                row[clean_col] = column_stats(stream_df[col].values)

            # --- 4. Store codec reference for stream_df (self-contained for decode) ---
            row["stream_df"] = {
                "stream_type": stream_type,
                "experiment_name": key["experiment_name"],
                "device_name": device_name,
                "chunk_start": str(chunk_start),
                "chunk_end": str(chunk_end),
                "epoch_start": str(epoch_start),
            }

            self.insert1(row, ignore_extra_fields=True)

    logger.info(f"Created FeederEncoderCodec table")
    logger.info(f"Definition:\n{codec_table_def}")

    # ---------------------------------------------------------------------------
    # 7. Populate codec table via make() — same as blob table uses populate()
    # ---------------------------------------------------------------------------
    logger.info("Populating codec-based FeederEncoderCodec via make()...")
    FeederEncoderCodec.populate(max_calls=6, display_progress=True)
    logger.info(f"Codec table has {len(FeederEncoderCodec())} entries")

    # ---------------------------------------------------------------------------
    # 8. Inspect the codec table — show what's stored
    # ---------------------------------------------------------------------------
    feeder1_filter = "device_name = 'Feeder1'"

    logger.info("\n" + "=" * 60)
    logger.info("CODEC TABLE CONTENTS (Feeder1)")
    codec_row = (FeederEncoderCodec & feeder1_filter).fetch1()

    logger.info(f"  sample_count: {codec_row['sample_count']}")
    logger.info(f"  timestamps:   {json.dumps(codec_row['timestamps'], indent=4)}")
    for col in encoder_columns:
        clean_col = re.sub(r"\([^)]*\)", "", col)
        logger.info(f"  {clean_col}:       {json.dumps(codec_row[clean_col], indent=4)}")
    logger.info(f"  stream_df:  <DataFrame decoded by codec>")

    # ---------------------------------------------------------------------------
    # 9. Fetch stream_df (codec) and compare with blob approach
    # ---------------------------------------------------------------------------
    from aeon.dj_pipeline import fetch_stream

    logger.info("\n" + "=" * 60)
    logger.info("BLOB APPROACH (current)")
    blob_query = feeder_encoder_table & feeder1_filter & "sample_count > 0"
    df_blob = fetch_stream(blob_query)
    logger.info(f"  Shape: {df_blob.shape}")
    logger.info(f"  Columns: {list(df_blob.columns)}")
    logger.info(f"  First 3 rows:\n{df_blob.head(3)}")

    logger.info("\nCODEC APPROACH — stream_df")
    df_codec = codec_row["stream_df"]  # Already decoded by codec
    logger.info(f"  Type: {type(df_codec)}")
    logger.info(f"  Shape: {df_codec.shape}")
    logger.info(f"  Columns: {list(df_codec.columns)}")
    logger.info(f"  First 3 rows:\n{df_codec.head(3)}")

    # ---------------------------------------------------------------------------
    # 10. Assert identical data
    # ---------------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON (blob vs codec stream_df)")

    assert df_codec.shape == df_blob.shape, (
        f"Shape mismatch: {df_codec.shape} vs {df_blob.shape}"
    )
    for col in df_blob.columns:
        blob_vals = df_blob[col].values
        codec_vals = df_codec[col].values
        if np.issubdtype(blob_vals.dtype, np.number):
            match = np.array_equal(blob_vals, codec_vals)
        else:
            match = (blob_vals == codec_vals).all()
        status = "OK" if match else "MISMATCH"
        logger.info(f"  Column '{col}': {status}")
        assert match, f"Data mismatch in column '{col}'"

    # Verify stats are consistent with actual data
    logger.info("\nSTATS VALIDATION:")
    ts_stats = codec_row["timestamps"]
    logger.info(f"  Sampling rate: {ts_stats.get('sampling_rate_hz')} Hz")
    logger.info(f"  Median dt:    {ts_stats.get('median_dt_ms')} ms")
    for col in encoder_columns:
        clean_col = re.sub(r"\([^)]*\)", "", col)
        stats = codec_row[clean_col]
        actual_min = float(df_codec[col].min())
        actual_max = float(df_codec[col].max())
        assert stats["min"] == actual_min, f"{clean_col} min mismatch"
        assert stats["max"] == actual_max, f"{clean_col} max mismatch"
        logger.info(f"  {clean_col}: min={stats['min']}, max={stats['max']}, mean={stats['mean']}")

    logger.info("\n" + "=" * 60)
    logger.info("SUCCESS: All comparisons passed!")
    logger.info("=" * 60)

    # ---------------------------------------------------------------------------
    # 11. Storage comparison
    # ---------------------------------------------------------------------------
    logger.info("\nSTORAGE COMPARISON:")
    blob_row = (feeder_encoder_table & feeder1_filter).fetch1()
    blob_size = sum(blob_row[k].nbytes for k in blob_row if hasattr(blob_row[k], "nbytes"))
    logger.info(f"  Blob table row size:  {blob_size:,} bytes ({blob_size/1024:.1f} KB)")

    # Codec table: JSON stats + codec reference
    codec_stored = {
        "timestamps": codec_row["timestamps"],
        **{re.sub(r"\([^)]*\)", "", c): codec_row[re.sub(r"\([^)]*\)", "", c)] for c in encoder_columns},
    }
    # stream_df is already decoded — estimate its JSON size from the reference structure
    codec_json_size = sum(len(json.dumps(v)) for v in codec_stored.values())
    codec_json_size += 150  # approximate stream_df JSON reference
    logger.info(f"  Codec table row size: {codec_json_size:,} bytes ({codec_json_size/1024:.1f} KB)")
    logger.info(f"  Reduction: {blob_size / max(codec_json_size, 1):.0f}x smaller in MySQL")

finally:
    logger.info("\nStopping MySQL container...")
    container.stop()
    logger.info("Done.")
