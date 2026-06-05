"""Codecs for lazy-loading stream data.

Both codecs store JSON references in MySQL; on fetch, they reconstruct readers
and return DataFrames built from the raw files on disk.

- ``AeonStreamCodec`` (``<aeon_stream>``) — HARP-clocked time-indexed streams,
  loaded via ``io_api.load(start, end)``.
- ``OnixStreamCodec`` (``<aeon_onix_stream>``) — ONIX-clocked stream groups
  (e.g., Bno055 IMU). Structural-only: loads + prefix-renames + concats. Does
  NOT apply HARP sync regression — that's exposed via ``OnixImuChunk.synced_df``.
"""

import datajoint as dj
import numpy as np
import pandas as pd


class AeonStreamCodec(dj.Codec):
    """Codec for lazy-loading stream data from raw files.

    Used for the `stream_df` column in auto-generated stream tables.
    On insert, stores a self-contained JSON reference with enough info
    to reconstruct the stream reader and load data at fetch time.
    On fetch, returns the full pd.DataFrame from raw files.

    Stored JSON format::

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

    _REQUIRED_KEYS = {
        "stream_type",
        "experiment_name",
        "device_name",
        "chunk_start",
        "chunk_end",
        "epoch_start",
    }

    def get_dtype(self, is_store: bool) -> str:
        """Return JSON as the storage type."""
        return "json"

    def encode(self, value, *, key=None, store_name=None):
        """Validate and store the stream reference dict as JSON."""
        if not isinstance(value, dict):
            raise TypeError(f"AeonStreamCodec expects a dict, got {type(value).__name__}")
        missing = self._REQUIRED_KEYS - value.keys()
        if missing:
            raise ValueError(f"AeonStreamCodec missing required keys: {missing}")
        return value

    def decode(self, stored, *, key=None):
        """Load stream data from raw files using the stored reference."""
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


class OnixStreamCodec(dj.Codec):
    """Structural-only codec for ONIX-clocked stream groups (e.g., Bno055 IMU).

    Stores a self-contained JSON reference. On fetch, loads the referenced
    binaries via the dotmap reader hierarchy in ``aeon/schema/ephys.py``,
    prefix-renames stream columns by their source class, concats on the shared
    sample index, and returns an **ONIX-clock-indexed** DataFrame.

    The codec deliberately does NOT apply the HARP sync regression — that's
    the caller's responsibility, exposed as ``OnixImuChunk.synced_df``.

    Stored JSON format::

        {
            "experiment_name": "...",
            "epoch_start": "...",
            "sync_start": "...",
            "device_name": "NeuropixelsV2Beta",
            "stream_group": "Bno055"
        }
    """

    name = "aeon_onix_stream"

    _REQUIRED_KEYS = {
        "experiment_name",
        "epoch_start",
        "sync_start",
        "device_name",
        "stream_group",
    }

    def get_dtype(self, is_store: bool) -> str:
        """Return JSON as the storage type."""
        return "json"

    def encode(self, value, *, key=None, store_name=None):
        """Validate and store the stream reference dict as JSON."""
        if not isinstance(value, dict):
            raise TypeError(f"OnixStreamCodec expects a dict, got {type(value).__name__}")
        missing = self._REQUIRED_KEYS - value.keys()
        if missing:
            raise ValueError(f"OnixStreamCodec missing required keys: {missing}")
        return value

    def decode(self, stored, *, key=None):
        """Load + merge the referenced ONIX-clocked stream group as an ONIX-indexed DataFrame.

        Uses ``chunk_indices``, ``onix_ts_start``, and ``onix_ts_end`` from the
        stored reference to reload exactly the data the populate-time row was
        built from: load every overlapping Bno055 binary chunk, concatenate,
        then filter to the sync window.
        """
        # Lazy imports to avoid circular references at module load time.
        from aeon.dj_pipeline import acquisition, ephys
        from aeon.dj_pipeline.utils.onix_imu import (
            IMU_COLUMNS,
            find_overlapping_bno055_chunks,
            load_and_merge_bno055,
        )

        if stored["stream_group"] != "Bno055":
            raise NotImplementedError(
                f"stream_group={stored['stream_group']!r} not supported. Only 'Bno055' is wired today."
            )

        sm_key = {
            "experiment_name": stored["experiment_name"],
            "epoch_start": pd.Timestamp(stored["epoch_start"]),
            "sync_start": pd.Timestamp(stored["sync_start"]),
        }

        # EphysEpoch is the ephys-side peer of acquisition.Epoch and owns its
        # own epoch_dir; acquisition.Epoch holds behavior epochs only.
        epoch_dir = (ephys.EphysEpoch & sm_key).fetch1("epoch_dir")
        raw_dir = acquisition.Experiment.get_data_directory(
            {"experiment_name": stored["experiment_name"]}, "raw-ephys"
        )
        if raw_dir is None:
            raise FileNotFoundError(
                f"No raw-ephys data directory registered for experiment {stored['experiment_name']!r}"
            )
        device_dir = raw_dir / epoch_dir / stored["device_name"]

        # Prefer the chunk_indices captured at populate time. Fall back to a
        # fresh overlap scan if absent (for backward-compat with older rows).
        chunk_indices = stored.get("chunk_indices")
        onix_ts_start = stored.get("onix_ts_start")
        onix_ts_end = stored.get("onix_ts_end")
        if onix_ts_start is None or onix_ts_end is None:
            ts_start_raw, ts_end_raw = (ephys.EphysSyncModel & sm_key).fetch1(
                "onix_ts_start", "onix_ts_end"
            )
            onix_ts_start = int(ts_start_raw)
            onix_ts_end = int(ts_end_raw)
        if chunk_indices is None:
            chunk_indices = find_overlapping_bno055_chunks(
                device_dir, stored["device_name"],
                int(onix_ts_start), int(onix_ts_end),
            )

        if not chunk_indices:
            return pd.DataFrame(columns=list(IMU_COLUMNS), index=pd.Index([], dtype=np.uint64))

        df = pd.concat(
            [load_and_merge_bno055(device_dir, stored["device_name"], n)
             for n in chunk_indices]
        )
        return df[(df.index >= int(onix_ts_start)) & (df.index <= int(onix_ts_end))]
