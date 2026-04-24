"""AeonStreamCodec — lazy-loading codec for stream data.

Stores a JSON reference in MySQL; on fetch, reconstructs the stream reader
and calls io_api.load() to return the full DataFrame from raw files.

This <aeon_stream> codec is used for the `stream_df` column in auto-generated
stream tables for lazy DataFrame loading.
"""

import datajoint as dj
import pandas as pd


class AeonStreamCodec(dj.Codec):
    """Codec for lazy-loading stream data from raw files.

    Used for the `stream_df` column in auto-generated stream tables.
    On insert, stores a self-contained JSON reference with enough info
    to reconstruct the stream reader and load data at fetch time.
    On fetch, returns the full pd.DataFrame from raw files.

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
