"""Integration tests using golden dataset for full ingestion pipeline.

Tests the "Three Decoupled Steps" architecture:
- Step 1: Catalog population from Pydantic class (populate_catalog_from_pydantic)
- Step 2: Table creation via streams_maker.main()
- Step 3: Data population via EpochConfig.make() and DeviceDataStream.make()

Requirements:
1. Golden dataset at ~/sciops-data/project_aeon/aeon/data/raw/AEON3/abcBehav0/
2. aeon_exp_foragingABC package installed

Tests gracefully skip if data unavailable.
"""

import pytest


# =============================================================================
# Step 1: Catalog Population Tests
# =============================================================================


@pytest.mark.integration
class TestStep1CatalogPopulation:
    """Test Step 1: Catalog population from Pydantic class.

    This happens at worker startup via populate_catalog_from_pydantic().
    The full_pipeline fixture handles this automatically.
    """

    def test_stream_types_populated(self, full_pipeline):
        """Verify StreamType catalog populated from Pydantic class."""
        streams = full_pipeline["streams"]

        stream_types = streams.StreamType.to_dicts()
        assert len(stream_types) > 0

        # Should have Video, BeamBreak, Encoder, etc.
        stream_type_names = [st["stream_type"] for st in stream_types]
        assert "Video" in stream_type_names

    def test_device_types_populated(self, full_pipeline):
        """Verify DeviceType catalog populated from Pydantic class."""
        streams = full_pipeline["streams"]

        device_types = streams.DeviceType.to_arrays("device_type")
        assert len(device_types) > 0

        # Should have Camera, Feeder, etc. (leaf class names, not inherited parent names)
        assert "Camera" in device_types
        assert "Feeder" in device_types

    def test_device_type_streams_linked(self, full_pipeline):
        """Verify DeviceType.Stream linking table populated."""
        streams = full_pipeline["streams"]

        device_streams = streams.DeviceType.Stream.to_dicts()
        assert len(device_streams) > 0


# =============================================================================
# Step 2: Table Creation Tests
# =============================================================================


@pytest.mark.integration
class TestStep2TableCreation:
    """Test Step 2: Table creation via streams_maker.main().

    ExperimentDevice tables (e.g., Camera) and DeviceDataStream tables
    (e.g., CameraVideo) should exist after streams_maker.main().
    """

    def test_experiment_device_tables_exist(self, full_pipeline):
        """Verify ExperimentDevice tables created (e.g., Camera)."""
        streams = full_pipeline["streams"]

        # Check Camera table exists (leaf class name, not parent SpinnakerCamera)
        assert hasattr(streams, "Camera")

    def test_device_data_stream_tables_exist(self, full_pipeline):
        """Verify DeviceDataStream tables created (e.g., CameraVideo)."""
        streams = full_pipeline["streams"]

        # Check at least one DeviceDataStream table exists
        video_tables = [name for name in dir(streams) if "Video" in name and not name.startswith("_")]
        assert len(video_tables) > 0


# =============================================================================
# Step 3: Data Population Tests (Epoch, Chunk, EpochConfig, Streams)
# =============================================================================


@pytest.mark.integration
class TestEpochIngestion:
    """Test Epoch.ingest_epochs() with golden dataset."""

    def test_epochs_ingested(self, test_epochs, full_pipeline, golden_dataset_config):
        """Verify epochs are detected and ingested from filesystem."""
        assert len(test_epochs) >= 1

    def test_epoch_dir_matches(self, test_epochs, full_pipeline, golden_dataset_config):
        """Verify ingested epoch has correct epoch_dir."""
        acquisition = full_pipeline["acquisition"]
        cfg = golden_dataset_config

        epoch = (acquisition.Epoch & {"experiment_name": cfg["experiment_name"]}).fetch1()
        assert epoch["epoch_dir"] == cfg["epoch_dir"]


@pytest.mark.integration
class TestEpochConfigMake:
    """Test EpochConfig.make() with golden dataset (Step 3 - DML only)."""

    def test_epoch_config_populates(self, test_epochs, full_pipeline, golden_dataset_config):
        """Verify EpochConfig.make() completes without error."""
        acquisition = full_pipeline["acquisition"]
        cfg = golden_dataset_config

        acquisition.EpochConfig.populate()

        assert len(acquisition.EpochConfig & {"experiment_name": cfg["experiment_name"]}) >= 1

    def test_epoch_config_meta_has_rig_metadata(self, test_epochs, full_pipeline, golden_dataset_config):
        """Verify EpochConfig.Meta contains rig metadata."""
        import json

        acquisition = full_pipeline["acquisition"]
        cfg = golden_dataset_config

        acquisition.EpochConfig.populate()

        meta = (acquisition.EpochConfig.Meta & {"experiment_name": cfg["experiment_name"]}).fetch1()

        # metadata may be a JSON string (MariaDB json-as-longtext) or a dict
        metadata = meta["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        assert "cameras" in metadata
        assert "feeders" in metadata
        assert len(metadata["cameras"]) == cfg["expected_camera_count"]
        assert len(metadata["feeders"]) == cfg["expected_feeder_count"]

    def test_streams_device_registered(self, test_epochs, full_pipeline, golden_dataset_config):
        """Verify streams.Device populated by EpochConfig.make()."""
        acquisition = full_pipeline["acquisition"]
        streams = full_pipeline["streams"]

        acquisition.EpochConfig.populate()

        devices = streams.Device.to_dicts()
        assert len(devices) > 0


@pytest.mark.integration
class TestChunkIngestion:
    """Test Chunk.ingest_chunks() with golden dataset."""

    def test_chunks_ingested(self, test_epochs, full_pipeline, golden_dataset_config):
        """Verify chunks are detected and ingested."""
        acquisition = full_pipeline["acquisition"]
        cfg = golden_dataset_config

        acquisition.Chunk.ingest_chunks(cfg["experiment_name"])

        assert len(acquisition.Chunk & {"experiment_name": cfg["experiment_name"]}) >= 1

    def test_chunk_start_time_valid(self, test_epochs, full_pipeline, golden_dataset_config):
        """Verify chunk has valid start time."""
        acquisition = full_pipeline["acquisition"]
        cfg = golden_dataset_config

        acquisition.Chunk.ingest_chunks(cfg["experiment_name"])

        chunk = (acquisition.Chunk & {"experiment_name": cfg["experiment_name"]}).fetch1()
        assert chunk["chunk_start"] is not None


@pytest.mark.integration
class TestStreamDataIngestion:
    """Test DeviceDataStream.make() with golden dataset.

    Strategy: Test ALL stream types, but limit entries per stream.
    This validates the full pipeline without ingesting all data.
    """

    POPULATE_LIMIT = 10

    def test_all_stream_tables_can_populate(self, test_epochs, full_pipeline, golden_dataset_config):
        """Verify stream tables can be populated with limit."""
        import datajoint as dj

        acquisition = full_pipeline["acquisition"]
        streams = full_pipeline["streams"]
        cfg = golden_dataset_config

        # Ensure prerequisites
        acquisition.EpochConfig.populate()
        acquisition.Chunk.ingest_chunks(cfg["experiment_name"])

        # Get all Computed tables in streams module (these are DeviceDataStream tables)
        stream_tables = []
        for name in dir(streams):
            if name.startswith("_"):
                continue
            obj = getattr(streams, name, None)
            if isinstance(obj, type) and issubclass(obj, dj.Imported) and obj is not dj.Imported:
                stream_tables.append((name, obj))

        if not stream_tables:
            pytest.skip("No stream tables found")

        results = {}
        for table_name, table in stream_tables:
            try:
                table.populate(max_calls=self.POPULATE_LIMIT, display_progress=False, suppress_errors=True)
                total = len(table & {"experiment_name": cfg["experiment_name"]})
                with_data = len(
                    table & {"experiment_name": cfg["experiment_name"]} & "sample_count > 0"
                )
                results[table_name] = {"total": total, "with_data": with_data}
            except Exception as e:
                results[table_name] = f"error: {e}"

        # At least some tables should have entries with actual data (sample_count > 0)
        with_data = {
            k: v for k, v in results.items() if isinstance(v, dict) and v["with_data"] > 0
        }
        assert len(with_data) > 0, f"No stream tables with sample_count > 0. Results: {results}"

    def test_video_stream_has_data(self, test_epochs, full_pipeline, golden_dataset_config):
        """Verify at least one Video stream populated with data."""
        import datajoint as dj

        acquisition = full_pipeline["acquisition"]
        streams = full_pipeline["streams"]
        cfg = golden_dataset_config

        # Ensure prerequisites
        acquisition.EpochConfig.populate()
        acquisition.Chunk.ingest_chunks(cfg["experiment_name"])

        # Find video tables
        video_tables = []
        for name in dir(streams):
            if "Video" in name and not name.startswith("_"):
                obj = getattr(streams, name, None)
                if isinstance(obj, type) and issubclass(obj, dj.Imported):
                    video_tables.append((name, obj))

        if not video_tables:
            pytest.skip("No Video stream tables found")

        for table_name, table in video_tables:
            table.populate(max_calls=self.POPULATE_LIMIT, display_progress=False, suppress_errors=True)

        # Check at least one video entry has actual data (sample_count > 0)
        for table_name, table in video_tables:
            query = table & {"experiment_name": cfg["experiment_name"]} & "sample_count > 0"
            if len(query):
                return  # Success

        pytest.fail("No Video stream entries with sample_count > 0")

    def test_harp_stream_has_data(self, test_epochs, full_pipeline, golden_dataset_config):
        """Verify at least one Harp-based stream populated."""
        import datajoint as dj

        acquisition = full_pipeline["acquisition"]
        streams = full_pipeline["streams"]
        cfg = golden_dataset_config

        # Ensure prerequisites
        acquisition.EpochConfig.populate()
        acquisition.Chunk.ingest_chunks(cfg["experiment_name"])

        # Find Harp-based tables (BeamBreak, Encoder, Weight, Pellet, etc.)
        harp_indicators = ["BeamBreak", "Encoder", "Weight", "Pellet", "Deliver"]
        harp_tables = []
        for name in dir(streams):
            if any(h in name for h in harp_indicators) and not name.startswith("_"):
                obj = getattr(streams, name, None)
                if isinstance(obj, type) and issubclass(obj, dj.Imported):
                    harp_tables.append((name, obj))

        if not harp_tables:
            pytest.skip("No Harp stream tables found")

        for table_name, table in harp_tables:
            table.populate(max_calls=self.POPULATE_LIMIT, display_progress=False, suppress_errors=True)

        # Check at least one has actual data (sample_count > 0)
        for table_name, table in harp_tables:
            query = table & {"experiment_name": cfg["experiment_name"]} & "sample_count > 0"
            if len(query):
                return  # Success

        pytest.fail("No Harp stream entries with sample_count > 0")


# =============================================================================
# fetch_stream Tests (uses populated stream data from above)
# =============================================================================


@pytest.mark.integration
class TestFetchStream:
    """Test fetch_stream() with real populated stream data.

    fetch_stream is the primary data access function for stream tables.
    For codec-based tables, it fetches stream_df (decoded via AeonStreamCodec)
    and returns a time-indexed DataFrame. These tests validate the full round-trip:
    raw data -> DB -> fetch_stream -> DataFrame.
    """

    POPULATE_LIMIT = 10

    def _ensure_stream_data(self, full_pipeline, golden_dataset_config):
        """Ensure stream data is populated (idempotent)."""
        import datajoint as dj

        acquisition = full_pipeline["acquisition"]
        streams = full_pipeline["streams"]
        cfg = golden_dataset_config

        acquisition.EpochConfig.populate()
        acquisition.Chunk.ingest_chunks(cfg["experiment_name"])

        # Populate all stream tables
        for name in dir(streams):
            if name.startswith("_"):
                continue
            obj = getattr(streams, name, None)
            if isinstance(obj, type) and issubclass(obj, dj.Imported) and obj is not dj.Imported:
                obj.populate(max_calls=self.POPULATE_LIMIT, display_progress=False, suppress_errors=True)

    def _find_stream_with_data(self, streams, cfg, name_filter=None):
        """Find a stream table with sample_count > 0 data."""
        import datajoint as dj

        for name in dir(streams):
            if name.startswith("_"):
                continue
            if name_filter and not name_filter(name):
                continue
            obj = getattr(streams, name, None)
            if isinstance(obj, type) and issubclass(obj, dj.Imported):
                query = obj & {"experiment_name": cfg["experiment_name"]} & "sample_count > 0"
                if len(query):
                    return name, obj, query
        return None

    def test_fetch_stream_returns_dataframe(self, test_epochs, full_pipeline, golden_dataset_config):
        """Verify fetch_stream returns a non-empty time-indexed DataFrame."""
        import pandas as pd

        from aeon.dj_pipeline import fetch_stream

        self._ensure_stream_data(full_pipeline, golden_dataset_config)
        streams = full_pipeline["streams"]
        cfg = golden_dataset_config

        result = self._find_stream_with_data(streams, cfg)
        if result is None:
            pytest.fail("No stream table with sample_count > 0 found")

        name, table, query = result
        df = fetch_stream(query)
        assert isinstance(df, pd.DataFrame)
        assert df.index.name == "time"
        assert not df.empty

    def test_fetch_stream_video_columns(self, test_epochs, full_pipeline, golden_dataset_config):
        """Verify fetch_stream on CameraVideo returns expected columns."""
        from aeon.dj_pipeline import fetch_stream

        self._ensure_stream_data(full_pipeline, golden_dataset_config)
        streams = full_pipeline["streams"]
        cfg = golden_dataset_config

        result = self._find_stream_with_data(
            streams, cfg, name_filter=lambda n: "CameraVideo" in n
        )
        if result is None:
            pytest.skip("No CameraVideo data populated")

        name, table, query = result
        df = fetch_stream(query)
        assert df.index.name == "time"
        assert "hw_counter" in df.columns
        assert "hw_timestamp" in df.columns
        assert len(df) > 0

    def test_fetch_stream_harp_columns(self, test_epochs, full_pipeline, golden_dataset_config):
        """Verify fetch_stream on a Harp stream returns expected columns."""
        from aeon.dj_pipeline import fetch_stream

        self._ensure_stream_data(full_pipeline, golden_dataset_config)
        streams = full_pipeline["streams"]
        cfg = golden_dataset_config

        harp_indicators = ["BeamBreak", "Encoder", "DeliverPellet"]
        result = self._find_stream_with_data(
            streams, cfg, name_filter=lambda n: any(h in n for h in harp_indicators)
        )
        if result is None:
            pytest.skip("No Harp stream data populated")

        name, table, query = result
        df = fetch_stream(query)
        assert df.index.name == "time"
        assert len(df) > 0
        assert len(df.columns) > 0

    def test_fetch_stream_drop_pk(self, test_epochs, full_pipeline, golden_dataset_config):
        """Verify drop_pk=True removes primary key columns from result."""
        from aeon.dj_pipeline import fetch_stream

        self._ensure_stream_data(full_pipeline, golden_dataset_config)
        streams = full_pipeline["streams"]
        cfg = golden_dataset_config

        result = self._find_stream_with_data(streams, cfg)
        if result is None:
            pytest.fail("No stream table with data found")

        name, table, query = result
        pk_cols = query.primary_key

        df_dropped = fetch_stream(query, drop_pk=True)
        for pk in pk_cols:
            assert pk not in df_dropped.columns

        df_kept = fetch_stream(query, drop_pk=False)
        pk_in_result = [pk for pk in pk_cols if pk in df_kept.columns]
        assert len(pk_in_result) > 0

    def test_fetch_stream_timestamps_rounded(self, test_epochs, full_pipeline, golden_dataset_config):
        """Verify round_microseconds option works correctly."""
        import numpy as np

        from aeon.dj_pipeline import fetch_stream

        self._ensure_stream_data(full_pipeline, golden_dataset_config)
        streams = full_pipeline["streams"]
        cfg = golden_dataset_config

        result = self._find_stream_with_data(streams, cfg)
        if result is None:
            pytest.skip("No stream data found")

        name, table, query = result
        df = fetch_stream(query, round_microseconds=True)
        assert not df.empty
        # Timestamps should be rounded to microseconds (no sub-us precision)
        nanos = df.index.astype(np.int64)
        assert all(nanos % 1000 == 0), "Timestamps should be rounded to microseconds"


# =============================================================================
# Codec Stream Data Regression Tests
# =============================================================================


@pytest.mark.integration
class TestCodecStreamData:
    """Verify codec-based stream tables store correct summary stats and return correct DataFrames."""

    POPULATE_LIMIT = 10

    def _ensure_and_find(self, full_pipeline, golden_dataset_config, name_filter=None):
        """Populate streams and find one with data."""
        import datajoint as dj

        acquisition = full_pipeline["acquisition"]
        streams = full_pipeline["streams"]
        cfg = golden_dataset_config

        acquisition.EpochConfig.populate()
        acquisition.Chunk.ingest_chunks(cfg["experiment_name"])

        for name in dir(streams):
            if name.startswith("_"):
                continue
            if name_filter and not name_filter(name):
                continue
            obj = getattr(streams, name, None)
            if isinstance(obj, type) and issubclass(obj, dj.Imported) and obj is not dj.Imported:
                obj.populate(max_calls=self.POPULATE_LIMIT, display_progress=False, suppress_errors=True)
                query = obj & {"experiment_name": cfg["experiment_name"]} & "sample_count > 0"
                if len(query):
                    return name, obj, query
        return None

    def test_stream_df_returns_dataframe(self, test_epochs, full_pipeline, golden_dataset_config):
        """Verify stream_df codec column returns a pandas DataFrame."""
        import pandas as pd

        result = self._ensure_and_find(full_pipeline, golden_dataset_config)
        if result is None:
            pytest.fail("No stream table with data found")

        name, table, query = result
        row = query.fetch1()
        assert isinstance(row["stream_df"], pd.DataFrame)
        assert not row["stream_df"].empty

    def test_sample_count_matches_stream_df(self, test_epochs, full_pipeline, golden_dataset_config):
        """Verify sample_count matches len(stream_df)."""
        result = self._ensure_and_find(full_pipeline, golden_dataset_config)
        if result is None:
            pytest.fail("No stream table with data found")

        name, table, query = result
        row = query.fetch1()
        assert row["sample_count"] == len(row["stream_df"])

    def test_timestamp_stats_match_stream_df(self, test_epochs, full_pipeline, golden_dataset_config):
        """Verify timestamp JSON stats match actual stream_df index."""
        result = self._ensure_and_find(full_pipeline, golden_dataset_config)
        if result is None:
            pytest.fail("No stream table with data found")

        name, table, query = result
        row = query.fetch1()
        ts_stats = row["timestamps"]
        df = row["stream_df"]

        assert ts_stats["count"] == len(df)
        assert "sampling_rate_hz" in ts_stats
        assert "sampling_rate_hz" in ts_stats

    def test_column_stats_match_stream_df(self, test_epochs, full_pipeline, golden_dataset_config):
        """Verify JSON summary stats match actual stream_df data."""
        import numpy as np

        result = self._ensure_and_find(
            full_pipeline, golden_dataset_config, name_filter=lambda n: "Encoder" in n
        )
        if result is None:
            pytest.skip("No Encoder stream data populated")

        name, table, query = result
        # Restrict to one device to get exactly one row
        row = (query & "device_name LIKE '%1'").fetch1()
        df = row["stream_df"]

        for col in df.columns:
            if col in row and isinstance(row[col], dict) and "min" in row[col]:
                stats = row[col]
                assert stats["min"] == float(np.nanmin(df[col].values))
                assert stats["max"] == float(np.nanmax(df[col].values))
                assert stats["count"] == len(df)

    def test_no_blob_columns_in_stream_tables(self, test_epochs, full_pipeline, golden_dataset_config):
        """Verify stream tables have no blob columns (all json + codec)."""
        import datajoint as dj

        streams = full_pipeline["streams"]

        for name in dir(streams):
            if name.startswith("_"):
                continue
            obj = getattr(streams, name, None)
            if isinstance(obj, type) and issubclass(obj, dj.Imported) and obj is not dj.Imported:
                for attr_name in obj.heading.secondary_attributes:
                    attr = obj.heading.attributes[attr_name]
                    assert not attr.is_blob, (
                        f"{name}.{attr_name} is still a blob column — expected json or codec"
                    )
