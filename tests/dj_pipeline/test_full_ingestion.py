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

        stream_types = streams.StreamType.fetch(as_dict=True)
        assert len(stream_types) > 0

        # Should have Video, BeamBreak, Encoder, etc.
        stream_type_names = [st["stream_type"] for st in stream_types]
        assert "Video" in stream_type_names

    def test_device_types_populated(self, full_pipeline):
        """Verify DeviceType catalog populated from Pydantic class."""
        streams = full_pipeline["streams"]

        device_types = streams.DeviceType.fetch("device_type")
        assert len(device_types) > 0

        # Should have SpinnakerCamera, UndergroundFeeder, etc.
        assert "SpinnakerCamera" in device_types

    def test_device_type_streams_linked(self, full_pipeline):
        """Verify DeviceType.Stream linking table populated."""
        streams = full_pipeline["streams"]

        device_streams = streams.DeviceType.Stream.fetch(as_dict=True)
        assert len(device_streams) > 0


# =============================================================================
# Step 2: Table Creation Tests
# =============================================================================


@pytest.mark.integration
class TestStep2TableCreation:
    """Test Step 2: Table creation via streams_maker.main().

    ExperimentDevice tables (e.g., SpinnakerCamera) and DeviceDataStream tables
    (e.g., SpinnakerCameraVideo) should exist after streams_maker.main().
    """

    def test_experiment_device_tables_exist(self, full_pipeline):
        """Verify ExperimentDevice tables created (e.g., SpinnakerCamera)."""
        streams = full_pipeline["streams"]

        # Check SpinnakerCamera table exists
        assert hasattr(streams, "SpinnakerCamera")

    def test_device_data_stream_tables_exist(self, full_pipeline):
        """Verify DeviceDataStream tables created (e.g., SpinnakerCameraVideo)."""
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

        epoch_configs = (acquisition.EpochConfig & {"experiment_name": cfg["experiment_name"]}).fetch()
        assert len(epoch_configs) >= 1

    def test_epoch_config_meta_has_rig_metadata(self, test_epochs, full_pipeline, golden_dataset_config):
        """Verify EpochConfig.Meta contains rig metadata."""
        acquisition = full_pipeline["acquisition"]
        cfg = golden_dataset_config

        acquisition.EpochConfig.populate()

        meta = (acquisition.EpochConfig.Meta & {"experiment_name": cfg["experiment_name"]}).fetch1()

        assert "cameras" in meta["metadata"]
        assert "feeders" in meta["metadata"]
        assert len(meta["metadata"]["cameras"]) == cfg["expected_camera_count"]
        assert len(meta["metadata"]["feeders"]) == cfg["expected_feeder_count"]

    def test_streams_device_registered(self, test_epochs, full_pipeline, golden_dataset_config):
        """Verify streams.Device populated by EpochConfig.make()."""
        acquisition = full_pipeline["acquisition"]
        streams = full_pipeline["streams"]

        acquisition.EpochConfig.populate()

        devices = streams.Device.fetch(as_dict=True)
        assert len(devices) > 0


@pytest.mark.integration
class TestChunkIngestion:
    """Test Chunk.ingest_chunks() with golden dataset."""

    def test_chunks_ingested(self, test_epochs, full_pipeline, golden_dataset_config):
        """Verify chunks are detected and ingested."""
        acquisition = full_pipeline["acquisition"]
        cfg = golden_dataset_config

        acquisition.Chunk.ingest_chunks(cfg["experiment_name"])

        chunks = (acquisition.Chunk & {"experiment_name": cfg["experiment_name"]}).fetch()
        assert len(chunks) >= 1

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
                table.populate(limit=self.POPULATE_LIMIT, display_progress=False, suppress_errors=True)
                count = len(table & {"experiment_name": cfg["experiment_name"]})
                results[table_name] = count
            except Exception as e:
                results[table_name] = f"error: {e}"

        # At least some tables should populate successfully
        populated = {k: v for k, v in results.items() if isinstance(v, int) and v > 0}
        assert len(populated) > 0, f"No stream tables populated. Results: {results}"

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
            table.populate(limit=self.POPULATE_LIMIT, display_progress=False, suppress_errors=True)

        # Check at least one video entry exists
        for table_name, table in video_tables:
            entries = (table & {"experiment_name": cfg["experiment_name"]}).fetch(as_dict=True)
            if entries:
                return  # Success

        pytest.fail("No Video stream entries populated")

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
            table.populate(limit=self.POPULATE_LIMIT, display_progress=False, suppress_errors=True)

        # Check at least one has data
        for table_name, table in harp_tables:
            count = len(table & {"experiment_name": cfg["experiment_name"]})
            if count > 0:
                return  # Success

        pytest.fail("No Harp stream entries populated")
