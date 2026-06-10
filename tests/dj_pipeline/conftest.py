"""Global configurations and fixtures for pytest.

Test commands:
    pytest -m unit                    # Unit tests (no database)
    pytest -m integration             # Integration tests (testcontainers MySQL)
    pytest tests/dj_pipeline/         # All tests
"""

import logging
import os
import tempfile
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def dj_download_to_tmp(request):
    """Redirect DataJoint attach downloads to a per-test tmpdir.

    DataJoint extracts <attach> columns to `dj.config["download_path"]` (cwd by
    default). Without this fixture, fetching sync_model rows leaves .joblib files
    in the repo root. Skipped for unit tests which mock datajoint entirely.
    """
    if request.node.get_closest_marker("unit"):
        yield
        return
    import datajoint as dj

    with tempfile.TemporaryDirectory() as tmpdir, dj.config.override(download_path=tmpdir):
        yield


# Single test prefix for ALL integration tests.
# Override via env var on systems where the DB user can only create specific prefixes.
TEST_DB_PREFIX = os.environ.get("TEST_DB_PREFIX", "test_aeon_")

# ============================================================================
# Golden Dataset Registry
# ============================================================================

GOLDEN_DATASETS = {
    # Original behavior golden: ~1h of abcBehav0 on AEON3, full 13 cameras + 6
    # feeders writing data to disk. Richest stream coverage (FeederEncoder
    # 848k samples, CameraPosition 12k samples). No paired ephys.
    "foraging_abc_2025_11_18": {
        "experiment_name": "abcBehav0-aeon3",
        "experiment_path": "AEON3/abcBehav0",
        "epoch_dir": "2025-11-18T10-13-15",
        "devices_schema": "swc.aeon_exp.foragingABC.experiment:Experiment",
        "arena_name": "arena-aeon3",
        "lab": "SWC",
        "location": "room-0",
        "experiment_type": "foraging",
        "required_files": [
            "Metadata.json",
            "CameraTop/CameraTop_2025-11-18T10-00-00.csv",
            "Feeder1/Feeder1_32_2025-11-18T10-00-00.bin",
        ],
        "expected_camera_count": 13,
        "expected_feeder_count": 6,
    },
    # Behavior arm of abcGolden01 — paired with foraging_abc_ephys_2026_05_11
    # (same experiment, same wall-clock window, AEON3 acquires behavior while
    # AEONX1 acquires ephys). Sparser on-disk data than abcBehav0 (5 cameras,
    # 4 feeders writing data) but exercises the newer file-name conventions:
    # mixed T07-00-00 for CSVs and T070000Z for bins (both parse cleanly via
    # swc.aeon.io.api.chunk_key).
    "foraging_abc_2026_05_11": {
        "experiment_name": "abcGolden01-aeon3",
        "experiment_path": "AEON3/abcGolden01",
        "epoch_dir": "2026-05-11T075134Z",
        "devices_schema": "swc.aeon_exp.foragingABC.experiment:Experiment",
        "arena_name": "circle-2m",
        "lab": "SWC",
        "location": "AEON3",
        "experiment_type": "foraging",
        "required_files": [
            "Metadata.json",
            "CameraNest/CameraNest_2026-05-11T07-00-00.csv",
            "Feeder3/Feeder3_32_2026-05-11T070000Z.bin",
        ],
        "expected_camera_count": 13,
        "expected_feeder_count": 6,
    },
    # Ephys golden dataset — 8-channel subset of abcGolden01 (NeuropixelsV2 ProbeB)
    # Electrodes 3982-3989 on shank3, 35 min continuous recording
    "foraging_abc_ephys_2026_05_11": {
        "experiment_name": "abcGolden01-aeonx1",
        "experiment_path": "AEONX1/abcGolden01",
        "epoch_dir": "2026-05-11T07-50-11",
        "subject": "IAA-1147881",
        "arena_name": "circle-2m",
        "lab": "SWC",
        "location": "AEON3",
        "experiment_type": "foraging",
        "device_name": "NeuropixelsV2",
        "probe_type": "neuropixels2.0-multishank",
        "electrode_config_name": "M81_ProbeB_4Shanks_1000_to_1700_um",
        "probe_serial": "23299108854",
        "n_channels": 8,                       # sorting subset (ElectrodeGroup)
        "n_recording_channels": 384,           # full recording width (active subset of probe)
        "electrodes": list(range(3982, 3990)),
        "required_files": [
            "Metadata.yml",
            "NeuropixelsV2/NeuropixelsV2_ProbeB_AmplifierData_0.bin",
            "NeuropixelsV2/NeuropixelsV2_ProbeB_Clock_0.bin",
        ],
        "expected_probe_count": 1,            # registered ProbeInsertion: ProbeB only (A disabled)
        "expected_discovered_probes": 2,      # raw discovery from epoch dir: ProbeA + ProbeB
        "golden_sorting_dir": "golden_test_sorting",
        "expected_unit_count": 14,
        "expected_total_spikes": 357_480,
        "expected_quality_counts": {"good": 7, "mua": 7},
    },
}

# Default golden data root (can be overridden via DJ_REPOSITORY_CONFIG env var)
DEFAULT_GOLDEN_DATA_ROOT = Path.home() / "sciops-data/project_aeon/aeon/data"

# NOTE: datajoint is imported lazily inside fixtures to allow unit tests
# (which mock datajoint) to run without triggering DB connections


def _check_golden_data(dataset_config: dict) -> Path:
    """Validate golden dataset files are present; call pytest.skip if not.

    Returns the epoch path on success.
    """
    import aeon.dj_pipeline as pipeline

    if "DJ_REPOSITORY_CONFIG" not in os.environ:
        pipeline.repository_config = {"ceph_aeon": str(DEFAULT_GOLDEN_DATA_ROOT)}

    if not Path(pipeline.repository_config["ceph_aeon"]).exists():
        pytest.skip(f"Golden data root not found: {pipeline.repository_config['ceph_aeon']}")

    from aeon.dj_pipeline.utils.paths import get_repository_path

    repo_path = get_repository_path("ceph_aeon")
    epoch_path = repo_path / "raw" / dataset_config["experiment_path"] / dataset_config["epoch_dir"]

    if not epoch_path.exists():
        pytest.skip(f"Golden dataset not found: {epoch_path}")

    for f in dataset_config["required_files"]:
        if not (epoch_path / f).exists():
            pytest.skip(f"Missing required file: {f}")

    return epoch_path


def _drop_test_schemas() -> None:
    """Drop all test-prefixed schemas, retrying to handle FK dependencies."""
    import datajoint as dj

    prefix = dj.config.database.database_prefix
    if not prefix or not prefix.endswith("_"):
        raise RuntimeError(
            f"Refusing to drop schemas: prefix={prefix!r} is not a safe test prefix. "
            "Set TEST_DB_PREFIX to a value ending in '_' (e.g. 'test_aeon_')."
        )

    schemas_to_drop = [s for s in dj.list_schemas() if s.startswith(prefix)]
    max_attempts = len(schemas_to_drop) + 1
    for _ in range(max_attempts):
        if not schemas_to_drop:
            break
        remaining = []
        for schema_name in schemas_to_drop:
            try:
                dj.Schema(schema_name).drop()
            except Exception as e:
                logger.warning(f"Failed to drop schema {schema_name}: {e}")
                remaining.append(schema_name)
        schemas_to_drop = remaining
    if schemas_to_drop:
        logger.error(f"Could not drop schemas after {max_attempts} attempts: {schemas_to_drop}")


# ============================================================================
# Integration Test Fixtures (testcontainers MySQL)
# ============================================================================


@pytest.fixture(scope="session")
def dj_config_integration(mysql_container):
    """Configure DataJoint and import pipeline with test prefix.

    Sets DJ config (including database_prefix) BEFORE importing pipeline
    modules. This way, module-level schema activation in lab.py, subject.py,
    acquisition.py, and streams.py naturally uses the test prefix — no manual
    re-decoration of table classes needed.
    """
    import datajoint as dj

    # Set config BEFORE any pipeline imports
    dj.config.safemode = False
    dj.config.database.database_prefix = TEST_DB_PREFIX

    if mysql_container is not None:
        # Testcontainers: use the container's connection details
        dj.config.database.host = os.environ["DJ_HOST"]
        dj.config.database.port = int(os.environ["DJ_PORT"])
        dj.config.database.user = os.environ["DJ_USER"]
        dj.config.database.password = os.environ["DJ_PASS"]
    # External DB: connection details already loaded from datajoint.json

    # Now import pipeline — all module-level schema activations use test prefix

    return {"database_prefix": TEST_DB_PREFIX}


@pytest.fixture(scope="session")
def streams_schema(dj_config_integration):
    """Provide access to streams catalog tables.

    Calls streams_maker.main() to ensure catalog tables (StreamType, DeviceType, etc.)
    are written to the auto-generated streams.py module.
    Session-scoped — shared by load_metadata integration tests and golden dataset tests.
    """
    import importlib
    import sys

    from aeon.dj_pipeline.utils import streams_maker

    f = streams_maker._STREAMS_MODULE_FILE
    original = f.read_bytes() if f.exists() else None

    # Delete auto-generated file so main() regenerates it with catalog tables
    f.unlink(missing_ok=True)

    # Remove stale module from cache so it gets reimported after regeneration
    sys.modules.pop("aeon.dj_pipeline.streams", None)

    streams_maker.main(create_tables=False)

    streams = importlib.import_module("aeon.dj_pipeline.streams")

    yield {
        "StreamType": streams.StreamType,
        "DeviceType": streams.DeviceType,
        "DeviceName": streams.DeviceName,
        "schema": streams.schema,
        "schema_name": streams.schema.database,
    }

    # Teardown
    # - drop test schema (may already be dropped by full_pipeline)
    # - restore original streams.py file
    import datajoint as dj

    try:
        dj.Schema(streams.schema.database).drop()
    except Exception:
        pass

    if original is not None:
        f.write_bytes(original)
    else:
        f.unlink(missing_ok=True)


@pytest.fixture(scope="session")
def pipeline_integration(dj_config_integration, streams_schema):
    """Integration test setup with DB and streams schema.

    Provides access to:
    - DataJoint config (dj_config_integration)
    - Streams catalog tables (streams_schema)
    - Virtual streams module for queries
    """
    import datajoint as dj

    streams = dj.VirtualModule("streams", streams_schema["schema_name"])

    return {
        "config": dj_config_integration,
        "streams": streams,
        **streams_schema,
    }


@pytest.fixture
def clean_streams_tables(pipeline_integration):
    """Clean streams tables before and after test.

    Use for tests that require empty catalog tables (FK handling tests).
    Note: DeviceType.Stream is a Part table - deleting DeviceType deletes it automatically.
    """
    streams = pipeline_integration["streams"]

    # Pre-test cleanup (order matters due to FK constraints)
    streams.DeviceName().delete()
    streams.DeviceType().delete()
    streams.StreamType().delete()

    yield

    # Post-test cleanup (same order)
    streams.DeviceName().delete()
    streams.DeviceType().delete()
    streams.StreamType().delete()


# ============================================================================
# Golden Dataset Integration Test Fixtures
# ============================================================================


@pytest.fixture(
    scope="session",
    params=[
        pytest.param("foraging_abc_2025_11_18", id="abcBehav0_2025-11-18"),
        pytest.param("foraging_abc_2026_05_11", id="abcGolden01_2026-05-11"),
    ],
)
def golden_dataset_config(request):
    """Behavior golden dataset configuration.

    Parametrized over both registered behavior goldens — every test consuming
    this fixture (and its session-scoped dependents like ``full_pipeline``,
    ``test_experiment``, ``test_epochs``) runs once per dataset.
    """
    return GOLDEN_DATASETS[request.param]


@pytest.fixture(scope="session")
def require_golden_data(dj_config_integration, golden_dataset_config):
    """Skip tests if golden dataset is unavailable. Returns epoch path."""
    return _check_golden_data(golden_dataset_config)


@pytest.fixture(scope="session")
def full_pipeline(dj_config_integration, streams_schema, golden_dataset_config):
    """Full pipeline setup with all schemas for golden dataset tests.

    Since dj_config_integration sets database_prefix BEFORE importing pipeline
    modules, all schemas (lab, subject, acquisition, streams) are already
    activated with test_aeon_* databases. No manual re-decoration needed.

    Requires swc.aeon_exp (private package) — skips if not installed.

    Steps:
    1. Populate catalog from Pydantic class (populate_catalog_from_pydantic)
    2. Create ExperimentDevice and DeviceDataStream tables via streams_maker.main()
    3. Tests can then call EpochConfig.make() for DML
    """
    pytest.importorskip("swc.aeon_exp", reason="requires swc.aeon_exp package")

    from aeon.dj_pipeline import acquisition, lab, subject
    from aeon.dj_pipeline.utils import streams_maker
    from aeon.dj_pipeline.utils.load_metadata import get_experiment_pydantic, populate_catalog_from_pydantic

    cfg = golden_dataset_config

    # Step 1: Populate catalog from Pydantic class
    experiment_class = get_experiment_pydantic(cfg["devices_schema"])
    populate_catalog_from_pydantic(experiment_class)

    # Step 2: Create ExperimentDevice and DeviceDataStream tables
    streams_module = streams_maker.main(create_tables=True)

    yield {
        "lab": lab,
        "subject": subject,
        "acquisition": acquisition,
        "streams": streams_module,
    }

    # No per-iteration teardown — golden_dataset_config is parametrized, so
    # this fixture runs once per dataset. Dropping schemas between iterations
    # would invalidate the cached streams_schema fixture. Final cleanup
    # happens when the MySQL testcontainer shuts down at session end.


@pytest.fixture(scope="session")
def test_experiment(full_pipeline, require_golden_data, golden_dataset_config):
    """Create experiment from golden dataset config."""
    lab = full_pipeline["lab"]
    acquisition = full_pipeline["acquisition"]
    cfg = golden_dataset_config

    # Insert required lookup entries (Arena and Location are Lookup tables with pre-populated contents)
    # Arena schema: arena_name, arena_description, arena_shape, arena_x_dim, arena_y_dim, arena_z_dim
    lab.Arena.insert1(
        {
            "arena_name": cfg["arena_name"],
            "arena_description": f"Arena for {cfg['experiment_name']}",
            "arena_shape": "circular",
            "arena_x_dim": 2.0,  # (m) diameter
            "arena_y_dim": 2.0,  # (m) diameter
            "arena_z_dim": 0.2,  # (m) wall height
        },
        skip_duplicates=True,
    )

    # Insert DevicesSchema
    acquisition.DevicesSchema.insert1(
        {"devices_schema_name": cfg["devices_schema"]},
        skip_duplicates=True,
    )

    # Parse epoch_dir for start time
    from aeon.dj_pipeline.utils.time_utils import parse_epoch_timestamp

    epoch_dt = parse_epoch_timestamp(cfg["epoch_dir"])

    # Insert experiment
    experiment_key = {
        "experiment_name": cfg["experiment_name"],
        "experiment_start_time": epoch_dt,
        "experiment_description": f"Golden dataset test: {cfg['experiment_name']}",
        "arena_name": cfg["arena_name"],
        "lab": cfg["lab"],
        "location": cfg["location"],
        "experiment_type": cfg["experiment_type"],
    }
    acquisition.Experiment.insert1(experiment_key, skip_duplicates=True)

    # Link DevicesSchema
    acquisition.Experiment.DevicesSchema.insert1(
        {
            "experiment_name": cfg["experiment_name"],
            "devices_schema_name": cfg["devices_schema"],
        },
        skip_duplicates=True,
    )

    # Insert Directory
    acquisition.Experiment.Directory.insert1(
        {
            "experiment_name": cfg["experiment_name"],
            "directory_type": "raw",
            "repository_name": "ceph_aeon",
            "directory_path": f"raw/{cfg['experiment_path']}",
        },
        skip_duplicates=True,
    )

    return experiment_key


@pytest.fixture(scope="session")
def test_epochs(test_experiment, full_pipeline, golden_dataset_config):
    """Ingest epochs using Epoch.ingest_epochs() to test actual ingestion logic."""
    acquisition = full_pipeline["acquisition"]
    cfg = golden_dataset_config

    # Use actual ingest_epochs function - detects epochs from filesystem
    acquisition.Epoch.ingest_epochs(cfg["experiment_name"])

    # Return list of ingested epoch keys
    epochs = (acquisition.Epoch & {"experiment_name": cfg["experiment_name"]}).keys()
    return epochs


# ============================================================================
# Ephys Golden Dataset Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def ephys_golden_dataset_config():
    """Get the ephys golden dataset configuration. No DB needed."""
    return GOLDEN_DATASETS["foraging_abc_ephys_2026_05_11"]


@pytest.fixture(scope="session")
def require_ephys_golden_data(dj_config_integration, ephys_golden_dataset_config):
    """Skip tests if ephys golden dataset is unavailable. Returns epoch path."""
    return _check_golden_data(ephys_golden_dataset_config)


@pytest.fixture(scope="session")
def ephys_full_pipeline(dj_config_integration, tmp_path_factory):
    """Full ephys pipeline setup: imports modules, configures stores, creates probe types."""
    import datajoint as dj

    store_dir = tmp_path_factory.mktemp("dj_store")
    dj.config.stores = {
        "dj_store": {
            "protocol": "file",
            "location": str(store_dir),
            "stage": str(store_dir),
        },
    }

    sorting_root = tmp_path_factory.mktemp("sorting_root")

    # Patch get_sorting_root_dir to return our temp directory
    import aeon.dj_pipeline.spike_sorting as ss_module
    from aeon.dj_pipeline import acquisition, ephys, lab, spike_sorting, spike_sorting_curation, subject

    _original_get_sorting_root = ss_module.get_sorting_root_dir
    ss_module.get_sorting_root_dir = lambda: sorting_root

    try:
        # NOTE: ProbeType + ElectrodeConfig are now populated automatically
        # by EphysEpochConfig.populate() from each epoch's ProbeInterface JSON
        # (see ephys.EphysEpochConfig.make).

        yield {
            "lab": lab,
            "subject": subject,
            "acquisition": acquisition,
            "ephys": ephys,
            "spike_sorting": spike_sorting,
            "spike_sorting_curation": spike_sorting_curation,
            "store_dir": store_dir,
            "sorting_root": sorting_root,
        }
    finally:
        # Restore the original function and drop test schemas, regardless of
        # whether the fixture body or any consumer test raised.
        ss_module.get_sorting_root_dir = _original_get_sorting_root
        _drop_test_schemas()


@pytest.fixture(scope="session")
def ctx(ephys_full_pipeline, ephys_golden_dataset_config):
    """Bundle (ephys module, spike_sorting modules, dataset config) for tests.

    Shrinks test signatures from
        def test_x(self, ephys_test_*, ephys_full_pipeline, ephys_golden_dataset_config):
    to
        def test_x(self, ephys_test_*, ctx):
    where ``ctx.ephys``, ``ctx.spike_sorting``, ``ctx.cfg`` are the
    commonly-used attributes.
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        ephys=ephys_full_pipeline["ephys"],
        spike_sorting=ephys_full_pipeline["spike_sorting"],
        spike_sorting_curation=ephys_full_pipeline["spike_sorting_curation"],
        cfg=ephys_golden_dataset_config,
    )


@pytest.fixture(scope="session")
def ephys_test_experiment(ephys_full_pipeline, require_ephys_golden_data, ephys_golden_dataset_config):
    """Create experiment, subject, and directory for ephys golden dataset."""
    from aeon.dj_pipeline import subject as subject_module

    acquisition = ephys_full_pipeline["acquisition"]
    cfg = ephys_golden_dataset_config

    subject_module.Subject.insert1(
        {
            "subject": cfg["subject"],
            "sex": "U",
            "subject_birth_date": "2024-01-01",
            "subject_description": "Golden dataset subject for ephys tests",
        },
        skip_duplicates=True,
    )

    from aeon.dj_pipeline.utils.time_utils import parse_epoch_timestamp

    epoch_dt = parse_epoch_timestamp(cfg["epoch_dir"])

    acquisition.Experiment.insert1(
        {
            "experiment_name": cfg["experiment_name"],
            "experiment_start_time": epoch_dt,
            "experiment_description": "Ephys golden dataset test",
            "arena_name": cfg["arena_name"],
            "lab": cfg["lab"],
            "location": cfg["location"],
            "experiment_type": cfg["experiment_type"],
        },
        skip_duplicates=True,
    )

    acquisition.Experiment.Subject.insert1(
        {"experiment_name": cfg["experiment_name"], "subject": cfg["subject"]},
        skip_duplicates=True,
    )

    # Split raw: "raw-ephys" for AEONX1 ephys data
    acquisition.Experiment.Directory.insert1(
        {
            "experiment_name": cfg["experiment_name"],
            "directory_type": "raw-ephys",
            "repository_name": "ceph_aeon",
            "directory_path": f"raw/{cfg['experiment_path']}",
        },
        skip_duplicates=True,
    )

    return {"experiment_name": cfg["experiment_name"], "epoch_start": epoch_dt}


@pytest.fixture(scope="session")
def ephys_test_epochs(
    ephys_test_experiment,
    ephys_full_pipeline,
    ephys_golden_dataset_config,
    require_ephys_golden_data,
    tmp_path_factory,
):
    """Drive the ephys ingest chain end-to-end against the golden epoch.

    1. ``EphysEpoch.ingest_epochs(exp)`` — reads first HarpSync CSV to compute
       HARP-clock epoch_start. No behavior CSV dependency.
    2. ``EphysEpochConfig.populate()`` — probe discovery, ProbeInsertion setup,
       per-probe ElectrodeConfig registration via create_electrode_config.
    3. ``EphysSyncModel.ingest(exp)`` — HARP↔ONIX regression per chunk.

    Returns the EphysEpoch rows for the experiment.
    """
    import json

    ephys = ephys_full_pipeline["ephys"]
    cfg = ephys_golden_dataset_config
    exp_name = cfg["experiment_name"]

    # Write probe_assignments.json to temp dir for EphysEpochConfig.populate()
    override_dir = tmp_path_factory.mktemp("probe_assignments")
    assignments = {
        "version": 1,
        "probe_assignments": {
            cfg["probe_serial"]: {"subject": cfg["subject"]},
        },
    }
    (override_dir / "probe_assignments.json").write_text(json.dumps(assignments))

    # Step 1: discover epochs (HARP-native epoch_start from first HarpSync CSV)
    ephys.EphysEpoch.ingest_epochs(exp_name)

    # Step 2: probe discovery + per-probe ElectrodeConfig registration
    ephys.EphysEpochConfig.probe_assignments_override_dir = override_dir
    try:
        ephys.EphysEpochConfig.populate(suppress_errors=False)
    finally:
        ephys.EphysEpochConfig.probe_assignments_override_dir = None

    # Step 3: sync models from HarpSync CSVs — required before EphysChunk.ingest_chunks
    ephys.EphysSyncModel.ingest(exp_name)

    return (ephys.EphysEpoch & {"experiment_name": exp_name}).to_dicts()


@pytest.fixture(scope="session")
def ephys_test_blocks(ephys_test_epochs, ephys_full_pipeline, ephys_golden_dataset_config):
    """Create EphysBlock entry for the golden dataset — single 35-minute block.

    Uses the HARP-native ``epoch_start`` from the EphysEpoch row (not the dir
    name, which is ONIX-wall-clock).
    """
    from datetime import timedelta

    ephys = ephys_full_pipeline["ephys"]
    cfg = ephys_golden_dataset_config
    exp_name = cfg["experiment_name"]

    probe_insertions = (ephys.ProbeInsertion & {"experiment_name": exp_name}).to_dicts()
    epoch_start = ephys_test_epochs[0]["epoch_start"]  # HARP-native

    for pi in probe_insertions:
        ephys.EphysBlock.insert1(
            {
                "experiment_name": exp_name,
                "subject": pi["subject"],
                "insertion_number": pi["insertion_number"],
                "block_start": epoch_start,
                "block_end": epoch_start + timedelta(minutes=35),
            },
            skip_duplicates=True,
        )

    return (ephys.EphysBlock & {"experiment_name": exp_name}).to_dicts()


@pytest.fixture(scope="session")
def ephys_chunks_ingested(ephys_test_epochs, ctx):
    """Run EphysChunk.ingest_chunks once for the golden dataset."""
    ctx.ephys.EphysChunk.ingest_chunks(ctx.cfg["experiment_name"])
    return (
        ctx.ephys.EphysChunk & {"experiment_name": ctx.cfg["experiment_name"]}
    ).to_dicts()


@pytest.fixture(scope="session")
def ephys_block_info_populated(ephys_chunks_ingested, ephys_test_blocks, ctx):
    """Run EphysBlockInfo.populate restricted to the golden experiment."""
    ctx.ephys.EphysBlockInfo.populate(
        {"experiment_name": ctx.cfg["experiment_name"]},
        display_progress=False,
        suppress_errors=False,
    )
    return (
        ctx.ephys.EphysBlockInfo & {"experiment_name": ctx.cfg["experiment_name"]}
    ).to_dicts()


@pytest.fixture(scope="session")
def ephys_sorting_setup(ephys_test_blocks, ephys_full_pipeline, ephys_golden_dataset_config):
    """Set up sorting prerequisites: ElectrodeGroup, SortingParamSet, SortingTask."""
    spike_sorting = ephys_full_pipeline["spike_sorting"]
    ephys = ephys_full_pipeline["ephys"]
    cfg = ephys_golden_dataset_config
    exp_name = cfg["experiment_name"]

    # ElectrodeConfig was populated by create_electrode_config from the JSON.
    electrode_config_key = {
        "probe_type": cfg["probe_type"],
        "electrode_config_name": cfg["electrode_config_name"],
    }
    spike_sorting.ElectrodeGroup.insert1(
        {
            **electrode_config_key,
            "electrode_group": "shank3",
            "electrode_group_description": "8 electrodes on shank3 (golden dataset)",
            "electrode_count": len(cfg["electrodes"]),
        },
        skip_duplicates=True,
    )
    # 8-electrode subset from cfg (3982-3989), not the full 384 in ElectrodeConfig.
    spike_sorting.ElectrodeGroup.Electrode.insert(
        (
            {**electrode_config_key, "electrode_group": "shank3", "electrode": e}
            for e in cfg["electrodes"]
        ),
        skip_duplicates=True,
    )

    spike_sorting.SortingParamSet.insert1(
        {
            "paramset_id": "400",
            "sorting_method": "kilosort4",
            "paramset_description": "KS4 golden dataset (8-ch, no drift correction)",
            "params": {
                "SI_SORTING_PARAMS": {
                    "nblocks": 0,
                    "Th_universal": 8,
                    "Th_learned": 7,
                    "do_CAR": False,
                    "skip_kilosort_preprocessing": False,
                    "clear_cache": True,
                },
                "SI_POSTPROCESSING_PARAMS": {
                    "extensions": {
                        "random_spikes": {},
                        "waveforms": {},
                        "templates": {},
                        "noise_levels": {},
                        "spike_amplitudes": {},
                        "spike_locations": {},
                        "quality_metrics": {},
                    },
                    "job_kwargs": {"n_jobs": 1, "chunk_duration": "1s"},
                },
            },
        },
        skip_duplicates=True,
    )

    blocks = (ephys.EphysBlock & {"experiment_name": exp_name}).to_dicts()
    for block in blocks:
        spike_sorting.SortingTask.insert1(
            {
                "experiment_name": block["experiment_name"],
                "subject": block["subject"],
                "insertion_number": block["insertion_number"],
                "block_start": block["block_start"],
                "block_end": block["block_end"],
                **electrode_config_key,
                "electrode_group": "shank3",
                "paramset_id": "400",
            },
            skip_duplicates=True,
        )

    return {
        "electrode_config_key": electrode_config_key,
        "electrode_group": "shank3",
        "paramset_id": "400",
    }


@pytest.fixture(scope="session")
def ephys_sorting_injected(
    ephys_sorting_setup,
    ephys_full_pipeline,
    ephys_golden_dataset_config,
    require_ephys_golden_data,
):
    """Run PreProcessing.populate() and force-inject SpikeSorting from golden data.

    Runs the full cascade up through PreProcessing, then copies
    pre-computed KS4 sorting output into the test's output directory and
    inserts SpikeSorting + SpikeSorting.File entries. Downstream tables
    (PostProcessing, SortedSpikes, SyncedSpikes) can then populate normally.
    """
    import shutil
    from datetime import UTC, datetime

    spike_sorting = ephys_full_pipeline["spike_sorting"]
    ephys = ephys_full_pipeline["ephys"]
    cfg = ephys_golden_dataset_config

    # Run prerequisite populates
    ephys.EphysChunk.ingest_chunks(cfg["experiment_name"])
    ephys.EphysBlockInfo.populate(display_progress=True, suppress_errors=False)
    spike_sorting.PreProcessing.populate(display_progress=True, suppress_errors=False)

    # Find the output_dir that PreProcessing created
    sorting_task_keys = (spike_sorting.SortingTask & {"experiment_name": cfg["experiment_name"]}).to_dicts()
    assert len(sorting_task_keys) >= 1, "No SortingTask entries found"

    key = sorting_task_keys[0]
    output_dir = spike_sorting.PreProcessing.infer_output_dir(key)

    # Locate golden sorting output
    epoch_path = require_ephys_golden_data
    golden_sorting_dir = epoch_path.parent / cfg["golden_sorting_dir"]
    golden_sorting_output = golden_sorting_dir / "sorting_output"
    if not golden_sorting_output.exists():
        pytest.skip(f"Golden sorting output not found: {golden_sorting_output}")

    # Copy golden sorting output into the pipeline's expected location
    test_sorting_dir = output_dir / "spike_sorting"
    if not test_sorting_dir.exists():
        shutil.copytree(golden_sorting_output, test_sorting_dir)

    # Re-save si_sorting.pkl with correct relative_to for this output_dir
    import spikeinterface as si

    si_native_dir = test_sorting_dir / "in_container_sorting"
    sorting = si.load(si_native_dir)
    sorting.dump_to_pickle(test_sorting_dir / "si_sorting.pkl", relative_to=output_dir)

    # Force-inject SpikeSorting entry
    if not (spike_sorting.SpikeSorting & key):
        spike_sorting.SpikeSorting.insert1(
            {
                **key,
                "execution_time": datetime.now(UTC),
                "execution_duration": 0.0,
            },
            allow_direct_insert=True,
        )
        spike_sorting.SpikeSorting.File.insert(
            [
                {
                    **key,
                    "file_name": f.relative_to(test_sorting_dir).as_posix(),
                    "file": f,
                }
                for f in test_sorting_dir.rglob("*")
                if f.is_file()
            ],
            allow_direct_insert=True,
        )

    return {
        "output_dir": output_dir,
        "sorting_dir": test_sorting_dir,
        "sorting_task_key": key,
    }
