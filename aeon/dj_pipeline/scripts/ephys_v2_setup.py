"""Setup and run script for the ephys v2 pipeline test.

Tests the full ephys pipeline end-to-end using AEONX1/social-ephys0.1 data,
with the v2 PK structure (experiment_name, subject, insertion_number).

Three phases:
  Phase 1: Setup & ingestion (no SLURM needed)
  Phase 2: Spike sorting (requires SLURM for SpikeSorting step)
  Phase 3: Post-sorting, curation & unit matching (no SLURM needed)

Prerequisites:
  - dj_local_conf.json configured with prefix "u_elissas_aeon_ephys_v2_test_"
  - On HPC with access to /ceph/aeon/
  - SpikeInterface installed (standard version, or Elissa's fork if needed)

Usage:
  uv run python -m aeon.dj_pipeline.scripts.ephys_v2_setup              # Run all Phase 1
  uv run python -m aeon.dj_pipeline.scripts.ephys_v2_setup --phase 2    # Run Phase 2
  uv run python -m aeon.dj_pipeline.scripts.ephys_v2_setup --step 5     # Run single step
  uv run python -m aeon.dj_pipeline.scripts.ephys_v2_setup --dry-run    # Preview
"""

import argparse
import sys
import uuid

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "social-ephys0.1-aeon3"
PRODUCTION_PREFIX = "aeon_"

# Subject (fake — real subject-probe mapping not yet implemented)
SUBJECT = "test-subject-001"

# Probe config (from previous test run — ephys_test_ingestion.py)
PROBE_NAME = "NP2004-001"
PROBE_TYPE = "neuropixels - NP2004"
ELECTRODE_CONFIG_NAME = "0-383"
N_ELECTRODES = 384

# The epoch directory that contains our test data files
TARGET_EPOCH_DIR = "2024-06-04T10-24-07"
PROBE_LABEL = "ProbeA"  # Only sort ProbeA (single probe test)

# Block schedule: 4 blocks × 3 hours, 1-hour overlap
BLOCK_START = "2024-06-04 11:00:00"
BLOCK_DURATION_HOURS = 3
BLOCK_ADVANCE_HOURS = 2  # 3 - 1 overlap = advance by 2
N_BLOCKS = 4

# Sorting config
PARAMSET_ID = 400
SORTING_METHOD = "kilosort4"
ELECTRODE_GROUP_NAME = "0-95"
ELECTRODE_GROUP_SIZE = 96

# Controls whether populate() raises on first error
SUPPRESS_ERRORS = False


# ---------------------------------------------------------------------------
# Safety: verify DB prefix before ANY pipeline imports
# ---------------------------------------------------------------------------
def verify_prefix_or_exit():
    """Check the database prefix BEFORE importing any pipeline modules.

    Critical because `from aeon.dj_pipeline import ephys` triggers
    `dj.Schema(get_schema_name("ephys"))` at import time, which CREATES
    schemas in the database.

    These setup/validation scripts are designed for testing only and should
    never be run against the production database.
    """
    import datajoint as dj

    prefix = dj.config.database.database_prefix or ""
    host = dj.config.database.host or ""

    if not prefix:
        print(f"\n  ✗ SAFETY CHECK FAILED: database prefix is empty.")
        print(f"    dj_local_conf.json may not have been found.")
        print(f"    Make sure you run from the repo root directory.")
        sys.exit(1)

    if prefix == PRODUCTION_PREFIX:
        print(f"\n  ✗ SAFETY CHECK FAILED: database prefix is '{prefix}' (production).")
        print(f"    This script is for testing only — do not run against production.")
        print(f"    Set a test prefix in dj_local_conf.json, e.g.:")
        print(f'      "custom": {{"database.prefix": "u_yourname_test_"}}')
        sys.exit(1)

    if "aeon-db2" in host:
        print(f"\n  ✗ SAFETY CHECK FAILED: connecting to production host '{host}'.")
        print(f"    This script is for testing only — do not run against production.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def print_header(step_num, title):
    print(f"\n{'='*60}")
    print(f"  Step {step_num}: {title}")
    print(f"{'='*60}\n")


def print_ok(msg):
    print(f"  ✓ {msg}")


def print_fail(msg):
    print(f"  ✗ {msg}")


def print_info(msg):
    print(f"  → {msg}")


# ===========================================================================
# PHASE 1: Setup & Ingestion
# ===========================================================================

def step_verify_config(dry_run=False):
    """Step 1: Verify DB config, ceph path, connection."""
    print_header(1, "Verify Configuration")

    if dry_run:
        print_info("Would check: dj_local_conf.json, DB connection, ceph path")
        return True

    import datajoint as dj

    prefix = dj.config.database.database_prefix or ""
    print_ok(f"Database prefix: {prefix}")

    try:
        conn = dj.conn()
        print_ok(f"Connected to {dj.config.database.host}:{dj.config.database.port}")
    except Exception as e:
        print_fail(f"Cannot connect to database: {e}")
        return False

    from aeon.dj_pipeline.utils.paths import get_repository_path

    try:
        repo_path = get_repository_path("ceph_aeon")
        print_ok(f"Repository path: {repo_path}")
    except Exception as e:
        print_fail(f"Cannot resolve repository path: {e}")
        return False

    data_dir = repo_path / "aeon" / "data" / "raw" / "AEONX1" / "social-ephys0.1"
    if not data_dir.exists():
        print_fail(f"Data directory not found: {data_dir}")
        return False
    print_ok(f"Data directory exists: {data_dir}")

    return True


def step_create_experiment(dry_run=False):
    """Step 2: Register experiment + directories + subject."""
    print_header(2, "Create Experiment & Subject")

    if dry_run:
        print_info(f"Would create experiment: {EXPERIMENT_NAME}")
        print_info(f"Would register raw dir: aeon/data/raw/AEONX1/social-ephys0.1")
        print_info(f"Would insert subject: {SUBJECT}")
        return True

    from aeon.dj_pipeline import acquisition, subject

    # Insert into Subject table first (Experiment.Subject has FK to it)
    subject.Subject.insert1(
        {
            "subject": SUBJECT,
            "sex": "U",
            "subject_birth_date": "2024-01-01",
            "subject_description": "Test subject for ephys v2 pipeline",
        },
        skip_duplicates=True,
    )
    print_ok(f"Subject inserted: {SUBJECT}")

    acquisition.Experiment.insert1(
        {
            "experiment_name": EXPERIMENT_NAME,
            "experiment_start_time": "2024-06-01 06:00:00",
            "experiment_description": "social ephys experiment 0.1 - AEON3",
            "arena_name": "circle-2m",
            "lab": "SWC",
            "location": "AEON3",
            "experiment_type": "social",
        },
        skip_duplicates=True,
    )
    print_ok(f"Experiment registered: {EXPERIMENT_NAME}")

    acquisition.Experiment.Directory.insert(
        [
            {
                "experiment_name": EXPERIMENT_NAME,
                "directory_type": "raw",
                "repository_name": "ceph_aeon",
                "directory_path": "aeon/data/raw/AEONX1/social-ephys0.1",
                "load_order": 0,
            },
        ],
        skip_duplicates=True,
    )
    print_ok("Raw directory registered")

    # Insert subject into Experiment.Subject
    acquisition.Experiment.Subject.insert1(
        {
            "experiment_name": EXPERIMENT_NAME,
            "subject": SUBJECT,
        },
        skip_duplicates=True,
    )
    print_ok(f"Subject registered: {SUBJECT}")

    # Verify
    exp = (acquisition.Experiment & {"experiment_name": EXPERIMENT_NAME}).fetch1()
    print_ok(f"  arena: {exp['arena_name']}, location: {exp['location']}")

    dirs = (acquisition.Experiment.Directory & {"experiment_name": EXPERIMENT_NAME}).to_dicts()
    for d in dirs:
        print_ok(f"  directory: {d['directory_type']} → {d['directory_path']}")

    subjects = (acquisition.Experiment.Subject & {"experiment_name": EXPERIMENT_NAME}).to_arrays("subject")
    print_ok(f"  subjects: {list(subjects)}")

    return True


def step_insert_probe_config(dry_run=False):
    """Step 3: Insert ProbeType, Probe, ElectrodeConfig (Lookup tables)."""
    print_header(3, "Insert Probe Configuration")

    if dry_run:
        print_info(f"Would create ProbeType: {PROBE_TYPE}")
        print_info(f"Would create Probe: {PROBE_NAME}")
        print_info(f"Would create ElectrodeConfig: {ELECTRODE_CONFIG_NAME} ({N_ELECTRODES} electrodes)")
        return True

    from aeon.dj_pipeline import ephys

    # ProbeType (manual insert — probeinterface needs internet, HPC doesn't have it)
    if not (ephys.ProbeType & {"probe_type": PROBE_TYPE}):
        ephys.ProbeType.insert1({"probe_type": PROBE_TYPE})
        # NP2004 = 4-shank, 384 electrodes per shank, 2-column layout
        # We only need basic geometry for the electrodes we're sorting
        electrodes = []
        for e in range(N_ELECTRODES):
            electrodes.append({
                "probe_type": PROBE_TYPE,
                "electrode": e,
                "shank": 0,
                "x_coord": (e % 2) * 32.0,  # 2-column, 32um spacing
                "y_coord": (e // 2) * 15.0,  # 15um vertical pitch
            })
        ephys.ProbeType.Electrode.insert(electrodes)
        print_ok(f"ProbeType created: {PROBE_TYPE} ({N_ELECTRODES} electrodes)")
    else:
        print_ok(f"ProbeType already exists: {PROBE_TYPE}")

    # Probe
    ephys.Probe.insert1(
        {"probe": PROBE_NAME, "probe_type": PROBE_TYPE, "probe_comment": ""},
        skip_duplicates=True,
    )
    print_ok(f"Probe registered: {PROBE_NAME}")

    # ElectrodeConfig
    electrode_config_key = {
        "probe_type": PROBE_TYPE,
        "electrode_config_name": ELECTRODE_CONFIG_NAME,
    }
    ephys.ElectrodeConfig.insert1(
        {
            **electrode_config_key,
            "electrode_config_description": "",
            "electrode_config_hash": uuid.uuid4(),
        },
        skip_duplicates=True,
    )
    print_ok(f"ElectrodeConfig: {ELECTRODE_CONFIG_NAME}")

    # Electrode entries
    existing = len(ephys.ElectrodeConfig.Electrode & electrode_config_key)
    if existing < N_ELECTRODES:
        ephys.ElectrodeConfig.Electrode.insert(
            (
                {**electrode_config_key, "electrode": e}
                for e in range(N_ELECTRODES)
            ),
            skip_duplicates=True,
        )
        print_ok(f"ElectrodeConfig.Electrode: {N_ELECTRODES} electrodes")
    else:
        print_ok(f"ElectrodeConfig.Electrode already has {existing} electrodes")

    return True


def step_ingest_epochs(dry_run=False):
    """Step 4: Skip — epoch ingestion not needed for ephys-only experiments.

    The social-ephys0.1 data has no CameraTop device, so Epoch.ingest_epochs()
    (which discovers epochs via chunk files for a reference device) finds nothing.
    Step 5 manually inserts the EphysEpoch we need instead.
    """
    print_header(4, "Ingest Acquisition Epochs (skipped)")

    print_info("Skipped — ephys-only experiment has no reference device for epoch discovery")
    print_info("(Epoch.ingest_epochs() needs CameraTop/FrameTop chunk files; this data only has Environment + NeuropixelsV2Beta)")
    print_info("Step 5 will manually insert the Epoch + EphysEpoch entries we need")

    return True


def step_manual_ephys_setup(dry_run=False):
    """Step 5: Manual ProbeInsertion + EphysEpoch setup (bypasses read_probe_assignments).

    Since read_probe_assignments() raises NotImplementedError, we manually:
    1. Insert ProbeInsertion (with subject)
    2. Insert Epoch + EphysEpoch for the target epoch
    3. Insert EphysEpoch.Insertion linking the probe to the epoch
    """
    print_header(5, "Manual Ephys Setup (ProbeInsertion + Epoch + EphysEpoch)")

    if dry_run:
        print_info(f"Would insert ProbeInsertion: subject={SUBJECT}, insertion=1, probe={PROBE_NAME}")
        print_info(f"Would insert Epoch + EphysEpoch for epoch dir: {TARGET_EPOCH_DIR}")
        print_info(f"Would insert EphysEpoch.Insertion: probe_label={PROBE_LABEL}")
        return True

    from datetime import datetime
    from aeon.dj_pipeline import acquisition, ephys

    # 1. Insert ProbeInsertion
    ephys.ProbeInsertion.insert1(
        {
            "experiment_name": EXPERIMENT_NAME,
            "subject": SUBJECT,
            "insertion_number": 1,
            "probe": PROBE_NAME,
        },
        skip_duplicates=True,
    )
    print_ok(f"ProbeInsertion: subject={SUBJECT}, insertion=1, probe={PROBE_NAME}")

    # 2. Insert Epoch directly (epoch ingestion skipped — no reference device)
    target_epoch_start = datetime.strptime(TARGET_EPOCH_DIR, "%Y-%m-%dT%H-%M-%S")
    acquisition.Epoch.insert1(
        {
            "experiment_name": EXPERIMENT_NAME,
            "epoch_start": target_epoch_start,
            "directory_type": "raw",
            "epoch_dir": TARGET_EPOCH_DIR,
        },
        skip_duplicates=True,
    )
    print_ok(f"Epoch inserted: {TARGET_EPOCH_DIR} → {target_epoch_start}")

    # 3. Insert EphysEpoch master row
    epoch_key = {"experiment_name": EXPERIMENT_NAME, "epoch_start": target_epoch_start}
    ephys.EphysEpoch.insert1(
        {**epoch_key, "has_ephys": True, "n_probes": 1},
        skip_duplicates=True,
        allow_direct_insert=True,
    )
    print_ok(f"EphysEpoch: has_ephys=True, n_probes=1")

    # 4. Insert EphysEpoch.Insertion part row
    ephys.EphysEpoch.Insertion.insert1(
        {
            **epoch_key,
            "subject": SUBJECT,
            "insertion_number": 1,
            "probe_label": PROBE_LABEL,
        },
        skip_duplicates=True,
        allow_direct_insert=True,
    )
    print_ok(f"EphysEpoch.Insertion: probe_label={PROBE_LABEL}")

    # Verify
    pi = (ephys.ProbeInsertion & {"experiment_name": EXPERIMENT_NAME}).to_dicts()
    print_ok(f"ProbeInsertion count: {len(pi)}")
    ei = (ephys.EphysEpoch.Insertion & {"experiment_name": EXPERIMENT_NAME}).to_dicts()
    print_ok(f"EphysEpoch.Insertion count: {len(ei)}")

    return True


def step_ingest_chunks(dry_run=False):
    """Step 6: Run ingest_chunks(experiment_name) — creates EphysChunk entries."""
    print_header(6, "Ingest Ephys Chunks")

    if dry_run:
        print_info(f"Would call: EphysChunk.ingest_chunks('{EXPERIMENT_NAME}')")
        return True

    from aeon.dj_pipeline import ephys

    print_info("Ingesting ephys chunks...")
    ephys.EphysChunk.ingest_chunks(EXPERIMENT_NAME)

    chunks = (ephys.EphysChunk & {"experiment_name": EXPERIMENT_NAME}).to_dicts()
    if not chunks:
        print_fail("No EphysChunk entries created")
        return False

    print_ok(f"EphysChunk entries: {len(chunks)}")

    # Show per-insertion counts
    from collections import Counter
    insertion_counts = Counter(c["insertion_number"] for c in chunks)
    for ins_num, count in sorted(insertion_counts.items()):
        print_info(f"  insertion {ins_num}: {count} chunks")

    return True


def step_create_blocks(dry_run=False):
    """Step 7: Create EphysBlock entries (4 blocks × 3h, 1h overlap) for each probe."""
    print_header(7, "Create EphysBlock Entries")

    import pandas as pd

    block_start = pd.Timestamp(BLOCK_START)
    blocks_to_create = []
    for i in range(N_BLOCKS):
        start = block_start + pd.Timedelta(hours=i * BLOCK_ADVANCE_HOURS)
        end = start + pd.Timedelta(hours=BLOCK_DURATION_HOURS)
        blocks_to_create.append((start, end))

    if dry_run:
        for i, (start, end) in enumerate(blocks_to_create, 1):
            print_info(f"Block {i}: {start} → {end}")
        print_info("Would create blocks for each ProbeInsertion")
        return True

    from aeon.dj_pipeline import ephys

    probe_insertions = (ephys.ProbeInsertion & {"experiment_name": EXPERIMENT_NAME}).to_dicts()
    if not probe_insertions:
        print_fail("No ProbeInsertions found — run step 5 first")
        return False

    total_blocks = 0
    for pi in probe_insertions:
        for start, end in blocks_to_create:
            block_key = {
                "experiment_name": EXPERIMENT_NAME,
                "subject": pi["subject"],
                "insertion_number": pi["insertion_number"],
                "block_start": start,
                "block_end": end,
            }
            ephys.EphysBlock.insert1(block_key, skip_duplicates=True)
            total_blocks += 1

    print_ok(f"Created {total_blocks} EphysBlock entries ({len(probe_insertions)} probes × {N_BLOCKS} blocks)")

    for i, (start, end) in enumerate(blocks_to_create, 1):
        print_info(f"  Block {i}: {start} → {end}")

    return True


def step_populate_block_info(dry_run=False):
    """Step 8: Run EphysBlockInfo.populate()."""
    print_header(8, "Populate EphysBlockInfo")

    if dry_run:
        print_info("Would call: EphysBlockInfo.populate()")
        return True

    from aeon.dj_pipeline import ephys

    print_info("Running EphysBlockInfo.populate()...")
    ephys.EphysBlockInfo.populate(
        display_progress=True, suppress_errors=SUPPRESS_ERRORS
    )

    block_infos = (ephys.EphysBlockInfo & {"experiment_name": EXPERIMENT_NAME}).to_dicts()
    if not block_infos:
        print_fail("No EphysBlockInfo entries — populate may have failed")
        return False

    print_ok(f"EphysBlockInfo entries: {len(block_infos)}")

    # Check chunks per block
    for bi in block_infos:
        chunk_count = len(ephys.EphysBlockInfo.Chunk & bi)
        print_info(
            f"  insertion {bi['insertion_number']}, "
            f"{bi['block_start']} → {bi['block_end']}: "
            f"{bi['block_duration']:.1f}h, {chunk_count} chunks"
        )

    return True


# ===========================================================================
# PHASE 2: Spike Sorting
# ===========================================================================

def step_create_electrode_groups(dry_run=False):
    """Step 9: Create single electrode group (0-95, 96 channels)."""
    print_header(9, "Create Electrode Group")

    if dry_run:
        print_info(f"Would create 1 group: {ELECTRODE_GROUP_NAME} ({ELECTRODE_GROUP_SIZE} channels)")
        return True

    from aeon.dj_pipeline import spike_sorting, ephys

    electrode_config_key = {
        "probe_type": PROBE_TYPE,
        "electrode_config_name": ELECTRODE_CONFIG_NAME,
    }

    group_electrodes = list(range(ELECTRODE_GROUP_SIZE))

    spike_sorting.ElectrodeGroup.insert1(
        {
            **electrode_config_key,
            "electrode_group": ELECTRODE_GROUP_NAME,
            "electrode_group_description": f"electrodes {ELECTRODE_GROUP_NAME}",
            "electrode_count": len(group_electrodes),
        },
        skip_duplicates=True,
    )
    spike_sorting.ElectrodeGroup.Electrode.insert(
        (
            {**electrode_config_key, "electrode_group": ELECTRODE_GROUP_NAME, "electrode": e}
            for e in group_electrodes
        ),
        skip_duplicates=True,
    )
    print_ok(f"  Group {ELECTRODE_GROUP_NAME} ({len(group_electrodes)} electrodes)")

    return True


def step_insert_sorting_params(dry_run=False):
    """Step 10: Insert SortingParamSet (Kilosort4)."""
    print_header(10, "Insert Sorting Parameters")

    if dry_run:
        print_info(f"Would insert paramset_id={PARAMSET_ID} ({SORTING_METHOD})")
        return True

    from aeon.dj_pipeline import spike_sorting

    if not (spike_sorting.SortingParamSet & {"paramset_id": PARAMSET_ID}):
        params = {
            "SI_PREPROCESSING_METHOD": "ephys_preproc",
            "SI_SORTING_PARAMS": {
                "n_pcs": 3,
                "do_CAR": False,
                "keep_good_only": True,
                "use_binary_file": True,
            },
            "SI_POSTPROCESSING_PARAMS": {
                "extensions": {
                    "random_spikes": {},
                    "waveforms": {},
                    "templates": {},
                    "noise_levels": {},
                    "correlograms": {},
                    "isi_histograms": {},
                    "principal_components": {"n_components": 5, "mode": "by_channel_local"},
                    "spike_amplitudes": {},
                    "spike_locations": {},
                    "template_metrics": {"include_multi_channel_metrics": True},
                    "template_similarity": {},
                    "unit_locations": {},
                    "quality_metrics": {},
                },
                "job_kwargs": {"n_jobs": 0.8, "chunk_duration": "1s"},
                "export_to_phy": False,
                "export_report": True,
            },
        }
        spike_sorting.SortingParamSet.insert1(
            {
                "paramset_id": PARAMSET_ID,
                "sorting_method": SORTING_METHOD,
                "paramset_description": "Default parameter set for Kilosort4 with SpikeInterface",
                "params": params,
            },
        )
        print_ok(f"Inserted paramset_id={PARAMSET_ID} ({SORTING_METHOD})")
    else:
        print_ok(f"SortingParamSet {PARAMSET_ID} already exists")

    return True


def step_create_sorting_tasks(dry_run=False):
    """Step 11: Create SortingTask entries (block × electrode group)."""
    print_header(11, "Create SortingTask Entries")

    if dry_run:
        print_info("Would create SortingTask for each block × electrode group")
        return True

    from aeon.dj_pipeline import ephys, spike_sorting

    blocks = (ephys.EphysBlock & {"experiment_name": EXPERIMENT_NAME}).to_dicts()

    if not blocks:
        print_fail("No EphysBlock entries — run step 7 first")
        return False

    count = 0
    for block in blocks:
        task_key = {
            "experiment_name": block["experiment_name"],
            "subject": block["subject"],
            "insertion_number": block["insertion_number"],
            "block_start": block["block_start"],
            "block_end": block["block_end"],
            "probe_type": PROBE_TYPE,
            "electrode_config_name": ELECTRODE_CONFIG_NAME,
            "electrode_group": ELECTRODE_GROUP_NAME,
            "paramset_id": PARAMSET_ID,
        }
        spike_sorting.SortingTask.insert1(task_key, skip_duplicates=True)
        count += 1

    print_ok(f"Created {count} SortingTask entries")
    return True


def step_preprocessing(dry_run=False):
    """Step 12: Run PreProcessing.populate()."""
    print_header(12, "Run PreProcessing")

    if dry_run:
        print_info("Would call: PreProcessing.populate()")
        return True

    from aeon.dj_pipeline import spike_sorting

    print_info("Running PreProcessing.populate()...")
    spike_sorting.PreProcessing.populate(
        display_progress=True, suppress_errors=SUPPRESS_ERRORS
    )

    count = len(spike_sorting.PreProcessing & {"experiment_name": EXPERIMENT_NAME})
    print_ok(f"PreProcessing entries: {count}")

    return True


def step_spike_sorting_slurm(dry_run=False):
    """Step 13: Submit SpikeSorting via SLURM."""
    print_header(13, "Submit SpikeSorting (SLURM)")

    print_info("*** SpikeSorting requires SLURM submission ***")
    print_info("Run the SLURM submission script separately:")
    print_info("  sbatch <slurm_script.sh>")
    print_info("")
    print_info("After SLURM jobs complete, continue with --step 14")
    print_info("")
    print_info("IMPORTANT: Do NOT change code while SLURM jobs are queued!")

    if dry_run:
        return True

    # Show what needs to be sorted
    from aeon.dj_pipeline import spike_sorting

    pending = len(spike_sorting.SortingTask - spike_sorting.SpikeSorting)
    done = len(spike_sorting.SpikeSorting & {"experiment_name": EXPERIMENT_NAME})
    print_info(f"  Pending: {pending} tasks")
    print_info(f"  Complete: {done} tasks")

    return True


def step_post_processing(dry_run=False):
    """Step 14: Run PostProcessing.populate()."""
    print_header(14, "Run PostProcessing")

    if dry_run:
        print_info("Would call: PostProcessing.populate()")
        return True

    from aeon.dj_pipeline import spike_sorting

    print_info("Running PostProcessing.populate()...")
    spike_sorting.PostProcessing.populate(
        display_progress=True, suppress_errors=SUPPRESS_ERRORS
    )

    count = len(spike_sorting.PostProcessing & {"experiment_name": EXPERIMENT_NAME})
    print_ok(f"PostProcessing entries: {count}")

    return True


def step_sorted_spikes(dry_run=False):
    """Step 15: Run SortedSpikes.populate()."""
    print_header(15, "Run SortedSpikes")

    if dry_run:
        print_info("Would call: SortedSpikes.populate()")
        return True

    from aeon.dj_pipeline import spike_sorting

    print_info("Running SortedSpikes.populate()...")
    spike_sorting.SortedSpikes.populate(
        display_progress=True, suppress_errors=SUPPRESS_ERRORS
    )

    count = len(spike_sorting.SortedSpikes & {"experiment_name": EXPERIMENT_NAME})
    print_ok(f"SortedSpikes entries: {count}")

    units = len(spike_sorting.SortedSpikes.Unit & {"experiment_name": EXPERIMENT_NAME})
    print_ok(f"SortedSpikes.Unit entries: {units}")

    return True


# ===========================================================================
# PHASE 3: Curation, Sync & Unit Matching
# ===========================================================================

def step_approve_auto_curation(dry_run=False):
    """Step 16: Approve automatic curation (no manual curation needed).

    Creates ManualCuration + OfficialCuration entries for each block, then
    runs ApplyOfficialCuration.populate() which auto-detects the "no curation
    file + parent=-1" pattern and approves the raw sorting as official.
    """
    print_header(16, "Approve Automatic Curation")

    if dry_run:
        print_info("Would insert ManualCuration + OfficialCuration for each block")
        print_info("Would run ApplyOfficialCuration.populate() (auto-approval path)")
        print_info("Auto-sorted results treated as official (no manual curation)")
        return True

    from datetime import datetime, timezone
    from aeon.dj_pipeline import spike_sorting
    from aeon.dj_pipeline import spike_sorting_curation as curation

    # Ensure CurationMethod has the entry we need
    if not (curation.CurationMethod & {"curation_method": "SpikeInterface"}):
        curation.CurationMethod.insert1(
            {"curation_method": "SpikeInterface"},
            skip_duplicates=True,
        )
        print_ok("CurationMethod 'SpikeInterface' inserted")

    # For each SpikeSorting entry, create ManualCuration + OfficialCuration
    sorting_entries = (spike_sorting.SpikeSorting & {"experiment_name": EXPERIMENT_NAME}).keys()
    now = datetime.now(timezone.utc)

    mc_count = 0
    oc_count = 0

    for sorting_key in sorting_entries:
        # ManualCuration: curation_id=0, parent=-1 (based on raw sorting)
        mc_key = {**sorting_key, "curation_id": 0}
        if not (curation.ManualCuration & mc_key):
            curation.ManualCuration.insert1(
                {
                    **mc_key,
                    "curation_datetime": now,
                    "parent_curation_id": -1,
                    "curation_method": "SpikeInterface",
                    "description": "Auto-approved: no manual curation applied",
                },
                skip_duplicates=True,
            )
            mc_count += 1

        # OfficialCuration: points to the ManualCuration entry
        # PK = SortedSpikes PK (same as SortingTask PK)
        sorted_pk = {k: sorting_key[k] for k in spike_sorting.SortedSpikes.primary_key}
        if not (curation.OfficialCuration & sorted_pk):
            curation.OfficialCuration.insert1(
                {**sorted_pk, "curation_id": 0},
                skip_duplicates=True,
            )
            oc_count += 1

    print_ok(f"ManualCuration entries: {mc_count}")
    print_ok(f"OfficialCuration entries: {oc_count}")

    # ApplyOfficialCuration: populate() now handles auto-approval natively
    print_info("Running ApplyOfficialCuration.populate()...")
    curation.ApplyOfficialCuration.populate(
        display_progress=True, suppress_errors=SUPPRESS_ERRORS
    )

    aoc_count = len(curation.ApplyOfficialCuration & {"experiment_name": EXPERIMENT_NAME})
    print_ok(f"ApplyOfficialCuration entries: {aoc_count}")
    print_info("(Auto-sorted results approved as official — no changes applied)")

    return True


def step_synced_spikes(dry_run=False):
    """Step 17: Run SyncedSpikes.populate()."""
    print_header(17, "Run SyncedSpikes")

    if dry_run:
        print_info("Would call: SyncedSpikes.populate()")
        return True

    from aeon.dj_pipeline import spike_sorting

    print_info("Running SyncedSpikes.populate()...")
    spike_sorting.SyncedSpikes.populate(
        display_progress=True, suppress_errors=SUPPRESS_ERRORS
    )

    count = len(spike_sorting.SyncedSpikes & {"experiment_name": EXPERIMENT_NAME})
    print_ok(f"SyncedSpikes entries: {count}")

    units = len(spike_sorting.SyncedSpikes.Unit & {"experiment_name": EXPERIMENT_NAME})
    print_ok(f"SyncedSpikes.Unit entries: {units}")

    return True


def step_insert_matching_params(dry_run=False):
    """Step 18: Insert UnitMatchingParamSet (seed = first block)."""
    print_header(18, "Insert Unit Matching Parameters")

    if dry_run:
        print_info(f"Would insert UnitMatchingParamSet with seed_block_start={BLOCK_START}")
        return True

    from aeon.dj_pipeline import spike_sorting

    matching_paramset_id = 1
    if not (spike_sorting.UnitMatchingParamSet & {"matching_paramset_id": matching_paramset_id}):
        spike_sorting.UnitMatchingParamSet.insert1(
            {
                "matching_paramset_id": matching_paramset_id,
                "matching_method": "spike_time_overlap",
                "seed_block_start": BLOCK_START,
                "matching_paramset_description": "Test: seed at first block, delta_time=0.4ms",
                "params": {"delta_time": 0.4},
            },
        )
        print_ok(f"UnitMatchingParamSet inserted: seed_block_start={BLOCK_START}")
    else:
        print_ok(f"UnitMatchingParamSet {matching_paramset_id} already exists")

    return True


def step_unit_matching(dry_run=False):
    """Step 19: Run UnitMatching.populate() (across overlapping blocks)."""
    print_header(19, "Run Unit Matching")

    if dry_run:
        print_info("Would call: UnitMatching.populate()")
        print_info("Tests matching across 1-hour overlap between consecutive blocks")
        return True

    from aeon.dj_pipeline import spike_sorting

    print_info("Running UnitMatching.populate()...")
    spike_sorting.UnitMatching.populate(
        display_progress=True, suppress_errors=SUPPRESS_ERRORS
    )

    count = len(spike_sorting.UnitMatching & {"experiment_name": EXPERIMENT_NAME})
    print_ok(f"UnitMatching entries: {count}")

    global_units = len(spike_sorting.GlobalUnit & {"experiment_name": EXPERIMENT_NAME})
    print_ok(f"GlobalUnit entries: {global_units}")

    # Show matching details
    matched_units = len(spike_sorting.UnitMatching.Unit & {"experiment_name": EXPERIMENT_NAME})
    print_ok(f"UnitMatching.Unit entries: {matched_units}")

    spikes_entries = len(spike_sorting.UnitMatching.Spikes & {"experiment_name": EXPERIMENT_NAME})
    print_ok(f"UnitMatching.Spikes entries: {spikes_entries}")

    return True


# ===========================================================================
# Main
# ===========================================================================

STEPS = [
    # Phase 1: Setup & Ingestion
    (1, "verify_config", step_verify_config),
    (1, "create_experiment", step_create_experiment),
    (1, "insert_probe_config", step_insert_probe_config),
    (1, "ingest_epochs", step_ingest_epochs),
    (1, "manual_ephys_setup", step_manual_ephys_setup),
    (1, "ingest_chunks", step_ingest_chunks),
    (1, "create_blocks", step_create_blocks),
    (1, "populate_block_info", step_populate_block_info),
    # Phase 2: Spike Sorting
    (2, "create_electrode_groups", step_create_electrode_groups),
    (2, "insert_sorting_params", step_insert_sorting_params),
    (2, "create_sorting_tasks", step_create_sorting_tasks),
    (2, "preprocessing", step_preprocessing),
    (2, "spike_sorting_slurm", step_spike_sorting_slurm),
    (2, "post_processing", step_post_processing),
    (2, "sorted_spikes", step_sorted_spikes),
    # Phase 3: Curation, Sync & Unit Matching
    (3, "approve_auto_curation", step_approve_auto_curation),
    (3, "synced_spikes", step_synced_spikes),
    (3, "insert_matching_params", step_insert_matching_params),
    (3, "unit_matching", step_unit_matching),
]


def main():
    parser = argparse.ArgumentParser(
        description="Setup and run the ephys v2 pipeline test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phases:
  1  Setup & Ingestion     Steps 1-8   (no SLURM)
  2  Spike Sorting         Steps 9-15  (SLURM at step 13)
  3  Curation & Matching   Steps 16-19 (no SLURM)

Steps:
  1   verify_config           Check DB config, ceph path
  2   create_experiment       Register experiment + directories + subject
  3   insert_probe_config     ProbeType, Probe, ElectrodeConfig
  4   ingest_epochs           Epoch.ingest_epochs()
  5   manual_ephys_setup      ProbeInsertion + EphysEpoch (bypass read_probe_assignments)
  6   ingest_chunks           EphysChunk.ingest_chunks()
  7   create_blocks           4 × 3h blocks, 1h overlap
  8   populate_block_info     EphysBlockInfo.populate()
  9   create_electrode_groups 1 group × 96 channels (0-95)
  10  insert_sorting_params   Kilosort4 paramset
  11  create_sorting_tasks    SortingTask entries
  12  preprocessing           PreProcessing.populate()
  13  spike_sorting_slurm     *** SLURM submission ***
  14  post_processing         PostProcessing.populate()
  15  sorted_spikes           SortedSpikes.populate()
  16  approve_auto_curation   Auto → ManualCuration → OfficialCuration → ApplyOfficialCuration
  17  synced_spikes           SyncedSpikes.populate()
  18  insert_matching_params  UnitMatchingParamSet (seed = first block)
  19  unit_matching           UnitMatching.populate()
        """,
    )
    parser.add_argument("--step", type=int, help="Run a single step (1-19)")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], help="Run all steps in a phase")
    parser.add_argument("--dry-run", action="store_true", help="Preview without executing")
    args = parser.parse_args()

    if not args.dry_run:
        verify_prefix_or_exit()

    print("=" * 60)
    print("  Ephys v2 Pipeline Test")
    print(f"  Experiment: {EXPERIMENT_NAME}")
    print(f"  Subject:    {SUBJECT}")
    print(f"  DB prefix:  {dj.config.database.database_prefix or ''}")
    if args.dry_run:
        print("  Mode: DRY RUN")
    print("=" * 60)

    if args.step:
        # Run single step
        if args.step < 1 or args.step > len(STEPS):
            print(f"\n✗ Invalid step {args.step}. Valid range: 1-{len(STEPS)}")
            sys.exit(1)
        step_entry = STEPS[args.step - 1]
        _, step_name, step_func = step_entry
        success = step_func(dry_run=args.dry_run)
        if not success:
            print(f"\n✗ Step {args.step} ({step_name}) failed.")
            sys.exit(1)
        print(f"\n✓ Step {args.step} ({step_name}) completed.")

    elif args.phase:
        # Run all steps in a phase
        phase_steps = [(i, name, func) for i, (phase, name, func) in enumerate(STEPS, 1) if phase == args.phase]
        for step_num, step_name, step_func in phase_steps:
            try:
                success = step_func(dry_run=args.dry_run)
                if not success:
                    print(f"\n✗ Step {step_num} ({step_name}) failed. Stopping.")
                    sys.exit(1)
            except Exception as e:
                print_fail(f"Step {step_num} ({step_name}) raised: {e}")
                import traceback
                traceback.print_exc()
                print(f"\n✗ Failed at step {step_num}. Fix and re-run with --step {step_num}")
                sys.exit(1)

        print(f"\n✓ Phase {args.phase} completed!")

    else:
        # Default: run Phase 1 only (safest — Phase 2 needs SLURM)
        print_info("Running Phase 1 (Setup & Ingestion) by default.")
        print_info("Use --phase 2 or --phase 3 for later phases.\n")
        phase_steps = [(i, name, func) for i, (phase, name, func) in enumerate(STEPS, 1) if phase == 1]
        for step_num, step_name, step_func in phase_steps:
            try:
                success = step_func(dry_run=args.dry_run)
                if not success:
                    print(f"\n✗ Step {step_num} ({step_name}) failed. Stopping.")
                    sys.exit(1)
            except Exception as e:
                print_fail(f"Step {step_num} ({step_name}) raised: {e}")
                import traceback
                traceback.print_exc()
                print(f"\n✗ Failed at step {step_num}. Fix and re-run with --step {step_num}")
                sys.exit(1)

        print("\n" + "=" * 60)
        print("  ✓ Phase 1 complete!")
        print("  Next: --phase 2 for spike sorting (requires SLURM)")
        print("=" * 60)


if __name__ == "__main__":
    main()
