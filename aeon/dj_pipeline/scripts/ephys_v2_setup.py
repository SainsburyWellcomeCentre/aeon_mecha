"""Setup and run script for the ephys v2 PK revamp test.

Tests the full ephys pipeline end-to-end using AEONX1/social-ephys0.1 data,
with the new v2 PK structure (experiment_name, insertion_number) — no subject.

Three phases:
  Phase 1: Setup & ingestion (no SLURM needed)
  Phase 2: Spike sorting (requires SLURM for SpikeSorting step)
  Phase 3: Post-sorting & curation (no SLURM needed)

Prerequisites:
  - dj_local_conf.json configured with prefix "elissas_aeon_ephys_test_"
  - On HPC with access to /ceph/aeon/
  - SpikeInterface installed from Elissa's fork (for Phase 2)

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
EXPECTED_PREFIX = "elissas_aeon_ephys_test_"

# Probe config (from previous test run — ephys_test_ingestion.py)
PROBE_NAME = "NP2004-001"
PROBE_TYPE = "neuropixels - NP2004"
ELECTRODE_CONFIG_NAME = "0-383"
N_ELECTRODES = 384

# Block schedule: 4 blocks × 3 hours, 1-hour overlap
BLOCK_START = "2024-06-04 11:00:00"
BLOCK_DURATION_HOURS = 3
BLOCK_ADVANCE_HOURS = 2  # 3 - 1 overlap = advance by 2
N_BLOCKS = 4

# Sorting config
PARAMSET_ID = 400
SORTING_METHOD = "kilosort4"
N_ELECTRODE_GROUPS = 4  # 96 channels each

# Controls whether populate() raises on first error
SUPPRESS_ERRORS = False


# ---------------------------------------------------------------------------
# Safety: verify DB prefix before ANY pipeline imports
# ---------------------------------------------------------------------------
def verify_prefix_or_exit():
    """Check the database prefix BEFORE importing any pipeline modules.

    Critical because `from aeon.dj_pipeline import ephys` triggers
    `dj.schema(get_schema_name("ephys"))` at import time, which CREATES
    schemas in the database.
    """
    import datajoint as dj

    if "custom" not in dj.config:
        dj.config["custom"] = {}

    prefix = dj.config["custom"].get("database.prefix", "")
    host = dj.config.get("database.host", "")

    if prefix != EXPECTED_PREFIX:
        print(f"\n  ✗ SAFETY CHECK FAILED: database prefix is '{prefix}'")
        print(f"    Expected: '{EXPECTED_PREFIX}'")
        print(f"    Host: '{host}'")
        if not prefix:
            print(f"    The prefix is empty — dj_local_conf.json may not have been found.")
            print(f"    Make sure you run from the aeon_mecha_ephys/ directory.")
        else:
            print(f"    Fix: ensure dj_local_conf.json has:")
            print(f'      "custom": {{"database.prefix": "{EXPECTED_PREFIX}"}}')
        sys.exit(1)

    if "aeon-db2" in host:
        print(f"\n  ✗ SAFETY CHECK FAILED: connecting to production host '{host}'")
        print(f"    This script should only run against aeon-db (test).")
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

    prefix = dj.config["custom"].get("database.prefix", "")
    print_ok(f"Database prefix: {prefix}")

    try:
        conn = dj.conn()
        print_ok(f"Connected to {dj.config['database.host']}:{dj.config['database.port']}")
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
    """Step 2: Register experiment + directories."""
    print_header(2, "Create Experiment")

    if dry_run:
        print_info(f"Would create experiment: {EXPERIMENT_NAME}")
        print_info(f"Would register raw dir: aeon/data/raw/AEONX1/social-ephys0.1")
        return True

    from aeon.dj_pipeline import acquisition

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

    # Verify
    exp = (acquisition.Experiment & {"experiment_name": EXPERIMENT_NAME}).fetch1()
    print_ok(f"  arena: {exp['arena_name']}, location: {exp['location']}")

    dirs = (acquisition.Experiment.Directory & {"experiment_name": EXPERIMENT_NAME}).fetch(as_dict=True)
    for d in dirs:
        print_ok(f"  directory: {d['directory_type']} → {d['directory_path']}")

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

    # ProbeType
    if not (ephys.ProbeType & {"probe_type": PROBE_TYPE}):
        ephys.create_probe_type(
            PROBE_TYPE, manufacturer="neuropixels", probe_name="NP2004"
        )
        print_ok(f"ProbeType created: {PROBE_TYPE}")
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
    """Step 4: Ingest acquisition epochs from filesystem."""
    print_header(4, "Ingest Acquisition Epochs")

    if dry_run:
        print_info(f"Would call: Epoch.ingest_epochs('{EXPERIMENT_NAME}')")
        return True

    from aeon.dj_pipeline import acquisition

    print_info("Ingesting epochs...")
    acquisition.Epoch.ingest_epochs(EXPERIMENT_NAME)

    epochs = (acquisition.Epoch & {"experiment_name": EXPERIMENT_NAME}).fetch(as_dict=True)
    if not epochs:
        print_fail("No epochs ingested")
        return False

    print_ok(f"Ingested {len(epochs)} epochs")
    # Show first/last
    epoch_starts = sorted([e["epoch_start"] for e in epochs])
    print_info(f"  First: {epoch_starts[0]}")
    print_info(f"  Last:  {epoch_starts[-1]}")

    return True


def step_ephys_epoch_populate(dry_run=False):
    """Step 5: Run EphysEpoch.populate() — discovers probes, creates ProbeInsertion (Probe must pre-exist from step 3)."""
    print_header(5, "Populate EphysEpoch (probe discovery)")

    if dry_run:
        print_info("Would call: EphysEpoch.populate()")
        print_info("Expected: auto-creates ProbeInsertion entries (Probe must pre-exist)")
        return True

    from aeon.dj_pipeline import ephys

    print_info("Running EphysEpoch.populate()...")
    ephys.EphysEpoch.populate(
        display_progress=True, suppress_errors=SUPPRESS_ERRORS
    )

    # Check results
    ephys_epochs = (ephys.EphysEpoch & {"experiment_name": EXPERIMENT_NAME}).fetch(as_dict=True)
    has_ephys = [e for e in ephys_epochs if e["has_ephys"]]
    no_ephys = [e for e in ephys_epochs if not e["has_ephys"]]
    print_ok(f"EphysEpoch: {len(ephys_epochs)} total, {len(has_ephys)} with ephys, {len(no_ephys)} without")

    # Check ProbeInsertion was auto-created
    probe_insertions = (ephys.ProbeInsertion & {"experiment_name": EXPERIMENT_NAME}).fetch(as_dict=True)
    if not probe_insertions:
        print_fail("No ProbeInsertion entries created — EphysEpoch.make() may have failed")
        return False

    print_ok(f"ProbeInsertion entries: {len(probe_insertions)}")
    for pi in probe_insertions:
        print_info(
            f"  insertion {pi['insertion_number']}: probe={pi['probe']}, "
            f"label={pi['probe_label']}"
        )

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

    chunks = (ephys.EphysChunk & {"experiment_name": EXPERIMENT_NAME}).fetch(as_dict=True)
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

    probe_insertions = (ephys.ProbeInsertion & {"experiment_name": EXPERIMENT_NAME}).fetch(as_dict=True)
    if not probe_insertions:
        print_fail("No ProbeInsertions found — run step 5 first")
        return False

    total_blocks = 0
    for pi in probe_insertions:
        for start, end in blocks_to_create:
            block_key = {
                "experiment_name": EXPERIMENT_NAME,
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

    block_infos = (ephys.EphysBlockInfo & {"experiment_name": EXPERIMENT_NAME}).fetch(as_dict=True)
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
    """Step 9: Create electrode groups (4 groups of 96 channels)."""
    print_header(9, "Create Electrode Groups")

    if dry_run:
        print_info(f"Would create {N_ELECTRODE_GROUPS} groups of {N_ELECTRODES // N_ELECTRODE_GROUPS} channels")
        return True

    from aeon.dj_pipeline import spike_sorting, ephys

    electrode_config_key = {
        "probe_type": PROBE_TYPE,
        "electrode_config_name": ELECTRODE_CONFIG_NAME,
    }
    config_electrodes = (ephys.ElectrodeConfig.Electrode & electrode_config_key).fetch(
        "electrode", order_by="electrode"
    )

    group_size = len(config_electrodes) // N_ELECTRODE_GROUPS
    for i in range(N_ELECTRODE_GROUPS):
        group_electrodes = list(config_electrodes[i * group_size : (i + 1) * group_size])
        group_name = f"{group_electrodes[0]}-{group_electrodes[-1]}"

        spike_sorting.ElectrodeGroup.insert1(
            {
                **electrode_config_key,
                "electrode_group": group_name,
                "electrode_group_description": f"electrodes {group_name}",
                "electrode_count": len(group_electrodes),
            },
            skip_duplicates=True,
        )
        spike_sorting.ElectrodeGroup.Electrode.insert(
            (
                {**electrode_config_key, "electrode_group": group_name, "electrode": e}
                for e in group_electrodes
            ),
            skip_duplicates=True,
        )
        print_ok(f"  Group {group_name} ({len(group_electrodes)} electrodes)")

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

    blocks = (ephys.EphysBlock & {"experiment_name": EXPERIMENT_NAME}).fetch(as_dict=True)
    electrode_groups = spike_sorting.ElectrodeGroup.fetch("electrode_group")

    if not blocks:
        print_fail("No EphysBlock entries — run step 7 first")
        return False

    count = 0
    for block in blocks:
        for eg in electrode_groups:
            task_key = {
                "experiment_name": block["experiment_name"],
                "insertion_number": block["insertion_number"],
                "block_start": block["block_start"],
                "block_end": block["block_end"],
                "probe_type": PROBE_TYPE,
                "electrode_config_name": ELECTRODE_CONFIG_NAME,
                "electrode_group": eg,
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
    print_info("           SLURM copies the environment at job start time.")

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
# PHASE 3: Curation & Downstream
# ===========================================================================

def step_fake_curation(dry_run=False):
    """Step 16: Fake curation — make auto results the official curation."""
    print_header(16, "Fake Curation (auto → official)")

    if dry_run:
        print_info("Would insert CurationMethod + OfficialCuration entries")
        print_info("Skipping ManualCuration / Phy GUI entirely")
        return True

    from aeon.dj_pipeline import spike_sorting_curation as curation

    # Ensure CurationMethod has the default entry
    if not (curation.CurationMethod & {"curation_method": "SpikeInterface"}):
        curation.CurationMethod.insert1(
            {"curation_method": "SpikeInterface"},
            skip_duplicates=True,
        )
        print_ok("CurationMethod 'SpikeInterface' inserted")

    # For each SortedSpikes entry, create an OfficialCuration pointing to raw sorting
    from aeon.dj_pipeline import spike_sorting

    sorted_entries = (spike_sorting.SortedSpikes & {"experiment_name": EXPERIMENT_NAME}).fetch(
        as_dict=True
    )

    count = 0
    for entry in sorted_entries:
        curation_key = {
            **{k: entry[k] for k in spike_sorting.SortedSpikes.primary_key},
            "curation_method": "SpikeInterface",
        }
        if not (curation.OfficialCuration & curation_key):
            curation.OfficialCuration.insert1(curation_key, skip_duplicates=True)
            count += 1

    print_ok(f"OfficialCuration entries created: {count}")
    print_info("(ManualCuration skipped — using auto sorting as official)")

    return True


def step_apply_curation(dry_run=False):
    """Step 17: Run ApplyOfficialCuration.populate()."""
    print_header(17, "Apply Official Curation")

    if dry_run:
        print_info("Would call: ApplyOfficialCuration.populate()")
        return True

    from aeon.dj_pipeline import spike_sorting_curation as curation

    print_info("Running ApplyOfficialCuration.populate()...")
    curation.ApplyOfficialCuration.populate(
        display_progress=True, suppress_errors=SUPPRESS_ERRORS
    )

    count = len(curation.ApplyOfficialCuration & {"experiment_name": EXPERIMENT_NAME})
    print_ok(f"ApplyOfficialCuration entries: {count}")

    return True


def step_synced_spikes(dry_run=False):
    """Step 18: Run SyncedSpikes.populate()."""
    print_header(18, "Run SyncedSpikes")

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


def step_unit_matching(dry_run=False):
    """Step 19: Run UnitMatching (across overlapping blocks)."""
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

    universal_units = len(spike_sorting.UniversalUnit & {"experiment_name": EXPERIMENT_NAME})
    print_ok(f"UniversalUnit entries: {universal_units}")

    return True


def step_chunked_spike_times(dry_run=False):
    """Step 20: Run ChunkedSpikeTimes.populate()."""
    print_header(20, "Run ChunkedSpikeTimes")

    if dry_run:
        print_info("Would call: ChunkedSpikeTimes.populate()")
        return True

    from aeon.dj_pipeline import spike_sorting

    print_info("Running ChunkedSpikeTimes.populate()...")
    spike_sorting.ChunkedSpikeTimes.populate(
        display_progress=True, suppress_errors=SUPPRESS_ERRORS
    )

    count = len(spike_sorting.ChunkedSpikeTimes & {"experiment_name": EXPERIMENT_NAME})
    print_ok(f"ChunkedSpikeTimes entries: {count}")

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
    (1, "ephys_epoch_populate", step_ephys_epoch_populate),
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
    # Phase 3: Curation & Downstream
    (3, "fake_curation", step_fake_curation),
    (3, "apply_curation", step_apply_curation),
    (3, "synced_spikes", step_synced_spikes),
    (3, "unit_matching", step_unit_matching),
    (3, "chunked_spike_times", step_chunked_spike_times),
]


def main():
    parser = argparse.ArgumentParser(
        description="Setup and run the ephys v2 pipeline test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phases:
  1  Setup & Ingestion     Steps 1-8   (no SLURM)
  2  Spike Sorting         Steps 9-15  (SLURM at step 13)
  3  Curation & Downstream Steps 16-20 (no SLURM)

Steps:
  1   verify_config           Check DB config, ceph path
  2   create_experiment       Register experiment + directories
  3   insert_probe_config     ProbeType, Probe, ElectrodeConfig
  4   ingest_epochs           Epoch.ingest_epochs()
  5   ephys_epoch_populate    EphysEpoch → ProbeInsertion (Probe must pre-exist)
  6   ingest_chunks           EphysChunk.ingest_chunks()
  7   create_blocks           4 × 3h blocks, 1h overlap
  8   populate_block_info     EphysBlockInfo.populate()
  9   create_electrode_groups 4 groups × 96 channels
  10  insert_sorting_params   Kilosort4 paramset
  11  create_sorting_tasks    SortingTask entries
  12  preprocessing           PreProcessing.populate()
  13  spike_sorting_slurm     *** SLURM submission ***
  14  post_processing         PostProcessing.populate()
  15  sorted_spikes           SortedSpikes.populate()
  16  fake_curation           Auto → OfficialCuration
  17  apply_curation          ApplyOfficialCuration.populate()
  18  synced_spikes           SyncedSpikes.populate()
  19  unit_matching           UnitMatching.populate()
  20  chunked_spike_times     ChunkedSpikeTimes.populate()
        """,
    )
    parser.add_argument("--step", type=int, help="Run a single step (1-20)")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], help="Run all steps in a phase")
    parser.add_argument("--dry-run", action="store_true", help="Preview without executing")
    args = parser.parse_args()

    if not args.dry_run:
        verify_prefix_or_exit()

    print("=" * 60)
    print("  Ephys v2 Pipeline Test")
    print(f"  Experiment: {EXPERIMENT_NAME}")
    print(f"  DB prefix:  {EXPECTED_PREFIX}")
    if args.dry_run:
        print("  Mode: DRY RUN")
    print("=" * 60)

    if args.step:
        # Run single step
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
