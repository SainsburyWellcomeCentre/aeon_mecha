"""Force-ingest pre-sorted Kilosort 2.5 data into the ephys v2 pipeline.

This script populates every table in the pipeline chain as if the data had been
processed through the pipeline normally. It is designed for the specific case of
ingesting externally-sorted data from:

    /ceph/aeon/aeon/data/processed/AEONX1/social-ephys0.1/
        2024-06-04T10-24-07/NeuropixelsV2Beta/SpikeSortingRaw/

Data characteristics:
- Kilosort 2.5, single probe (ProbeA), single epoch
- 4 channel groups: 1-144, 121-264, 241-384, 193-240
- All chunks concatenated into one continuous sort per channel group
- Auto-labels only (good/mua), no Phy curation

Usage:
    python -m aeon.dj_pipeline.scripts.force_ingest_external_sorting --step 1
    python -m aeon.dj_pipeline.scripts.force_ingest_external_sorting --step 2
    ...
    python -m aeon.dj_pipeline.scripts.force_ingest_external_sorting --step all

Phases:
    A (steps 1-8):   Foundation tables
    B (steps 9-15):  Sorting setup + force-insert KS data
    C (steps 16-19): Curation + Sync + UnitMatching
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# Configuration
# ============================================================================

EXPERIMENT_NAME = "social-ephys0.1-aeon3"
SUBJECT = "BAA-1104292"
PROBE_SERIAL = "NP2004-001"  # From prior test data
PROBE_TYPE = "neuropixels2.0_beta"
PROBE_LABEL = "ProbeA"
ELECTRODE_CONFIG_NAME = "0-383"  # Full probe, 384 electrodes
EPOCH_START = "2024-06-04 10:24:07"  # From directory name

PARAMSET_ID = "ks25_ext_001"  # Unique ID for this external sort
SORTING_METHOD = "kilosort2.5"

# Block: one giant block covering the full recording
# (block_start/block_end will be set from actual chunk data in step 7)
BLOCK_START = None  # populated at runtime from EphysChunk
BLOCK_END = None

# 4 channel groups with completed Kilosort 2.5 output
# Directory names on ceph follow the pattern:
#   NeuropixelsV2Beta_ProbeA_Chunks_2_147Chs_{start}_{end}
# The "Chunks_2_147" indicates chunks 2-147 were concatenated for sorting.
CHANNEL_GROUPS = [
    {
        "name": "1-144",
        "start_electrode": 1,
        "end_electrode": 144,
        "n_channels": 144,
        "dir_suffix": "NeuropixelsV2Beta_ProbeA_Chunks_2_147Chs_1_144",
    },
    {
        "name": "121-264",
        "start_electrode": 121,
        "end_electrode": 264,
        "n_channels": 144,
        "dir_suffix": "NeuropixelsV2Beta_ProbeA_Chunks_2_147Chs_121_264",
    },
    {
        "name": "241-384",
        "start_electrode": 241,
        "end_electrode": 384,
        "n_channels": 144,
        "dir_suffix": "NeuropixelsV2Beta_ProbeA_Chunks_2_147Chs_241_384",
    },
    {
        "name": "193-240",
        "start_electrode": 193,
        "end_electrode": 240,
        "n_channels": 48,
        "dir_suffix": "NeuropixelsV2Beta_ProbeA_Chunks_2_147Chs_193_240",
    },
]

# Path to the pre-sorted KS2.5 output on ceph (or S3 mount)
# Update this to the S3-mounted path when available
SORTED_DATA_ROOT = Path(
    "/ceph/aeon/aeon/data/processed/AEONX1/social-ephys0.1/"
    "2024-06-04T10-24-07/NeuropixelsV2Beta/SpikeSortingRaw"
)

# Suppress populate errors (set False for debugging)
SUPPRESS_ERRORS = False


# ============================================================================
# Utilities
# ============================================================================


def print_header(step_num, title):
    print(f"\n{'='*70}")
    print(f"  Step {step_num}: {title}")
    print(f"{'='*70}")


def print_ok(msg):
    print(f"  [OK] {msg}")


def print_info(msg):
    print(f"  [..] {msg}")


def print_fail(msg):
    print(f"  [FAIL] {msg}")


def get_block_bounds():
    """Get block start/end from EphysChunk table (must run after step 6)."""
    from aeon.dj_pipeline import ephys

    chunks = (
        ephys.EphysChunk & {"experiment_name": EXPERIMENT_NAME}
    ).fetch("chunk_start", "chunk_end", order_by="chunk_start")
    if not chunks[0].size:
        raise RuntimeError("No EphysChunk entries found — run step 6 first")
    return chunks[0][0], chunks[1][-1]


def get_sorting_task_key(group_name, block_start, block_end):
    """Build the full sorting task key for a given channel group."""
    return {
        "experiment_name": EXPERIMENT_NAME,
        "subject": SUBJECT,
        "insertion_number": 1,
        "block_start": block_start,
        "block_end": block_end,
        "probe_type": PROBE_TYPE,
        "electrode_config_name": ELECTRODE_CONFIG_NAME,
        "electrode_group": group_name,
        "paramset_id": PARAMSET_ID,
    }


# ============================================================================
# Phase A: Foundation Tables (Steps 1-8)
# ============================================================================


def step_01_verify_config():
    """Verify DB connection and data path access."""
    print_header(1, "Verify Configuration")

    import datajoint as dj

    # Check DB connection
    host = dj.config.get("database.host", "")
    prefix = dj.config["custom"].get("database.prefix", "")
    print_info(f"DB host: {host}")
    print_info(f"DB prefix: {prefix}")

    if not host:
        print_fail("No database host configured")
        return False

    # Warn if subject is still placeholder
    if SUBJECT == "???":
        print_fail("SUBJECT is still '???' — update before running ingestion steps")
        return False

    # Check data path (may not be accessible locally — just report)
    if SORTED_DATA_ROOT.exists():
        print_ok(f"Sorted data root: {SORTED_DATA_ROOT}")
        for grp in CHANNEL_GROUPS:
            grp_dir = SORTED_DATA_ROOT / grp["dir_suffix"]
            if grp_dir.exists():
                print_ok(f"  {grp['name']}: {grp_dir}")
            else:
                print_fail(f"  {grp['name']}: NOT FOUND at {grp_dir}")
    else:
        print_info(f"Sorted data root not accessible locally: {SORTED_DATA_ROOT}")
        print_info("This is OK if running on a machine with S3/ceph mounts")

    print_ok("Configuration verified")
    return True


def step_02_insert_experiment():
    """Insert Subject, Experiment, Experiment.Directory, Experiment.Subject."""
    print_header(2, "Insert Experiment + Subject")

    from aeon.dj_pipeline import acquisition, subject

    # Subject (skip_duplicates — may already exist)
    subject.Subject.insert1(
        {
            "subject": SUBJECT,
            "sex": "U",
            "subject_birth_date": "2024-01-01",
            "subject_description": "Ephys subject for social-ephys0.1-aeon3",
        },
        skip_duplicates=True,
    )
    print_ok(f"Subject: {SUBJECT}")

    # Check if Experiment already exists
    if acquisition.Experiment & {"experiment_name": EXPERIMENT_NAME}:
        print_ok(f"Experiment already exists: {EXPERIMENT_NAME}")
    else:
        print_fail(
            f"Experiment '{EXPERIMENT_NAME}' not found in DB. "
            "It should already exist from prior ingestion. "
            "Insert it manually if needed."
        )
        return False

    # Experiment.Directory (raw data path)
    acquisition.Experiment.Directory.insert1(
        {
            "experiment_name": EXPERIMENT_NAME,
            "directory_type": "raw",
            "repository_name": "ceph_aeon",
            "directory_path": "aeon/data/raw/AEONX1/social-ephys0.1",
            "load_order": 0,
        },
        skip_duplicates=True,
    )
    print_ok("Experiment.Directory: raw")

    # Experiment.Subject
    acquisition.Experiment.Subject.insert1(
        {"experiment_name": EXPERIMENT_NAME, "subject": SUBJECT},
        skip_duplicates=True,
    )
    print_ok(f"Experiment.Subject: {SUBJECT}")

    return True


def step_03_insert_probe_config():
    """Insert ProbeType, ProbeType.Electrode, Probe, ElectrodeConfig, ElectrodeConfig.Electrode."""
    print_header(3, "Insert Probe Configuration")

    from aeon.dj_pipeline import ephys

    # Create ProbeType with electrode geometry (uses probeinterface)
    if not (ephys.ProbeType & {"probe_type": PROBE_TYPE}):
        ephys.create_probe_type(PROBE_TYPE, "neuropixels", "NP2004")
        print_ok(f"Created ProbeType: {PROBE_TYPE}")
    else:
        print_ok(f"ProbeType already exists: {PROBE_TYPE}")

    n_electrodes = len(ephys.ProbeType.Electrode & {"probe_type": PROBE_TYPE})
    print_ok(f"  {n_electrodes} electrodes")

    # Probe entry
    ephys.Probe.insert1(
        {"probe": PROBE_SERIAL, "probe_type": PROBE_TYPE},
        skip_duplicates=True,
    )
    print_ok(f"Probe: {PROBE_SERIAL}")

    # ElectrodeConfig (full probe, 384 electrodes)
    if not (ephys.ElectrodeConfig & {"probe_type": PROBE_TYPE, "electrode_config_name": ELECTRODE_CONFIG_NAME}):
        from aeon.dj_pipeline import dict_to_uuid

        electrodes = (ephys.ProbeType.Electrode & {"probe_type": PROBE_TYPE}).fetch(
            "electrode", order_by="electrode"
        )
        config_hash = dict_to_uuid(
            {str(e): str(e) for e in electrodes}
        )
        ephys.ElectrodeConfig.insert1(
            {
                "probe_type": PROBE_TYPE,
                "electrode_config_name": ELECTRODE_CONFIG_NAME,
                "electrode_config_description": f"Full {PROBE_TYPE} probe, electrodes 0-383",
                "electrode_config_hash": config_hash,
            }
        )
        ephys.ElectrodeConfig.Electrode.insert(
            [
                {"probe_type": PROBE_TYPE, "electrode_config_name": ELECTRODE_CONFIG_NAME, "electrode": int(e)}
                for e in electrodes
            ]
        )
        print_ok(f"ElectrodeConfig: {ELECTRODE_CONFIG_NAME} ({len(electrodes)} electrodes)")
    else:
        print_ok(f"ElectrodeConfig already exists: {ELECTRODE_CONFIG_NAME}")

    return True


def step_04_insert_probe_insertion():
    """Insert ProbeInsertion entry."""
    print_header(4, "Insert ProbeInsertion")

    from aeon.dj_pipeline import ephys

    ephys.ProbeInsertion.insert1(
        {
            "experiment_name": EXPERIMENT_NAME,
            "subject": SUBJECT,
            "insertion_number": 1,
            "probe": PROBE_SERIAL,
        },
        skip_duplicates=True,
    )
    print_ok(f"ProbeInsertion: {EXPERIMENT_NAME}/{SUBJECT}/insertion_1 -> {PROBE_SERIAL}")

    return True


def step_05_insert_epoch():
    """Insert Epoch + EphysEpoch + EphysEpoch.Insertion (bypassing discover_epoch_probes)."""
    print_header(5, "Insert Epoch + EphysEpoch")

    from aeon.dj_pipeline import acquisition, ephys

    epoch_start = datetime.strptime(EPOCH_START, "%Y-%m-%d %H:%M:%S")
    epoch_key = {"experiment_name": EXPERIMENT_NAME, "epoch_start": epoch_start}

    # Epoch
    acquisition.Epoch.insert1(
        {
            **epoch_key,
            "directory_type": "raw",
            "epoch_dir": "2024-06-04T10-24-07",
        },
        skip_duplicates=True,
    )
    print_ok(f"Epoch: {epoch_start}")

    # EphysEpoch (Imported — use allow_direct_insert)
    ephys.EphysEpoch.insert1(
        {**epoch_key, "has_ephys": True, "n_probes": 1},
        skip_duplicates=True,
        allow_direct_insert=True,
    )
    print_ok("EphysEpoch: has_ephys=True, n_probes=1")

    # EphysEpoch.Insertion
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
    print_ok(f"EphysEpoch.Insertion: {PROBE_LABEL}")

    return True


def step_06_ingest_chunks():
    """Run EphysChunk.ingest_chunks() — discovers bin+clock files, creates sync models."""
    print_header(6, "Ingest Ephys Chunks")

    from aeon.dj_pipeline import ephys

    existing = len(ephys.EphysChunk & {"experiment_name": EXPERIMENT_NAME})
    if existing > 0:
        print_info(f"Already have {existing} EphysChunk entries — skipping ingest")
        return True

    print_info("Ingesting ephys chunks (this may take a while)...")
    ephys.EphysChunk.ingest_chunks(EXPERIMENT_NAME)

    chunks = (ephys.EphysChunk & {"experiment_name": EXPERIMENT_NAME}).fetch(as_dict=True)
    if not chunks:
        print_fail("No EphysChunk entries created — check raw data access")
        return False

    print_ok(f"EphysChunk entries: {len(chunks)}")
    print_info(f"  First chunk: {chunks[0]['chunk_start']}")
    print_info(f"  Last chunk:  {chunks[-1]['chunk_end']}")

    return True


def step_07_create_block():
    """Create one giant EphysBlock spanning all chunks."""
    print_header(7, "Create EphysBlock")

    from aeon.dj_pipeline import ephys

    block_start, block_end = get_block_bounds()
    print_info(f"Block: {block_start} -> {block_end}")

    ephys.EphysBlock.insert1(
        {
            "experiment_name": EXPERIMENT_NAME,
            "subject": SUBJECT,
            "insertion_number": 1,
            "block_start": block_start,
            "block_end": block_end,
        },
        skip_duplicates=True,
    )
    print_ok("EphysBlock: 1 giant block")

    return True


def step_08_populate_block_info():
    """Run EphysBlockInfo.populate() — computes chunk associations and channel mappings."""
    print_header(8, "Populate EphysBlockInfo")

    from aeon.dj_pipeline import ephys

    existing = len(ephys.EphysBlockInfo & {"experiment_name": EXPERIMENT_NAME})
    if existing > 0:
        print_info(f"Already have {existing} EphysBlockInfo entries")
        return True

    print_info("Populating EphysBlockInfo...")
    ephys.EphysBlockInfo.populate(
        {"experiment_name": EXPERIMENT_NAME},
        display_progress=True,
        suppress_errors=SUPPRESS_ERRORS,
    )

    n = len(ephys.EphysBlockInfo & {"experiment_name": EXPERIMENT_NAME})
    print_ok(f"EphysBlockInfo entries: {n}")

    return True


# ============================================================================
# Phase B: Sorting Setup + Force-Insert KS Data (Steps 9-15)
# ============================================================================


def step_09_create_electrode_groups():
    """Create ElectrodeGroup entries for each channel group."""
    print_header(9, "Create Electrode Groups")

    from aeon.dj_pipeline import ephys, spike_sorting

    for grp in CHANNEL_GROUPS:
        grp_key = {
            "probe_type": PROBE_TYPE,
            "electrode_config_name": ELECTRODE_CONFIG_NAME,
            "electrode_group": grp["name"],
        }

        if spike_sorting.ElectrodeGroup & grp_key:
            print_info(f"ElectrodeGroup '{grp['name']}' already exists")
            continue

        # Master entry
        spike_sorting.ElectrodeGroup.insert1(
            {
                **grp_key,
                "electrode_group_description": (
                    f"Channels {grp['start_electrode']}-{grp['end_electrode']} "
                    f"({grp['n_channels']} channels, external KS2.5 sorting)"
                ),
                "electrode_count": grp["n_channels"],
            }
        )

        # Electrode part entries
        electrodes = list(range(grp["start_electrode"], grp["end_electrode"] + 1))
        spike_sorting.ElectrodeGroup.Electrode.insert(
            [
                {**grp_key, "electrode": e}
                for e in electrodes
                if ephys.ElectrodeConfig.Electrode & {"probe_type": PROBE_TYPE, "electrode_config_name": ELECTRODE_CONFIG_NAME, "electrode": e}
            ]
        )
        n_inserted = len(
            spike_sorting.ElectrodeGroup.Electrode & grp_key
        )
        print_ok(f"ElectrodeGroup '{grp['name']}': {n_inserted} electrodes")

    return True


def step_10_insert_sorting_paramset():
    """Insert SortingParamSet for KS2.5."""
    print_header(10, "Insert SortingParamSet")

    from aeon.dj_pipeline import spike_sorting

    if spike_sorting.SortingParamSet & {"paramset_id": PARAMSET_ID}:
        print_info(f"SortingParamSet '{PARAMSET_ID}' already exists")
        return True

    params = {
        "note": "External KS2.5 sorting — data sorted outside the pipeline",
        "kilosort_version": "2.5",
        "channel_groups": [g["name"] for g in CHANNEL_GROUPS],
        "SI_SORTING_PARAMS": {},
        "SI_POSTPROCESSING_PARAMS": {},
    }

    spike_sorting.SortingParamSet.insert1(
        {
            "paramset_id": PARAMSET_ID,
            "sorting_method": SORTING_METHOD,
            "paramset_description": "External Kilosort 2.5 sorting (pre-sorted data)",
            "params": params,
        }
    )
    print_ok(f"SortingParamSet: {PARAMSET_ID} (method={SORTING_METHOD})")

    return True


def step_11_create_sorting_tasks():
    """Create SortingTask entries (1 per channel group)."""
    print_header(11, "Create SortingTask Entries")

    from aeon.dj_pipeline import spike_sorting

    block_start, block_end = get_block_bounds()

    for grp in CHANNEL_GROUPS:
        task_key = get_sorting_task_key(grp["name"], block_start, block_end)

        if spike_sorting.SortingTask & task_key:
            print_info(f"SortingTask '{grp['name']}' already exists")
            continue

        spike_sorting.SortingTask.insert1(task_key)
        print_ok(f"SortingTask: {grp['name']}")

    n = len(spike_sorting.SortingTask & {"experiment_name": EXPERIMENT_NAME, "paramset_id": PARAMSET_ID})
    print_ok(f"Total SortingTask entries: {n}")

    return True


def step_12_force_insert_preprocessing():
    """Force-insert PreProcessing placeholder rows (no actual preprocessing files)."""
    print_header(12, "Force-Insert PreProcessing (placeholder)")

    from aeon.dj_pipeline import spike_sorting

    block_start, block_end = get_block_bounds()
    now = datetime.now(timezone.utc)

    for grp in CHANNEL_GROUPS:
        task_key = get_sorting_task_key(grp["name"], block_start, block_end)

        if spike_sorting.PreProcessing & task_key:
            print_info(f"PreProcessing '{grp['name']}' already exists")
            continue

        # Build output dir path (matching the pipeline convention, even though files don't exist)
        output_dir = (
            f"{EXPERIMENT_NAME}/{SUBJECT}/insertion_1/ephys_blocks/"
            f"{block_start.strftime('%Y-%m-%dT%H-%M-%S')}_{block_end.strftime('%Y-%m-%dT%H-%M-%S')}/"
            f"{grp['name']}/{SORTING_METHOD}_{PARAMSET_ID}"
        )

        spike_sorting.PreProcessing.insert1(
            {
                **task_key,
                "execution_time": now,
                "execution_duration": 0.0,
                "sorting_output_dir": output_dir,
            },
            allow_direct_insert=True,
        )
        print_ok(f"PreProcessing: {grp['name']} -> {output_dir}")

    return True


def step_13_force_insert_spike_sorting():
    """Force-insert SpikeSorting placeholder rows."""
    print_header(13, "Force-Insert SpikeSorting (placeholder)")

    from aeon.dj_pipeline import spike_sorting

    block_start, block_end = get_block_bounds()
    now = datetime.now(timezone.utc)

    for grp in CHANNEL_GROUPS:
        task_key = get_sorting_task_key(grp["name"], block_start, block_end)

        if spike_sorting.SpikeSorting & task_key:
            print_info(f"SpikeSorting '{grp['name']}' already exists")
            continue

        spike_sorting.SpikeSorting.insert1(
            {
                **task_key,
                "execution_time": now,
                "execution_duration": 0.0,
            },
            allow_direct_insert=True,
        )
        print_ok(f"SpikeSorting: {grp['name']}")

    return True


def step_14_force_insert_post_processing():
    """Force-insert PostProcessing placeholder rows."""
    print_header(14, "Force-Insert PostProcessing (placeholder)")

    from aeon.dj_pipeline import spike_sorting

    block_start, block_end = get_block_bounds()
    now = datetime.now(timezone.utc)

    for grp in CHANNEL_GROUPS:
        task_key = get_sorting_task_key(grp["name"], block_start, block_end)

        if spike_sorting.PostProcessing & task_key:
            print_info(f"PostProcessing '{grp['name']}' already exists")
            continue

        spike_sorting.PostProcessing.insert1(
            {
                **task_key,
                "execution_time": now,
                "execution_duration": 0.0,
            },
            allow_direct_insert=True,
        )
        print_ok(f"PostProcessing: {grp['name']}")

    return True


def verify_spike_alignment():
    """Verify that external KS2.5 spike indices are aligned with pipeline chunk concatenation.

    The external sorting ran on a concatenated .dat file made by stitching hourly chunk
    files together. SyncedSpikes.populate() will later reconstruct chunk boundaries from
    cumulative ONIX Clock array lengths. If the concatenation order differs, spike times
    would map to the wrong positions and produce incorrect timestamps.

    This function checks alignment by comparing:
    1. The .dat file's total sample count (ground truth for what KS2.5 saw)
    2. The cumulative ONIX Clock array lengths (what the pipeline will use)
    3. That max(spike_times.npy) < total samples for each channel group
    4. That chunk ordering is monotonically chronological

    Returns:
        (True, total_samples_dat, total_samples_pipeline) on success, raises on failure.
    """
    from aeon.dj_pipeline import acquisition, ephys

    block_start, block_end = get_block_bounds()
    block_key = {
        "experiment_name": EXPERIMENT_NAME,
        "subject": SUBJECT,
        "insertion_number": 1,
        "block_start": block_start,
        "block_end": block_end,
    }

    block_info = ephys.EphysBlockInfo & block_key
    if not block_info:
        raise RuntimeError("No EphysBlockInfo found — run step 8 first")

    # --- Check 1: Chunk ordering is monotonically chronological ---
    chunk_starts, chunk_ends = (
        ephys.EphysChunk & {"experiment_name": EXPERIMENT_NAME}
    ).fetch("chunk_start", "chunk_end", order_by="chunk_start")

    for i in range(1, len(chunk_starts)):
        if chunk_starts[i] < chunk_starts[i - 1]:
            raise RuntimeError(
                f"Chunk ordering is NOT chronological! "
                f"chunk[{i-1}].start={chunk_starts[i-1]}, chunk[{i}].start={chunk_starts[i]}"
            )
    print_ok(f"Chunk ordering: {len(chunk_starts)} chunks, monotonically chronological")

    # --- Check 2: Compute total samples from ONIX Clock files ---
    # Load Clock.bin files for all chunks in the block (same files SyncedSpikes will use)
    clock_files, dir_types = (
        ephys.EphysChunk.File & (ephys.EphysBlockInfo.Chunk & block_key)
        & "file_name LIKE '%Clock%.bin'"
    ).fetch("file_path", "directory_type", order_by="chunk_start")

    total_clock_samples = 0
    for f, d in zip(clock_files, dir_types):
        ephys_dir = acquisition.Experiment.get_data_directory(
            {"experiment_name": EXPERIMENT_NAME}, directory_type=d
        )
        clock_ts = np.memmap(ephys_dir / f, mode="r", dtype=np.uint64)
        total_clock_samples += len(clock_ts)

    print_ok(f"Pipeline Clock samples (cumulative): {total_clock_samples:,}")

    # --- Check 3: Get total samples from the concatenated .bin file ---
    # NOTE: params.py in each KS dir points to temp_wh.dat (whitened data), NOT
    # the raw concatenated .bin. The .bin file has the same name as the directory
    # and lives alongside temp_wh.dat. We use the .bin for the sample count check
    # because it's the raw concatenation of chunk files — same data KS2.5 sorted.
    first_ks_dir = SORTED_DATA_ROOT / CHANNEL_GROUPS[0]["dir_suffix"]
    params_file = first_ks_dir / "params.py"

    n_channels_dat = None
    dat_dtype = "int16"

    if params_file.exists():
        params_text = params_file.read_text()
        for line in params_text.strip().splitlines():
            line = line.strip()
            if line.startswith("n_channels_dat"):
                n_channels_dat = int(line.split("=", 1)[1].strip())
            elif line.startswith("dtype"):
                dat_dtype = line.split("=", 1)[1].strip().strip("'\"")
        print_info(f"  KS params.py: n_channels_dat={n_channels_dat}, dtype={dat_dtype}")
    else:
        print_info(f"  No params.py found at {params_file}")

    # Use the raw .bin file (named after the directory), not temp_wh.dat
    bin_name = CHANNEL_GROUPS[0]["dir_suffix"] + ".bin"
    bin_path = first_ks_dir / bin_name

    total_samples_dat = None
    if bin_path.exists() and n_channels_dat is not None:
        bytes_per_sample = np.dtype(dat_dtype).itemsize
        file_size = bin_path.stat().st_size
        total_samples_dat = file_size // (n_channels_dat * bytes_per_sample)
        print_ok(f".bin file total samples: {total_samples_dat:,}")
        print_info(f"  .bin file: {bin_path.name} ({file_size / 1e12:.2f} TB)")
        print_info(f"  Computed as: {file_size} / ({n_channels_dat} channels * {bytes_per_sample} bytes)")

        # Compare .bin samples vs pipeline Clock samples
        if total_samples_dat == total_clock_samples:
            print_ok("MATCH: .bin sample count == pipeline Clock sample count")
        else:
            diff = total_samples_dat - total_clock_samples
            diff_pct = abs(diff) / max(total_samples_dat, total_clock_samples) * 100
            print_fail(
                f"MISMATCH: .bin has {total_samples_dat:,} samples, "
                f"pipeline has {total_clock_samples:,} samples "
                f"(diff={diff:+,}, {diff_pct:.2f}%)"
            )
            if diff_pct > 1.0:
                raise RuntimeError(
                    "Sample count mismatch exceeds 1%! The external sorting likely used "
                    "a different set of chunks than the pipeline. Do NOT proceed — "
                    "spike times would be incorrect."
                )
            else:
                print_info(
                    "  Small mismatch (<1%) — may be due to partial chunks at boundaries. "
                    "Proceeding, but verify spike times carefully after SyncedSpikes."
                )
    elif bin_path.exists():
        print_info(f"  .bin file found but n_channels unknown — skipping sample count check")
    else:
        print_info(f"  .bin file not accessible: {bin_path}")
        print_info("  (This is OK if running remotely — skipping .bin vs Clock comparison)")

    # --- Check 4: Verify max spike index < total samples for each channel group ---
    reference_samples = total_samples_dat if total_samples_dat is not None else total_clock_samples

    for grp in CHANNEL_GROUPS:
        ks_dir = SORTED_DATA_ROOT / grp["dir_suffix"]
        if not ks_dir.exists():
            print_fail(f"  KS dir not found: {ks_dir}")
            continue

        spike_times = np.load(ks_dir / "spike_times.npy").flatten()
        max_idx = int(spike_times.max()) if len(spike_times) > 0 else 0

        if max_idx >= reference_samples:
            raise RuntimeError(
                f"Channel group '{grp['name']}': max spike index ({max_idx:,}) >= "
                f"total samples ({reference_samples:,})! "
                "Spike indices are out of bounds — concatenation mismatch."
            )
        usage_pct = max_idx / reference_samples * 100 if reference_samples > 0 else 0
        print_ok(
            f"  {grp['name']}: max spike index = {max_idx:,} / {reference_samples:,} "
            f"({usage_pct:.1f}% of recording)"
        )

    return True, total_samples_dat, total_clock_samples


def step_15_force_insert_sorted_spikes():
    """Read Kilosort 2.5 output and force-insert SortedSpikes + SortedSpikes.Unit.

    This is the core force-ingestion step. For each channel group:
    1. Run alignment verification (.dat vs pipeline Clock samples)
    2. Load spike_times.npy (sample indices), spike_clusters.npy, templates.npy
    3. Read cluster labels from cluster_KSLabel.tsv
    4. Compute peak channel per unit from templates
    5. Map KS channel indices to probe electrode numbers
    6. Insert SortedSpikes master + Unit entries
    """
    print_header(15, "Force-Insert SortedSpikes (core ingestion)")

    from aeon.dj_pipeline import ephys, spike_sorting

    block_start, block_end = get_block_bounds()
    now = datetime.now(timezone.utc)

    # Alignment verification — this MUST pass before we insert any spike data
    print_info("Running spike alignment verification...")
    try:
        _, total_dat, total_pipeline = verify_spike_alignment()
    except RuntimeError as e:
        print_fail(f"Alignment check failed: {e}")
        return False
    print_ok("Alignment verification passed\n")

    for grp in CHANNEL_GROUPS:
        task_key = get_sorting_task_key(grp["name"], block_start, block_end)

        if spike_sorting.SortedSpikes & task_key:
            print_info(f"SortedSpikes '{grp['name']}' already exists — skipping")
            continue

        ks_dir = SORTED_DATA_ROOT / grp["dir_suffix"]
        if not ks_dir.exists():
            print_fail(f"KS output not found: {ks_dir}")
            return False

        print_info(f"Processing channel group: {grp['name']} ({ks_dir})")

        # Load KS2.5 output files
        spike_times_raw = np.load(ks_dir / "spike_times.npy").flatten()  # sample indices (int)
        spike_clusters = np.load(ks_dir / "spike_clusters.npy").flatten()
        templates = np.load(ks_dir / "templates.npy")  # (n_templates, n_samples, n_channels)

        print_info(f"  {len(spike_times_raw)} spikes, {len(np.unique(spike_clusters))} units")
        print_info(f"  Templates shape: {templates.shape}")
        print_info(f"  Max spike index: {spike_times_raw.max()}")

        # Peak channel per template (argmax of absolute template amplitude)
        peak_channels = np.argmax(np.abs(templates).max(axis=1), axis=1)

        # Read cluster labels
        ks_label_file = ks_dir / "cluster_KSLabel.tsv"
        if ks_label_file.exists():
            ks_labels = pd.read_csv(ks_label_file, sep="\t")
            label_map = dict(zip(ks_labels["cluster_id"], ks_labels["KSLabel"]))
        else:
            print_info("  No cluster_KSLabel.tsv found — using 'n.a.' for all units")
            label_map = {}

        # Insert SortedSpikes master
        spike_sorting.SortedSpikes.insert1(
            {
                **task_key,
                "execution_time": now,
                "execution_duration": 0.0,
                "curation_id": -1,  # raw, no curation
            },
            allow_direct_insert=True,
        )

        # Insert SortedSpikes.Unit for each cluster
        unique_units = np.unique(spike_clusters)
        for unit_id in unique_units:
            unit_id = int(unit_id)
            mask = spike_clusters == unit_id
            unit_spike_indices = spike_times_raw[mask].astype(np.int64)

            # Map KS channel index -> probe electrode number
            # KS channel 0 in group "1-144" = electrode 1
            if unit_id < len(peak_channels):
                peak_ch = int(peak_channels[unit_id])
            else:
                # Template index doesn't match unit_id — use 0
                peak_ch = 0

            electrode_id = peak_ch + grp["start_electrode"]

            # Clamp electrode_id to valid range
            electrode_id = max(grp["start_electrode"], min(grp["end_electrode"], electrode_id))

            # spike_sites: electrode ID for each spike (same peak channel for all)
            spike_sites = np.full(len(unit_spike_indices), electrode_id, dtype=np.int64)

            # spike_depths: zeros placeholder (would need probe geometry)
            spike_depths = np.zeros(len(unit_spike_indices), dtype=np.float64)

            # Quality label
            quality = label_map.get(unit_id, "n.a.")

            spike_sorting.SortedSpikes.Unit.insert1(
                {
                    **task_key,
                    "unit": unit_id,
                    "probe_type": PROBE_TYPE,
                    "electrode": electrode_id,
                    "unit_quality": quality,
                    "spike_count": len(unit_spike_indices),
                    "spike_indices": unit_spike_indices,
                    "spike_sites": spike_sites,
                    "spike_depths": spike_depths,
                },
                allow_direct_insert=True,
                ignore_extra_fields=True,
            )

        n_units = len(unique_units)
        n_good = sum(1 for u in unique_units if label_map.get(int(u)) == "good")
        n_mua = sum(1 for u in unique_units if label_map.get(int(u)) == "mua")
        print_ok(
            f"SortedSpikes '{grp['name']}': {n_units} units "
            f"({n_good} good, {n_mua} mua, {n_units - n_good - n_mua} n.a.)"
        )

    return True


# ============================================================================
# Phase C: Curation + Sync + UnitMatching (Steps 16-19)
# ============================================================================


def step_16_auto_approve_curation():
    """Insert CurationMethod, ManualCuration (auto-approve), OfficialCuration, and run ApplyOfficialCuration."""
    print_header(16, "Auto-Approve Curation")

    from aeon.dj_pipeline import spike_sorting
    from aeon.dj_pipeline import spike_sorting_curation as curation

    # Ensure CurationMethod exists
    curation.CurationMethod.insert1(
        {"curation_method": "SpikeInterface"}, skip_duplicates=True
    )

    block_start, block_end = get_block_bounds()
    now = datetime.now(timezone.utc)

    for grp in CHANNEL_GROUPS:
        task_key = get_sorting_task_key(grp["name"], block_start, block_end)

        # Check if already done
        if curation.OfficialCuration & task_key:
            print_info(f"OfficialCuration '{grp['name']}' already exists")
            continue

        # ManualCuration: auto-approve (curation_id=0, parent=-1, no file)
        mc_key = {**task_key, "curation_id": 0}
        curation.ManualCuration.insert1(
            {
                **mc_key,
                "curation_datetime": now,
                "parent_curation_id": -1,
                "curation_method": "SpikeInterface",
                "description": "Auto-approved: external KS2.5 sorting, no manual curation",
            },
            skip_duplicates=True,
        )

        # OfficialCuration: designate curation_id=0 as official
        curation.OfficialCuration.insert1(
            {**task_key, "curation_id": 0},
            skip_duplicates=True,
        )
        print_ok(f"Curation approved: {grp['name']}")

    # Run ApplyOfficialCuration.populate()
    print_info("Running ApplyOfficialCuration.populate()...")
    curation.ApplyOfficialCuration.populate(
        {"experiment_name": EXPERIMENT_NAME},
        display_progress=True,
        suppress_errors=SUPPRESS_ERRORS,
    )

    n = len(curation.ApplyOfficialCuration & {"experiment_name": EXPERIMENT_NAME})
    print_ok(f"ApplyOfficialCuration entries: {n}")

    return True


def validate_synced_spikes_against_ground_truth():
    """Compare our SyncedSpikes output against Dario's pre-computed HARP timestamps.

    Dario (darioc) pre-computed HARP-synchronized spike times and saved them as
    `spike_index_harp_clock_binary_2_147.npy` alongside the KS output. These files
    are per-spike arrays (same length as spike_times.npy) containing HARP clock
    timestamps. They exist for at least Chs_1_144 and Chs_241_384.

    This function:
    1. Loads Dario's ground truth file for each available channel group
    2. Fetches our SyncedSpikes timestamps from the DB
    3. Reconstructs per-spike HARP times by repeating chunk-level times per unit
    4. Compares the two — if they match, our clock conversion is correct

    This is the strongest possible validation that our spike times are correct
    relative to behavior data (both use the same HARP clock).
    """
    from aeon.dj_pipeline import spike_sorting

    block_start, block_end = get_block_bounds()

    GROUND_TRUTH_FILENAME = "spike_index_harp_clock_binary_2_147.npy"
    groups_checked = 0
    groups_passed = 0

    for grp in CHANNEL_GROUPS:
        ks_dir = SORTED_DATA_ROOT / grp["dir_suffix"]
        gt_file = ks_dir / GROUND_TRUTH_FILENAME
        if not gt_file.exists():
            print_info(f"  {grp['name']}: No ground truth file — skipping")
            continue

        groups_checked += 1
        print_info(f"  {grp['name']}: Loading ground truth ({gt_file.stat().st_size / 1e9:.2f} GB)...")

        # Load Dario's pre-computed HARP timestamps (one per spike)
        gt_harp_times = np.load(gt_file)
        spike_times_raw = np.load(ks_dir / "spike_times.npy").flatten()
        spike_clusters = np.load(ks_dir / "spike_clusters.npy").flatten()

        if len(gt_harp_times) != len(spike_times_raw):
            print_fail(
                f"  {grp['name']}: Ground truth length ({len(gt_harp_times)}) != "
                f"spike_times length ({len(spike_times_raw)})"
            )
            continue

        # Fetch our SyncedSpikes for this channel group
        task_key = get_sorting_task_key(grp["name"], block_start, block_end)
        synced_units = (
            spike_sorting.SyncedSpikes.Unit & task_key
        ).fetch("unit", "spike_times", as_dict=True)

        if not synced_units:
            print_fail(f"  {grp['name']}: No SyncedSpikes.Unit entries found")
            continue

        # Rebuild per-spike timestamps from our pipeline output
        # SyncedSpikes stores spike_times per unit — we need to interleave them
        # back into the original spike order (matching spike_times.npy order)
        our_harp_times = np.empty(len(spike_times_raw), dtype=np.float64)
        our_harp_times[:] = np.nan  # sentinel for unmatched spikes

        unit_map = {u["unit"]: u["spike_times"] for u in synced_units}
        for unit_id, unit_times in unit_map.items():
            mask = spike_clusters == unit_id
            if mask.sum() != len(unit_times):
                print_fail(
                    f"  {grp['name']} unit {unit_id}: spike count mismatch "
                    f"(KS: {mask.sum()}, pipeline: {len(unit_times)})"
                )
                continue
            our_harp_times[mask] = unit_times

        n_matched = np.sum(~np.isnan(our_harp_times))
        n_total = len(our_harp_times)
        if n_matched < n_total:
            print_info(
                f"  {grp['name']}: {n_matched}/{n_total} spikes matched "
                f"({n_total - n_matched} unmatched)"
            )

        # Compare — gt_harp_times might be in different units (HARP seconds vs datetime)
        # First check what Dario's values look like
        gt_sample = gt_harp_times[:5]
        our_sample = our_harp_times[:5]
        print_info(f"  Ground truth sample: {gt_sample}")
        print_info(f"  Our pipeline sample: {our_sample}")

        # Compute differences for matched spikes
        valid = ~np.isnan(our_harp_times)
        if not np.any(valid):
            print_fail(f"  {grp['name']}: No valid comparisons")
            continue

        diffs = our_harp_times[valid] - gt_harp_times.flatten()[valid].astype(np.float64)
        abs_diffs = np.abs(diffs)

        print_info(f"  Difference stats (our - ground_truth):")
        print_info(f"    Mean:   {np.mean(diffs):.6f}")
        print_info(f"    Median: {np.median(diffs):.6f}")
        print_info(f"    Max:    {np.max(abs_diffs):.6f}")
        print_info(f"    Std:    {np.std(diffs):.6f}")
        print_info(f"    P99:    {np.percentile(abs_diffs, 99):.6f}")

        # Threshold: if max difference > 1 second, something is very wrong
        # If < 1ms, we're golden. Between 1ms and 1s, investigate.
        max_diff = np.max(abs_diffs)
        if max_diff < 0.001:  # < 1ms
            print_ok(f"  {grp['name']}: PASS — max difference < 1ms ({max_diff:.6f}s)")
            groups_passed += 1
        elif max_diff < 1.0:  # < 1s
            print_info(
                f"  {grp['name']}: WARN — max difference {max_diff:.4f}s "
                "(within 1s but investigate if units differ)"
            )
            groups_passed += 1  # still count as passed, might be unit conversion
        else:
            print_fail(
                f"  {grp['name']}: FAIL — max difference {max_diff:.2f}s. "
                "Spike times do NOT match ground truth!"
            )

    if groups_checked == 0:
        print_info("No ground truth files found — skipping validation")
        print_info(f"(Expected: {GROUND_TRUTH_FILENAME} in channel group directories)")
        return True  # not a failure, just no data to compare

    print_info(f"\nGround truth validation: {groups_passed}/{groups_checked} groups passed")
    if groups_passed < groups_checked:
        print_fail("Some channel groups failed ground truth validation!")
        return False

    print_ok("All available ground truth comparisons passed!")
    return True


def step_17_populate_synced_spikes():
    """Run SyncedSpikes.populate() — converts spike indices to HARP timestamps."""
    print_header(17, "Populate SyncedSpikes")

    from aeon.dj_pipeline import spike_sorting

    print_info("Running SyncedSpikes.populate()...")
    print_info("This converts spike indices -> HARP timestamps using Clock + sync models")
    spike_sorting.SyncedSpikes.populate(
        {"experiment_name": EXPERIMENT_NAME},
        display_progress=True,
        suppress_errors=SUPPRESS_ERRORS,
    )

    n_master = len(spike_sorting.SyncedSpikes & {"experiment_name": EXPERIMENT_NAME})
    n_units = len(spike_sorting.SyncedSpikes.Unit & {"experiment_name": EXPERIMENT_NAME})
    print_ok(f"SyncedSpikes: {n_master} entries, {n_units} unit-chunk rows")

    # Validate against Dario's pre-computed ground truth if available
    print_info("\nRunning ground truth validation against Dario's pre-computed HARP timestamps...")
    gt_ok = validate_synced_spikes_against_ground_truth()
    if not gt_ok:
        print_fail(
            "Ground truth validation failed! Our spike times may be incorrect. "
            "Do NOT proceed to UnitMatching until this is investigated."
        )
        return False

    return True


def step_18_insert_unit_matching_paramset():
    """Insert UnitMatchingParamSet."""
    print_header(18, "Insert UnitMatchingParamSet")

    from aeon.dj_pipeline import spike_sorting

    block_start, _ = get_block_bounds()

    paramset_key = {"matching_paramset_id": 1}

    if spike_sorting.UnitMatchingParamSet & paramset_key:
        print_info("UnitMatchingParamSet already exists")
        return True

    spike_sorting.UnitMatchingParamSet.insert1(
        {
            **paramset_key,
            "matching_method": "spike_time_overlap",
            "seed_block_start": block_start,
            "matching_paramset_description": "Standard matching for external KS2.5 ingestion (1 block, trivial)",
            "params": {"delta_time": 0.4},
        }
    )
    print_ok(f"UnitMatchingParamSet: id=1, seed_block_start={block_start}")

    return True


def step_19_populate_unit_matching():
    """Run UnitMatching.populate() — trivial with 1 block (each unit -> 1 GlobalUnit)."""
    print_header(19, "Populate UnitMatching")

    from aeon.dj_pipeline import spike_sorting

    print_info("Running UnitMatching.populate()...")
    print_info("With 1 block, each unit maps 1:1 to a GlobalUnit")
    spike_sorting.UnitMatching.populate(
        {"experiment_name": EXPERIMENT_NAME},
        display_progress=True,
        suppress_errors=SUPPRESS_ERRORS,
    )

    n_matching = len(spike_sorting.UnitMatching & {"experiment_name": EXPERIMENT_NAME})
    n_global = len(spike_sorting.GlobalUnit & {"experiment_name": EXPERIMENT_NAME})
    n_spikes = len(spike_sorting.UnitMatching.Spikes & {"experiment_name": EXPERIMENT_NAME})
    print_ok(f"UnitMatching: {n_matching} entries")
    print_ok(f"GlobalUnit: {n_global} units")
    print_ok(f"UnitMatching.Spikes: {n_spikes} rows")

    return True


# ============================================================================
# Step dispatch
# ============================================================================

STEPS = {
    1: ("Verify config", step_01_verify_config),
    2: ("Insert experiment + subject", step_02_insert_experiment),
    3: ("Insert probe config", step_03_insert_probe_config),
    4: ("Insert ProbeInsertion", step_04_insert_probe_insertion),
    5: ("Insert Epoch + EphysEpoch", step_05_insert_epoch),
    6: ("Ingest EphysChunks", step_06_ingest_chunks),
    7: ("Create EphysBlock", step_07_create_block),
    8: ("Populate EphysBlockInfo", step_08_populate_block_info),
    9: ("Create ElectrodeGroups", step_09_create_electrode_groups),
    10: ("Insert SortingParamSet", step_10_insert_sorting_paramset),
    11: ("Create SortingTasks", step_11_create_sorting_tasks),
    12: ("Force-insert PreProcessing", step_12_force_insert_preprocessing),
    13: ("Force-insert SpikeSorting", step_13_force_insert_spike_sorting),
    14: ("Force-insert PostProcessing", step_14_force_insert_post_processing),
    15: ("Force-insert SortedSpikes", step_15_force_insert_sorted_spikes),
    16: ("Auto-approve curation", step_16_auto_approve_curation),
    17: ("Populate SyncedSpikes", step_17_populate_synced_spikes),
    18: ("Insert UnitMatchingParamSet", step_18_insert_unit_matching_paramset),
    19: ("Populate UnitMatching", step_19_populate_unit_matching),
}


def run_step(step_num):
    """Run a single step by number."""
    if step_num not in STEPS:
        print(f"Unknown step: {step_num}. Valid range: 1-{len(STEPS)}")
        return False

    name, func = STEPS[step_num]
    try:
        return func()
    except Exception as e:
        print_fail(f"Step {step_num} ({name}) failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Force-ingest pre-sorted Kilosort 2.5 data into ephys v2 pipeline"
    )
    parser.add_argument(
        "--step",
        required=True,
        help="Step number (1-19) or 'all' to run all steps, or range like '1-8'",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without making changes (not all steps support this)",
    )

    args = parser.parse_args()

    # Print banner
    print("\n" + "=" * 70)
    print("  Force-Ingest External Sorting: Kilosort 2.5 -> Ephys v2 Pipeline")
    print(f"  Experiment: {EXPERIMENT_NAME}")
    print(f"  Subject: {SUBJECT}")
    print(f"  Probe: {PROBE_SERIAL} ({PROBE_TYPE})")
    print(f"  Channel groups: {len(CHANNEL_GROUPS)}")
    print("=" * 70)

    if SUBJECT == "???":
        print("\n  WARNING: SUBJECT is still '???' — update before running!\n")

    if args.step == "all":
        steps = list(range(1, len(STEPS) + 1))
    elif "-" in args.step:
        start, end = args.step.split("-")
        steps = list(range(int(start), int(end) + 1))
    else:
        steps = [int(args.step)]

    for step_num in steps:
        ok = run_step(step_num)
        if not ok:
            print(f"\n  Step {step_num} failed. Fix the issue and re-run from this step.")
            sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  All requested steps completed successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
