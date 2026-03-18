"""Production deployment script for the ephys v2 pipeline.

Deploys the ephys v2 pipeline (v0.2.0) to production:
  - aeon-db2 database, aeon_ prefix
  - AEONX1/social-ephys0.1 data on ceph
  - 30-hour blocks with 6-hour overlap

Five phases:
  Phase 0: Reconnaissance (inspect existing data, discover subjects/probes/timerange)
  Phase 1: Drop old tables (reverse dependency order, with safety checks)
  Phase 2: Ingestion (experiment, probes, epochs, chunks, blocks)
  Phase 3: Spike sorting setup (electrode groups, params, tasks, preprocessing)
  Phase 4: Post-sorting (curation, sync, unit matching)

Prerequisites:
  - dj_local_conf.json configured for aeon-db2 with prefix "aeon_"
  - On HPC with access to /ceph/aeon/
  - Workspace cloned from v0.2.0 tag

Usage:
  uv run python prod_ephys_deploy.py --phase 0          # Reconnaissance
  uv run python prod_ephys_deploy.py --phase 1          # Drop old tables
  uv run python prod_ephys_deploy.py --step 8           # Run single step
  uv run python prod_ephys_deploy.py --dry-run --step 8 # Preview step
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — EDIT THESE before running
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "social-ephys0.1-aeon3"

# Subject — confirmed from behavioral data (AEON2/social-ephys0.1/2024-06-04T10-29-49)
# Brain area targets: retrosplenial cortex + superior colliculus
SUBJECT = "BAA-1104292"

# Probe config (same hardware as test — NeuropixelsV2Beta)
PROBE_NAME = "NP2004-001"
PROBE_TYPE = "neuropixels - NP2004"
ELECTRODE_CONFIG_NAME = "0-383"
N_ELECTRODES = 384

# Epoch + probe labels — Phase 0 confirmed single probe for June 2024 epochs
PROBE_LABELS = ["ProbeA"]

# Block schedule: 30h blocks, 6h overlap (24h advance)
# First epoch starts 2024-06-04 10:24:07 — round up slightly
BLOCK_START = "2024-06-04 11:00:00"
BLOCK_DURATION_HOURS = 30
BLOCK_ADVANCE_HOURS = 24  # 30 - 6 overlap = advance by 24
N_BLOCKS = 7  # ~1 week of data

# Sorting config
PARAMSET_ID = 1  # Production uses paramset_id=1 (test used 400)
SORTING_METHOD = "kilosort4"
ELECTRODE_GROUP_NAME = "0-95"
ELECTRODE_GROUP_SIZE = 96

# Safety
PRODUCTION_PREFIX = "aeon_"
PRODUCTION_HOST = "aeon-db2"

# Controls whether populate() raises on first error
SUPPRESS_ERRORS = False


# ---------------------------------------------------------------------------
# Safety: verify PRODUCTION DB before ANY pipeline imports
# ---------------------------------------------------------------------------
def verify_production_or_exit():
    """Check that we're pointing at the production database.

    Critical because `from aeon.dj_pipeline import ephys` triggers
    schema creation at import time.
    """
    import datajoint as dj

    if "custom" not in dj.config:
        dj.config["custom"] = {}

    prefix = dj.config["custom"].get("database.prefix", "")
    host = dj.config.get("database.host", "")

    if not prefix:
        print(f"\n  X SAFETY CHECK FAILED: database prefix is empty.")
        print(f"    dj_local_conf.json may not have been found.")
        print(f"    Make sure you run from the repo root directory.")
        sys.exit(1)

    if prefix != PRODUCTION_PREFIX:
        print(f"\n  X SAFETY CHECK FAILED: prefix is '{prefix}', expected '{PRODUCTION_PREFIX}'.")
        print(f"    This script is for PRODUCTION only.")
        sys.exit(1)

    if PRODUCTION_HOST not in host:
        print(f"\n  X SAFETY CHECK FAILED: host is '{host}', expected '{PRODUCTION_HOST}'.")
        print(f"    This script is for PRODUCTION only.")
        sys.exit(1)

    print(f"  + Production safety check passed: {host} / {prefix}")


def verify_config_ready():
    """Check that placeholder values have been filled in."""
    errors = []
    if "PLACEHOLDER" in SUBJECT:
        errors.append("SUBJECT is still a placeholder — run Phase 0 first")
    if "PLACEHOLDER" in BLOCK_START:
        errors.append("BLOCK_START is still a placeholder — run Phase 0 first")
    if errors:
        print("\n  X Configuration incomplete:")
        for e in errors:
            print(f"    - {e}")
        print("\n  Run Phase 0 (--phase 0) to discover the right values,")
        print("  then edit the constants at the top of this script.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def print_header(step_num, title):
    print(f"\n{'='*60}")
    print(f"  Step {step_num}: {title}")
    print(f"{'='*60}\n")


def print_ok(msg):
    print(f"  + {msg}")


def print_fail(msg):
    print(f"  X {msg}")


def print_info(msg):
    print(f"  > {msg}")


def print_warn(msg):
    print(f"  ! {msg}")


# ===========================================================================
# PHASE 0: Reconnaissance
# ===========================================================================

def step_recon_connection(dry_run=False):
    """Step 1: Verify production DB connection and ceph access."""
    print_header(1, "Verify Production Connection")

    if dry_run:
        print_info("Would check: DB connection to aeon-db2, ceph path")
        return True

    import datajoint as dj

    prefix = dj.config["custom"].get("database.prefix", "")
    host = dj.config.get("database.host", "")
    print_ok(f"Database: {host} / prefix: {prefix}")

    try:
        conn = dj.conn()
        print_ok(f"Connected to {host}:{dj.config['database.port']}")
    except Exception as e:
        print_fail(f"Cannot connect: {e}")
        return False

    data_dir = Path("/ceph/aeon/aeon/data/raw/AEONX1/social-ephys0.1")
    if not data_dir.exists():
        print_fail(f"Data directory not found: {data_dir}")
        return False
    print_ok(f"Ceph data directory exists: {data_dir}")

    store_dir = Path("/ceph/aeon/aeon/dj_store")
    if not store_dir.exists():
        print_fail(f"Store directory not found: {store_dir}")
        return False
    print_ok(f"Store directory exists: {store_dir}")

    return True


def step_recon_existing_tables(dry_run=False):
    """Step 2: List existing production ephys tables and row counts."""
    print_header(2, "Existing Production Tables")

    if dry_run:
        print_info("Would show: SHOW TABLES for aeon_ephys, aeon_spike_sorting, aeon_spike_sorting_curation")
        return True

    import datajoint as dj

    conn = dj.conn()
    schemas_to_check = ["aeon_ephys", "aeon_spike_sorting", "aeon_spike_sorting_curation"]

    for schema_name in schemas_to_check:
        print(f"\n  --- {schema_name} ---")
        try:
            tables = conn.query(f"SHOW TABLES IN `{schema_name}`").fetchall()
            if not tables:
                print_info("  (no tables)")
                continue
            for (table_name,) in tables:
                try:
                    count = conn.query(
                        f"SELECT COUNT(*) FROM `{schema_name}`.`{table_name}`"
                    ).fetchone()[0]
                    print_info(f"  {table_name}: {count} rows")
                except Exception:
                    print_info(f"  {table_name}: (error reading)")
        except Exception:
            print_info(f"  Schema does not exist yet")

    return True


def step_recon_epochs(dry_run=False):
    """Step 3: Discover available epochs and their time range."""
    print_header(3, "Discover Epochs on Ceph")

    if dry_run:
        print_info("Would scan: /ceph/aeon/aeon/data/raw/AEONX1/social-ephys0.1/")
        return True

    data_dir = Path("/ceph/aeon/aeon/data/raw/AEONX1/social-ephys0.1")
    epochs = sorted(
        p for p in data_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )

    if not epochs:
        print_fail("No epoch directories found!")
        return False

    print_ok(f"Total epochs: {len(epochs)}")
    print_ok(f"First epoch: {epochs[0].name}")
    print_ok(f"Last epoch:  {epochs[-1].name}")

    # Parse timestamps
    from datetime import datetime as dt
    first = dt.strptime(epochs[0].name, "%Y-%m-%dT%H-%M-%S")
    last = dt.strptime(epochs[-1].name, "%Y-%m-%dT%H-%M-%S")
    span = last - first
    print_ok(f"Time span: {span.days} days ({first} to {last})")

    # Show first 10 and last 5 for context
    print(f"\n  First 10 epochs:")
    for e in epochs[:10]:
        print_info(f"  {e.name}")
    if len(epochs) > 15:
        print_info(f"  ... ({len(epochs) - 15} more) ...")
    print(f"\n  Last 5 epochs:")
    for e in epochs[-5:]:
        print_info(f"  {e.name}")

    print(f"\n  RECOMMENDATION for BLOCK_START:")
    print_info(f"  Set BLOCK_START to a time slightly after the first epoch start.")
    print_info(f"  First epoch: {first}")
    # Suggest rounding up to the next hour
    suggested = first.replace(minute=0, second=0) + timedelta(hours=1)
    print_info(f"  Suggested:   \"{suggested.strftime('%Y-%m-%d %H:%M:%S')}\"")
    print_info(f"  With {N_BLOCKS} blocks x {BLOCK_ADVANCE_HOURS}h advance = {N_BLOCKS * BLOCK_ADVANCE_HOURS / 24:.1f} days coverage")

    return True


def step_recon_probes(dry_run=False):
    """Step 4: Inspect probe structure in sample epochs."""
    print_header(4, "Discover Probes in Epoch Data")

    if dry_run:
        print_info("Would inspect NeuropixelsV2Beta directories in first few epochs")
        return True

    data_dir = Path("/ceph/aeon/aeon/data/raw/AEONX1/social-ephys0.1")
    epochs = sorted(
        p for p in data_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )[:5]  # Check first 5 epochs

    probe_labels_seen = set()
    device_types_seen = set()

    for epoch_dir in epochs:
        print(f"\n  Epoch: {epoch_dir.name}")
        for device_dir in sorted(epoch_dir.iterdir()):
            if not device_dir.is_dir():
                continue
            if "Neuropixels" in device_dir.name:
                device_types_seen.add(device_dir.name)
                # List files to discover probe labels
                bin_files = sorted(device_dir.glob("*_AmplifierData_*.bin"))
                labels = set()
                for bf in bin_files:
                    parts = bf.name.split("_")
                    if len(parts) >= 3:
                        labels.add(parts[1])  # e.g. "ProbeA"
                probe_labels_seen.update(labels)
                print_info(f"  {device_dir.name}/: {len(bin_files)} bin files, probes: {sorted(labels)}")

                # Check metadata
                meta_files = list(device_dir.glob("Metadata*"))
                if meta_files:
                    print_info(f"    Metadata: {[m.name for m in meta_files]}")

    print(f"\n  SUMMARY:")
    print_ok(f"  Device types: {sorted(device_types_seen)}")
    print_ok(f"  Probe labels: {sorted(probe_labels_seen)}")
    if len(probe_labels_seen) > 1:
        print_info(f"  DUAL PROBE setup detected — set PROBE_LABELS = {sorted(probe_labels_seen)}")
    elif len(probe_labels_seen) == 1:
        print_info(f"  SINGLE PROBE setup — set PROBE_LABELS = {sorted(probe_labels_seen)}")
    else:
        print_warn(f"  No probe labels found — check data structure manually")

    return True


def step_recon_subjects(dry_run=False):
    """Step 5: Check subject info in acquisition tables."""
    print_header(5, "Check Subject Information")

    if dry_run:
        print_info("Would query Experiment.Subject and Subject tables")
        return True

    import datajoint as dj

    conn = dj.conn()
    prefix = dj.config["custom"].get("database.prefix", "")

    # Check Experiment.Subject
    try:
        rows = conn.query(
            f"SELECT * FROM `{prefix}subject`.`subject`"
        ).fetchall()
        if rows:
            print_ok(f"Subjects in subject table: {len(rows)}")
            for row in rows[:20]:
                print_info(f"  {row}")
        else:
            print_info("No subjects found in subject table")
    except Exception as e:
        print_warn(f"Could not query subject table: {e}")

    # Check Experiment.Subject
    try:
        rows = conn.query(
            f"SELECT * FROM `{prefix}acquisition`.`experiment_subject`"
        ).fetchall()
        if rows:
            print_ok(f"Experiment-subject pairs: {len(rows)}")
            for row in rows[:20]:
                print_info(f"  {row}")
        else:
            print_info("No experiment-subject entries found")
    except Exception as e:
        print_warn(f"Could not query experiment_subject: {e}")

    print(f"\n  NOTE: If no subject is found for this experiment, you'll need to")
    print(f"  ask the SWC team for the correct subject name, or check the")
    print(f"  experiment metadata / lab notebooks.")
    print(f"\n  After determining the subject name, update SUBJECT at the top of this script.")

    return True


def step_recon_existing_data(dry_run=False):
    """Step 6: Show existing DB values for tables we'll interact with."""
    print_header(6, "Existing DB Data (tables we interact with)")

    if dry_run:
        print_info("Would query: Experiment, Experiment.Directory, Experiment.Subject,")
        print_info("  Subject, ProbeType, Probe, ElectrodeConfig, EphysBlock (old)")
        return True

    import datajoint as dj

    conn = dj.conn()
    prefix = dj.config["custom"].get("database.prefix", "")
    exp_name = EXPERIMENT_NAME

    queries = [
        ("Experiment", f"SELECT * FROM `{prefix}acquisition`.`#experiment` WHERE experiment_name='{exp_name}'"),
        ("Experiment.Directory", f"SELECT * FROM `{prefix}acquisition`.`#experiment__directory` WHERE experiment_name='{exp_name}'"),
        ("Experiment.Subject", f"SELECT * FROM `{prefix}acquisition`.`#experiment__subject` WHERE experiment_name='{exp_name}'"),
        ("Subject (all)", f"SELECT * FROM `{prefix}subject`.`#subject`"),
        ("ProbeType", f"SELECT * FROM `{prefix}ephys`.`#probe_type`"),
        ("Probe", f"SELECT * FROM `{prefix}ephys`.`#probe`"),
        ("ElectrodeConfig", f"SELECT * FROM `{prefix}ephys`.electrode_config"),
        ("ElectrodeConfig.Electrode (count)", f"SELECT electrode_config_name, COUNT(*) as n_electrodes FROM `{prefix}ephys`.electrode_config__electrode GROUP BY electrode_config_name"),
        ("EphysBlock (old, if exists)", f"SELECT * FROM `{prefix}ephys`.`_ephys_block`"),
        ("TargetArea", f"SELECT * FROM `{prefix}ephys`.`#target_area`"),
    ]

    for label, sql in queries:
        print(f"\n  --- {label} ---")
        try:
            rows = conn.query(sql).fetchall()
            if rows:
                # Get column names
                cols = conn.query(sql.replace("SELECT *", "SELECT *", 1))
                for row in rows:
                    print_info(f"  {row}")
            else:
                print_info("  (empty)")
        except Exception as e:
            err = str(e)
            if "doesn't exist" in err:
                print_info("  (table does not exist)")
            else:
                print_warn(f"  Error: {err}")

    print(f"\n  COMPARE these values with the script's config constants above.")
    print(f"  Tables with skip_duplicates=True won't overwrite existing data.")

    return True


# ===========================================================================
# PHASE 1: Drop Old Production Tables
# ===========================================================================

def step_safety_check(dry_run=False):
    """Step 6: Final safety check before dropping tables."""
    print_header(7, "Safety Check Before Drop")

    if dry_run:
        print_info("Would verify: host=aeon-db2, prefix=aeon_")
        return True

    import datajoint as dj

    host = dj.config.get("database.host", "")
    prefix = dj.config["custom"].get("database.prefix", "")

    print_ok(f"Host: {host}")
    print_ok(f"Prefix: {prefix}")

    assert PRODUCTION_HOST in host, f"Not pointing at production host! Got: {host}"
    assert prefix == PRODUCTION_PREFIX, f"Not using production prefix! Got: {prefix}"

    print_ok("Safety check passed — confirmed production DB")
    print_warn("The next steps will DROP tables from PRODUCTION.")
    print_warn("Make sure you have reviewed Phase 0 reconnaissance output.")

    return True


def step_drop_curation_schema(dry_run=False):
    """Step 7: Drop spike_sorting_curation schema (depends on spike_sorting)."""
    print_header(8, "Drop spike_sorting_curation Schema")

    if dry_run:
        print_info("Would drop entire aeon_spike_sorting_curation schema")
        print_info("DataJoint will prompt for manual confirmation (safemode=True)")
        return True

    from aeon.dj_pipeline import spike_sorting_curation as ssc

    print_warn("Dropping spike_sorting_curation schema...")
    print_warn("DataJoint will ask for confirmation — type 'yes' to proceed")
    print_info("Tables to be dropped:")

    try:
        import datajoint as dj
        conn = dj.conn()
        tables = conn.query(
            f"SHOW TABLES IN `{PRODUCTION_PREFIX}spike_sorting_curation`"
        ).fetchall()
        for (t,) in tables:
            print_info(f"  {t}")
    except Exception:
        print_info("  (schema may not exist)")
        return True

    ssc.schema.drop()  # Will prompt for confirmation (safemode=True)
    print_ok("spike_sorting_curation schema dropped")
    return True


def step_drop_spike_sorting_schema(dry_run=False):
    """Step 8: Drop spike_sorting schema (depends on ephys)."""
    print_header(9, "Drop spike_sorting Schema")

    if dry_run:
        print_info("Would drop entire aeon_spike_sorting schema")
        print_info("DataJoint will prompt for manual confirmation")
        return True

    from aeon.dj_pipeline import spike_sorting as ss

    print_warn("Dropping spike_sorting schema...")
    print_warn("DataJoint will ask for confirmation — type 'yes' to proceed")
    print_info("Tables to be dropped:")

    try:
        import datajoint as dj
        conn = dj.conn()
        tables = conn.query(
            f"SHOW TABLES IN `{PRODUCTION_PREFIX}spike_sorting`"
        ).fetchall()
        for (t,) in tables:
            print_info(f"  {t}")
    except Exception:
        print_info("  (schema may not exist)")
        return True

    ss.schema.drop()  # Will prompt for confirmation
    print_ok("spike_sorting schema dropped")
    return True


def step_drop_ephys_tables(dry_run=False):
    """Step 9: Drop changed ephys tables (keep Lookup tables)."""
    print_header(10, "Drop Changed Ephys Tables")

    tables_to_drop = [
        "EphysBlockInfo",  # depends on EphysBlock
        "EphysBlock",      # depends on EphysChunk/ProbeInsertion
        "EphysChunk",      # depends on EphysEpoch/ProbeInsertion
        "EphysEpoch",      # depends on acquisition.Epoch
        "ProbeInsertion",  # PK changed
    ]
    tables_to_keep = [
        "ProbeType", "ProbeType.Electrode",
        "Probe",
        "ElectrodeConfig", "ElectrodeConfig.Electrode",
        "TargetArea",
    ]

    if dry_run:
        print_info("Tables to DROP (PK or definition changed):")
        for t in tables_to_drop:
            print_info(f"  - {t}")
        print_info("Tables to KEEP (Lookup, unchanged):")
        for t in tables_to_keep:
            print_info(f"  - {t}")
        return True

    from aeon.dj_pipeline import ephys

    print_warn("Dropping ephys tables with changed definitions...")
    print_warn("Each table will prompt for confirmation")
    print_info(f"Keeping: {', '.join(tables_to_keep)}")

    for table_name in tables_to_drop:
        table = getattr(ephys, table_name, None)
        if table is None:
            print_warn(f"  Table {table_name} not found in module — skipping")
            continue
        try:
            n = len(table())
            print_info(f"  Dropping {table_name} ({n} rows)...")
            table.drop()  # Will prompt for confirmation
            print_ok(f"  {table_name} dropped")
        except Exception as e:
            print_warn(f"  {table_name}: {e}")

    return True


def step_verify_clean_state(dry_run=False):
    """Step 10: Verify schemas are clean and recreate tables."""
    print_header(11, "Verify Clean State + Recreate Tables")

    if dry_run:
        print_info("Would verify clean state and re-import modules to create tables")
        return True

    import datajoint as dj
    import importlib

    conn = dj.conn()

    # Check what remains
    schemas_to_check = ["aeon_ephys", "aeon_spike_sorting", "aeon_spike_sorting_curation"]
    for schema_name in schemas_to_check:
        try:
            tables = conn.query(f"SHOW TABLES IN `{schema_name}`").fetchall()
            table_names = [t[0] for t in tables]
            if table_names:
                print_info(f"  {schema_name}: {table_names}")
            else:
                print_info(f"  {schema_name}: empty")
        except Exception:
            print_info(f"  {schema_name}: does not exist")

    # Re-import to trigger schema creation
    print_info("Re-importing pipeline modules to recreate tables...")
    from aeon.dj_pipeline import ephys, spike_sorting, spike_sorting_curation

    # Force re-creation by reimporting
    importlib.reload(ephys)
    importlib.reload(spike_sorting)
    importlib.reload(spike_sorting_curation)

    # Verify tables exist
    for schema_name in schemas_to_check:
        tables = conn.query(f"SHOW TABLES IN `{schema_name}`").fetchall()
        table_names = [t[0] for t in tables]
        print_ok(f"  {schema_name}: {len(table_names)} tables")
        for t in table_names:
            print_info(f"    {t}")

    return True


# ===========================================================================
# PHASE 2: Ingestion
# ===========================================================================

def step_verify_existing_data(dry_run=False):
    """Step 12: Verify tables we did NOT drop have the expected data.

    These tables should already exist in production. We verify they have
    the values we expect and ERROR if anything is missing or wrong.
    We never insert into these tables.
    """
    print_header(12, "Verify Existing Data (tables we did NOT drop)")

    if dry_run:
        print_info("Would verify: Experiment, Subject, Directory, ProbeType, Probe, ElectrodeConfig")
        return True

    from aeon.dj_pipeline import acquisition, subject, ephys

    errors = []

    # --- Subject ---
    if subject.Subject & {"subject": SUBJECT}:
        existing = (subject.Subject & {"subject": SUBJECT}).fetch1()
        print_ok(f"Subject exists: {SUBJECT}")
        print_info(f"  {existing}")
    else:
        errors.append(f"Subject '{SUBJECT}' not found in subject table")

    # --- Experiment ---
    if acquisition.Experiment & {"experiment_name": EXPERIMENT_NAME}:
        existing = (acquisition.Experiment & {"experiment_name": EXPERIMENT_NAME}).fetch1()
        print_ok(f"Experiment exists: {EXPERIMENT_NAME}")
        print_info(f"  start_time: {existing.get('experiment_start_time')}")
        print_info(f"  arena: {existing.get('arena_name')}, location: {existing.get('location')}")
    else:
        errors.append(f"Experiment '{EXPERIMENT_NAME}' not found")

    # --- Directory ---
    dir_key = {"experiment_name": EXPERIMENT_NAME, "directory_type": "raw"}
    if acquisition.Experiment.Directory & dir_key:
        existing = (acquisition.Experiment.Directory & dir_key).fetch1()
        print_ok(f"Directory exists: {existing.get('directory_path')}")
    else:
        errors.append(f"No raw directory registered for {EXPERIMENT_NAME}")

    # --- Experiment.Subject ---
    exp_subj_key = {"experiment_name": EXPERIMENT_NAME, "subject": SUBJECT}
    if acquisition.Experiment.Subject & exp_subj_key:
        print_ok(f"Experiment.Subject exists: {SUBJECT}")
    else:
        errors.append(f"Experiment.Subject link missing: {EXPERIMENT_NAME} <-> {SUBJECT}")

    # --- ProbeType ---
    if ephys.ProbeType & {"probe_type": PROBE_TYPE}:
        n_elec = len(ephys.ProbeType.Electrode & {"probe_type": PROBE_TYPE})
        print_ok(f"ProbeType exists: {PROBE_TYPE} ({n_elec} electrodes)")
    else:
        errors.append(f"ProbeType '{PROBE_TYPE}' not found")

    # --- Probe ---
    if ephys.Probe & {"probe": PROBE_NAME}:
        print_ok(f"Probe exists: {PROBE_NAME}")
    else:
        errors.append(f"Probe '{PROBE_NAME}' not found")

    # --- ElectrodeConfig ---
    ec_key = {"probe_type": PROBE_TYPE, "electrode_config_name": ELECTRODE_CONFIG_NAME}
    if ephys.ElectrodeConfig & ec_key:
        n_elec = len(ephys.ElectrodeConfig.Electrode & ec_key)
        print_ok(f"ElectrodeConfig exists: {ELECTRODE_CONFIG_NAME} ({n_elec} electrodes)")
    else:
        errors.append(f"ElectrodeConfig '{ELECTRODE_CONFIG_NAME}' not found")

    if errors:
        print_fail("Missing expected data — cannot proceed:")
        for e in errors:
            print_fail(f"  {e}")
        print_info("These tables were NOT dropped and should already exist in production.")
        print_info("If they are missing, something is wrong — investigate before continuing.")
        return False

    print_ok("All expected existing data verified")
    return True

    return True


def step_manual_ephys_setup(dry_run=False):
    """Step 14: Manual ProbeInsertion + EphysEpoch setup.

    Since read_probe_assignments() raises NotImplementedError, we manually:
    1. Insert ProbeInsertion(s) — one per probe label
    2. Insert Epoch + EphysEpoch for relevant epochs
    3. Insert EphysEpoch.Insertion linking probes to epochs
    """
    print_header(13, "Manual Ephys Setup (ProbeInsertion + Epoch + EphysEpoch)")

    if dry_run:
        for i, label in enumerate(PROBE_LABELS, 1):
            print_info(f"Would insert ProbeInsertion: subject={SUBJECT}, insertion={i}, probe={PROBE_NAME}, label={label}")
        print_info(f"Would insert Epoch + EphysEpoch for all epoch directories")
        return True

    from aeon.dj_pipeline import acquisition, ephys

    # 1. Insert ProbeInsertions (one per probe label)
    for i, label in enumerate(PROBE_LABELS, 1):
        ephys.ProbeInsertion.insert1(
            {
                "experiment_name": EXPERIMENT_NAME,
                "subject": SUBJECT,
                "insertion_number": i,
                "probe": PROBE_NAME,
            },
            skip_duplicates=True,
        )
        print_ok(f"ProbeInsertion: subject={SUBJECT}, insertion={i} ({label})")

    # 2. Discover all epoch directories
    data_dir = Path("/ceph/aeon/aeon/data/raw/AEONX1/social-ephys0.1")
    epoch_dirs = sorted(
        p for p in data_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    print_info(f"Found {len(epoch_dirs)} epoch directories")

    # 3. Insert Epoch + EphysEpoch + EphysEpoch.Insertion for each
    n_inserted = 0
    n_skipped = 0
    for epoch_dir in epoch_dirs:
        epoch_start = datetime.strptime(epoch_dir.name, "%Y-%m-%dT%H-%M-%S")

        # Check which probes are present in this epoch
        nv2b_dir = epoch_dir / "NeuropixelsV2Beta"
        nv2_dir = epoch_dir / "NeuropixelsV2"
        device_dir = nv2b_dir if nv2b_dir.exists() else (nv2_dir if nv2_dir.exists() else None)

        if device_dir is None:
            n_skipped += 1
            continue

        # Discover probe labels in this epoch
        bin_files = list(device_dir.glob("*_AmplifierData_*.bin"))
        epoch_probe_labels = set()
        for bf in bin_files:
            parts = bf.name.split("_")
            if len(parts) >= 3:
                epoch_probe_labels.add(parts[1])

        if not epoch_probe_labels:
            n_skipped += 1
            continue

        # Insert Epoch
        acquisition.Epoch.insert1(
            {
                "experiment_name": EXPERIMENT_NAME,
                "epoch_start": epoch_start,
                "directory_type": "raw",
                "epoch_dir": epoch_dir.name,
            },
            skip_duplicates=True,
        )

        # Insert EphysEpoch master
        epoch_key = {"experiment_name": EXPERIMENT_NAME, "epoch_start": epoch_start}
        active_probes = epoch_probe_labels.intersection(PROBE_LABELS)
        ephys.EphysEpoch.insert1(
            {**epoch_key, "has_ephys": True, "n_probes": len(active_probes)},
            skip_duplicates=True,
            allow_direct_insert=True,
        )

        # Insert EphysEpoch.Insertion for each probe present
        for i, label in enumerate(PROBE_LABELS, 1):
            if label in epoch_probe_labels:
                ephys.EphysEpoch.Insertion.insert1(
                    {
                        **epoch_key,
                        "subject": SUBJECT,
                        "insertion_number": i,
                        "probe_label": label,
                    },
                    skip_duplicates=True,
                    allow_direct_insert=True,
                )

        n_inserted += 1

    print_ok(f"Epochs processed: {n_inserted} inserted, {n_skipped} skipped (no ephys data)")

    # Verify
    pi = len(ephys.ProbeInsertion & {"experiment_name": EXPERIMENT_NAME})
    ee = len(ephys.EphysEpoch & {"experiment_name": EXPERIMENT_NAME})
    ei = len(ephys.EphysEpoch.Insertion & {"experiment_name": EXPERIMENT_NAME})
    print_ok(f"ProbeInsertion: {pi}, EphysEpoch: {ee}, EphysEpoch.Insertion: {ei}")

    return True


def step_ingest_chunks(dry_run=False):
    """Step 15: Run ingest_chunks(experiment_name) — creates EphysChunk entries."""
    print_header(14, "Ingest Ephys Chunks")

    if dry_run:
        print_info(f"Would call: EphysChunk.ingest_chunks('{EXPERIMENT_NAME}')")
        return True

    from aeon.dj_pipeline import ephys

    print_info("Ingesting ephys chunks (this may take a while for 294 epochs)...")
    ephys.EphysChunk.ingest_chunks(EXPERIMENT_NAME)

    chunks = (ephys.EphysChunk & {"experiment_name": EXPERIMENT_NAME}).fetch(as_dict=True)
    if not chunks:
        print_fail("No EphysChunk entries created")
        return False

    print_ok(f"EphysChunk entries: {len(chunks)}")

    from collections import Counter
    insertion_counts = Counter(c["insertion_number"] for c in chunks)
    for ins_num, count in sorted(insertion_counts.items()):
        print_info(f"  insertion {ins_num}: {count} chunks")

    return True


def step_create_blocks(dry_run=False):
    """Step 16: Create EphysBlock entries (30h blocks, 6h overlap)."""
    print_header(15, "Create EphysBlock Entries")

    import pandas as pd

    block_start = pd.Timestamp(BLOCK_START)
    blocks_to_create = []
    for i in range(N_BLOCKS):
        start = block_start + pd.Timedelta(hours=i * BLOCK_ADVANCE_HOURS)
        end = start + pd.Timedelta(hours=BLOCK_DURATION_HOURS)
        blocks_to_create.append((start, end))

    if dry_run:
        print_info(f"{N_BLOCKS} blocks, {BLOCK_DURATION_HOURS}h each, {BLOCK_DURATION_HOURS - BLOCK_ADVANCE_HOURS}h overlap:")
        for i, (start, end) in enumerate(blocks_to_create, 1):
            print_info(f"  Block {i}: {start} -> {end}")
        print_info(f"Would create blocks for each ProbeInsertion ({len(PROBE_LABELS)} probes)")
        print_info(f"Total: {N_BLOCKS * len(PROBE_LABELS)} blocks")
        return True

    from aeon.dj_pipeline import ephys

    probe_insertions = (ephys.ProbeInsertion & {"experiment_name": EXPERIMENT_NAME}).fetch(as_dict=True)
    if not probe_insertions:
        print_fail("No ProbeInsertions found — run step 14 first")
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

    print_ok(f"Created {total_blocks} EphysBlock entries ({len(probe_insertions)} probes x {N_BLOCKS} blocks)")
    for i, (start, end) in enumerate(blocks_to_create, 1):
        print_info(f"  Block {i}: {start} -> {end}")

    return True


def step_populate_block_info(dry_run=False):
    """Step 17: Run EphysBlockInfo.populate()."""
    print_header(16, "Populate EphysBlockInfo")

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
        print_fail("No EphysBlockInfo entries")
        return False

    print_ok(f"EphysBlockInfo entries: {len(block_infos)}")
    for bi in block_infos:
        chunk_count = len(ephys.EphysBlockInfo.Chunk & bi)
        print_info(
            f"  ins={bi['insertion_number']}, "
            f"{bi['block_start']} -> {bi['block_end']}: "
            f"{bi['block_duration']:.1f}h, {chunk_count} chunks"
        )

    return True


# ===========================================================================
# PHASE 3: Spike Sorting Setup
# ===========================================================================

def step_create_electrode_groups(dry_run=False):
    """Step 18: Create electrode group(s) for sorting."""
    print_header(17, "Create Electrode Groups")

    if dry_run:
        print_info(f"Would create 1 group: {ELECTRODE_GROUP_NAME} ({ELECTRODE_GROUP_SIZE} channels)")
        return True

    from aeon.dj_pipeline import spike_sorting

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
    print_ok(f"Group {ELECTRODE_GROUP_NAME} ({len(group_electrodes)} electrodes)")

    return True


def step_insert_sorting_params(dry_run=False):
    """Step 19: Insert SortingParamSet (Kilosort4)."""
    print_header(18, "Insert Sorting Parameters")

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
                "paramset_description": "Production Kilosort4 with SpikeInterface",
                "params": params,
            },
        )
        print_ok(f"Inserted paramset_id={PARAMSET_ID} ({SORTING_METHOD})")
    else:
        print_ok(f"SortingParamSet {PARAMSET_ID} already exists")

    return True


def step_create_sorting_tasks(dry_run=False):
    """Step 20: Create SortingTask entries (block x electrode group x probe)."""
    print_header(19, "Create SortingTask Entries")

    if dry_run:
        print_info("Would create SortingTask for each block x electrode group x probe")
        return True

    from aeon.dj_pipeline import ephys, spike_sorting

    blocks = (ephys.EphysBlock & {"experiment_name": EXPERIMENT_NAME}).fetch(as_dict=True)
    if not blocks:
        print_fail("No EphysBlock entries — run step 16 first")
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
    """Step 21: Run PreProcessing.populate()."""
    print_header(20, "Run PreProcessing")

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


def step_spike_sorting_info(dry_run=False):
    """Step 22: Show SpikeSorting status (SLURM submission needed)."""
    print_header(21, "SpikeSorting Status (SLURM)")

    print_info("*** SpikeSorting requires SLURM submission ***")
    print_info("Edit aeon/dj_pipeline/scripts/prod_spike_sorting.py, then: sbatch prod_spike_sorting.sh")
    print_info("")
    print_info("IMPORTANT: Submit ONE block at a time (30h blocks are large)")
    print_info("IMPORTANT: Do NOT change code while SLURM jobs are queued!")
    print_info("")

    if dry_run:
        return True

    from aeon.dj_pipeline import spike_sorting

    pending = len(spike_sorting.SortingTask - spike_sorting.SpikeSorting)
    done = len(spike_sorting.SpikeSorting & {"experiment_name": EXPERIMENT_NAME})
    print_info(f"  Pending: {pending} tasks")
    print_info(f"  Complete: {done} tasks")

    if pending > 0:
        print_info(f"\n  Pending keys:")
        pending_keys = (spike_sorting.SortingTask - spike_sorting.SpikeSorting).fetch("KEY")
        for k in pending_keys:
            print_info(f"    ins={k['insertion_number']} {k['block_start']} -> {k['block_end']}")

    return True


# ===========================================================================
# PHASE 4: Post-Sorting (Curation, Sync, Unit Matching)
# ===========================================================================

def step_post_processing(dry_run=False):
    """Step 23: Run PostProcessing.populate()."""
    print_header(22, "Run PostProcessing")

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
    """Step 24: Run SortedSpikes.populate()."""
    print_header(23, "Run SortedSpikes + Waveform + SortingQuality")

    if dry_run:
        print_info("Would call: SortedSpikes.populate(), Waveform.populate(), SortingQuality.populate()")
        return True

    from aeon.dj_pipeline import spike_sorting

    print_info("Running SortedSpikes.populate()...")
    spike_sorting.SortedSpikes.populate(
        display_progress=True, suppress_errors=SUPPRESS_ERRORS
    )
    count = len(spike_sorting.SortedSpikes & {"experiment_name": EXPERIMENT_NAME})
    units = len(spike_sorting.SortedSpikes.Unit & {"experiment_name": EXPERIMENT_NAME})
    print_ok(f"SortedSpikes: {count} entries, {units} units")

    print_info("Running Waveform.populate()...")
    spike_sorting.Waveform.populate(
        display_progress=True, suppress_errors=SUPPRESS_ERRORS
    )
    print_ok(f"Waveform: {len(spike_sorting.Waveform & {'experiment_name': EXPERIMENT_NAME})} entries")

    print_info("Running SortingQuality.populate()...")
    spike_sorting.SortingQuality.populate(
        display_progress=True, suppress_errors=SUPPRESS_ERRORS
    )
    print_ok(f"SortingQuality: {len(spike_sorting.SortingQuality & {'experiment_name': EXPERIMENT_NAME})} entries")

    return True


def step_approve_auto_curation(dry_run=False):
    """Step 25: Auto-approve curation (raw sorting as official)."""
    print_header(24, "Approve Automatic Curation")

    if dry_run:
        print_info("Would insert ManualCuration + OfficialCuration for each block")
        print_info("Would run ApplyOfficialCuration.populate()")
        return True

    from aeon.dj_pipeline import spike_sorting
    from aeon.dj_pipeline import spike_sorting_curation as curation

    # CurationMethod
    if not (curation.CurationMethod & {"curation_method": "SpikeInterface"}):
        curation.CurationMethod.insert1(
            {"curation_method": "SpikeInterface"},
            skip_duplicates=True,
        )
        print_ok("CurationMethod 'SpikeInterface' inserted")

    sorting_entries = (spike_sorting.SpikeSorting & {"experiment_name": EXPERIMENT_NAME}).fetch("KEY")
    now = datetime.now(timezone.utc)

    mc_count = 0
    oc_count = 0

    for sorting_key in sorting_entries:
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

        sorted_pk = {k: sorting_key[k] for k in spike_sorting.SortedSpikes.primary_key}
        if not (curation.OfficialCuration & sorted_pk):
            curation.OfficialCuration.insert1(
                {**sorted_pk, "curation_id": 0},
                skip_duplicates=True,
            )
            oc_count += 1

    print_ok(f"ManualCuration: {mc_count} new entries")
    print_ok(f"OfficialCuration: {oc_count} new entries")

    print_info("Running ApplyOfficialCuration.populate()...")
    curation.ApplyOfficialCuration.populate(
        display_progress=True, suppress_errors=SUPPRESS_ERRORS
    )

    aoc_count = len(curation.ApplyOfficialCuration & {"experiment_name": EXPERIMENT_NAME})
    print_ok(f"ApplyOfficialCuration: {aoc_count} entries")

    return True


def step_synced_spikes(dry_run=False):
    """Step 26: Run SyncedSpikes.populate()."""
    print_header(25, "Run SyncedSpikes")

    if dry_run:
        print_info("Would call: SyncedSpikes.populate()")
        return True

    from aeon.dj_pipeline import spike_sorting

    print_info("Running SyncedSpikes.populate()...")
    spike_sorting.SyncedSpikes.populate(
        display_progress=True, suppress_errors=SUPPRESS_ERRORS
    )

    count = len(spike_sorting.SyncedSpikes & {"experiment_name": EXPERIMENT_NAME})
    units = len(spike_sorting.SyncedSpikes.Unit & {"experiment_name": EXPERIMENT_NAME})
    print_ok(f"SyncedSpikes: {count} entries, {units} units")

    return True


def step_insert_matching_params(dry_run=False):
    """Step 27: Insert UnitMatchingParamSet (seed = first block)."""
    print_header(26, "Insert Unit Matching Parameters")

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
                "matching_paramset_description": "Production: seed at first block, delta_time=0.4ms",
                "params": {"delta_time": 0.4},
            },
        )
        print_ok(f"UnitMatchingParamSet: seed_block_start={BLOCK_START}")
    else:
        print_ok(f"UnitMatchingParamSet {matching_paramset_id} already exists")

    return True


def step_unit_matching(dry_run=False):
    """Step 28: Run UnitMatching.populate() (iterative frontier expansion)."""
    print_header(27, "Run Unit Matching")

    if dry_run:
        print_info("Would call: UnitMatching.populate() iteratively")
        return True

    from aeon.dj_pipeline import spike_sorting

    # UnitMatching processes one frontier at a time via key_source
    # Keep calling populate until no more work
    iteration = 0
    while True:
        iteration += 1
        remaining = len(spike_sorting.UnitMatching.key_source - spike_sorting.UnitMatching)
        if remaining == 0:
            break
        print_info(f"Iteration {iteration}: {remaining} blocks remaining...")
        spike_sorting.UnitMatching.populate(
            display_progress=True, suppress_errors=SUPPRESS_ERRORS
        )

    count = len(spike_sorting.UnitMatching & {"experiment_name": EXPERIMENT_NAME})
    global_units = len(spike_sorting.GlobalUnit & {"experiment_name": EXPERIMENT_NAME})
    matched_units = len(spike_sorting.UnitMatching.Unit & {"experiment_name": EXPERIMENT_NAME})
    spikes_entries = len(spike_sorting.UnitMatching.Spikes & {"experiment_name": EXPERIMENT_NAME})

    print_ok(f"UnitMatching: {count} entries ({iteration} iterations)")
    print_ok(f"GlobalUnit: {global_units}")
    print_ok(f"UnitMatching.Unit: {matched_units}")
    print_ok(f"UnitMatching.Spikes: {spikes_entries}")

    return True


# ===========================================================================
# Step Registry
# ===========================================================================

STEPS = [
    # Phase 0: Reconnaissance
    (0, "recon_connection", step_recon_connection),
    (0, "recon_existing_tables", step_recon_existing_tables),
    (0, "recon_epochs", step_recon_epochs),
    (0, "recon_probes", step_recon_probes),
    (0, "recon_subjects", step_recon_subjects),
    (0, "recon_existing_data", step_recon_existing_data),
    # Phase 1: Drop Old Tables
    (1, "safety_check", step_safety_check),
    (1, "drop_curation_schema", step_drop_curation_schema),
    (1, "drop_spike_sorting_schema", step_drop_spike_sorting_schema),
    (1, "drop_ephys_tables", step_drop_ephys_tables),
    (1, "verify_clean_state", step_verify_clean_state),
    # Phase 2: Ingestion
    (2, "verify_existing_data", step_verify_existing_data),
    (2, "manual_ephys_setup", step_manual_ephys_setup),
    (2, "ingest_chunks", step_ingest_chunks),
    (2, "create_blocks", step_create_blocks),
    (2, "populate_block_info", step_populate_block_info),
    # Phase 3: Spike Sorting Setup
    (3, "create_electrode_groups", step_create_electrode_groups),
    (3, "insert_sorting_params", step_insert_sorting_params),
    (3, "create_sorting_tasks", step_create_sorting_tasks),
    (3, "preprocessing", step_preprocessing),
    (3, "spike_sorting_info", step_spike_sorting_info),
    # Phase 4: Post-Sorting
    (4, "post_processing", step_post_processing),
    (4, "sorted_spikes", step_sorted_spikes),
    (4, "approve_auto_curation", step_approve_auto_curation),
    (4, "synced_spikes", step_synced_spikes),
    (4, "insert_matching_params", step_insert_matching_params),
    (4, "unit_matching", step_unit_matching),
]


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Production deployment of ephys v2 pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phases:
  0  Reconnaissance       Steps 1-6    (read-only, no changes)
  1  Drop Old Tables      Steps 7-11   (DESTRUCTIVE — drops production data)
  2  Ingestion            Steps 12-16  (verify existing + insert new data)
  3  Spike Sorting Setup  Steps 17-21  (setup + preprocessing, SLURM at 21)
  4  Post-Sorting         Steps 22-27  (curation, sync, unit matching)

Steps:
  --- Phase 0: Reconnaissance ---
   1  recon_connection      Verify DB + ceph
   2  recon_existing_tables List current production tables
   3  recon_epochs          Discover epochs + time range
   4  recon_probes          Inspect probe structure
   5  recon_subjects        Check subject table
   6  recon_existing_data   Show existing Experiment, Probe, etc. values

  --- Phase 1: Drop Old Tables ---
   7  safety_check          Final safety verification
   8  drop_curation_schema  Drop spike_sorting_curation
   9  drop_spike_sorting    Drop spike_sorting
  10  drop_ephys_tables     Drop changed ephys tables (keep Lookups)
  11  verify_clean_state    Verify + recreate schemas

  --- Phase 2: Ingestion ---
  12  verify_existing_data  Verify Experiment/Subject/Probe exist (error if not)
  13  manual_ephys_setup    ProbeInsertion + EphysEpoch (all epochs)
  14  ingest_chunks         EphysChunk.ingest_chunks()
  15  create_blocks         30h blocks, 6h overlap
  16  populate_block_info   EphysBlockInfo.populate()

  --- Phase 3: Spike Sorting ---
  17  create_electrode_groups  Electrode group 0-95
  18  insert_sorting_params    Kilosort4 paramset
  19  create_sorting_tasks     SortingTask entries
  20  preprocessing            PreProcessing.populate()
  21  spike_sorting_info       *** SLURM submission info ***

  --- Phase 4: Post-Sorting ---
  22  post_processing       PostProcessing.populate()
  23  sorted_spikes         SortedSpikes + Waveform + SortingQuality
  24  approve_auto_curation ManualCuration + OfficialCuration
  25  synced_spikes         SyncedSpikes.populate()
  26  insert_matching_params UnitMatchingParamSet
  27  unit_matching          UnitMatching.populate()
        """,
    )
    parser.add_argument("--step", type=int, help="Run a single step (1-28)")
    parser.add_argument("--phase", type=int, choices=[0, 1, 2, 3, 4], help="Run all steps in a phase")
    parser.add_argument("--dry-run", action="store_true", help="Preview without executing")
    args = parser.parse_args()

    # Phase 0 is read-only, so skip config verification for placeholders
    needs_config = True
    if args.phase == 0 or (args.step and args.step <= 6):
        needs_config = False

    if not args.dry_run:
        verify_production_or_exit()
        if needs_config:
            verify_config_ready()

    import datajoint as dj
    prefix = dj.config["custom"].get("database.prefix", "")
    host = dj.config.get("database.host", "")

    print("=" * 60)
    print("  Ephys v2 PRODUCTION Deployment")
    print(f"  Database:   {host} / {prefix}")
    print(f"  Experiment: {EXPERIMENT_NAME}")
    if needs_config:
        print(f"  Subject:    {SUBJECT}")
        print(f"  Probes:     {PROBE_LABELS}")
        print(f"  Blocks:     {N_BLOCKS} x {BLOCK_DURATION_HOURS}h ({BLOCK_DURATION_HOURS - BLOCK_ADVANCE_HOURS}h overlap)")
    if args.dry_run:
        print("  Mode: DRY RUN")
    print("=" * 60)

    if args.step:
        if args.step < 1 or args.step > len(STEPS):
            print(f"\nX Invalid step {args.step}. Valid range: 1-{len(STEPS)}")
            sys.exit(1)
        step_entry = STEPS[args.step - 1]
        _, step_name, step_func = step_entry
        success = step_func(dry_run=args.dry_run)
        if not success:
            print(f"\nX Step {args.step} ({step_name}) failed.")
            sys.exit(1)
        print(f"\n+ Step {args.step} ({step_name}) completed.")

    elif args.phase is not None:
        phase_steps = [
            (i, name, func)
            for i, (phase, name, func) in enumerate(STEPS, 1)
            if phase == args.phase
        ]
        for step_num, step_name, step_func in phase_steps:
            try:
                success = step_func(dry_run=args.dry_run)
                if not success:
                    print(f"\nX Step {step_num} ({step_name}) failed. Stopping.")
                    sys.exit(1)
            except Exception as e:
                print(f"  X Step {step_num} ({step_name}) raised: {e}")
                import traceback
                traceback.print_exc()
                print(f"\nX Failed at step {step_num}. Fix and re-run with --step {step_num}")
                sys.exit(1)

        print(f"\n+ Phase {args.phase} completed!")

    else:
        print("\n  No phase or step specified. Use --phase or --step.")
        print("  Start with: --phase 0 (reconnaissance)")
        print("  Or: --dry-run --phase 2 (preview ingestion)")
        sys.exit(0)


if __name__ == "__main__":
    main()
