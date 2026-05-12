# Step 4 -- Curation

After spike sorting completes, you have a set of automatically detected units
(putative neurons). Curation is the process of reviewing those units and
labeling them as **good**, **multi-unit activity (MUA)**, or **noise**. You can
also merge units that were incorrectly split, split units that were incorrectly
merged, or remove units entirely.

The pipeline supports two curation paths:

- **Path A -- Auto-approval:** Skip manual review and accept the automated
  results as-is. Good for testing or when you plan to curate later.
- **Path B -- Manual curation:** Review units interactively in the
  SpikeInterface GUI, then save your labels to the database.

Both paths end at the same place: a `ManualCuration` entry, an
`OfficialCuration` entry, and an `ApplyOfficialCuration.populate()` call that
applies the curation to downstream tables.

---

## Path A -- Auto-Approval (Recommended for Testing)

If you want to move through the pipeline quickly -- for testing, or because you
plan to come back and curate properly later -- you can auto-approve the sorting
results. This creates the required database entries without modifying any spike
data.

```python
from datetime import datetime, timezone
from aeon.dj_pipeline import spike_sorting
from aeon.dj_pipeline import spike_sorting_curation as curation

# Ensure CurationMethod exists
if not (curation.CurationMethod & {"curation_method": "SpikeInterface"}):
    curation.CurationMethod.insert1(
        {"curation_method": "SpikeInterface"},
        skip_duplicates=True,
    )

# For each sorted block, create ManualCuration + OfficialCuration
sorting_entries = (spike_sorting.SpikeSorting & {"experiment_name": EXPERIMENT_NAME}).keys()
now = datetime.now(timezone.utc)

for sorting_key in sorting_entries:
    # ManualCuration: curation_id=0, parent_curation_id=-1
    mc_key = {**sorting_key, "curation_id": 0}
    if not (curation.ManualCuration & mc_key):
        curation.ManualCuration.insert1({
            **mc_key,
            "curation_datetime": now,
            "parent_curation_id": -1,
            "curation_method": "SpikeInterface",
            "description": "Auto-approved: no manual curation applied",
        }, skip_duplicates=True)

    # OfficialCuration: points to the ManualCuration entry
    sorted_pk = {k: sorting_key[k] for k in spike_sorting.SortedSpikes.primary_key}
    if not (curation.OfficialCuration & sorted_pk):
        curation.OfficialCuration.insert1(
            {**sorted_pk, "curation_id": 0},
            skip_duplicates=True,
        )

# Apply: auto-detects no curation file + parent=-1 pattern, just
# updates curation_id on existing SortedSpikes -- no data deleted.
curation.ApplyOfficialCuration.populate(display_progress=True)
```

### What auto-approval does

Auto-approval uses `curation_id=0` with `parent_curation_id=-1` and no
curation file on disk. When `ApplyOfficialCuration.populate()` detects this
combination, it knows there are no actual changes to apply. Instead of deleting
and re-populating `SortedSpikes`, it simply updates the `curation_id` field on
the existing entries from -1 to 0. No spike data is deleted, no downstream
tables are invalidated, and no re-computation happens. The sorted results pass
through unchanged.

---

## Path B -- Manual Curation with SpikeInterface GUI

Manual curation lets you interactively review each unit's waveforms, firing
rates, and quality metrics, then label or modify them. The pipeline uses the
SpikeInterface GUI as the curation tool.

### Requirements

- Access to the sorting output files on Ceph (`/ceph/aeon`). You need to be on
  an HPC compute node, or if you are on-site at SWC, you can mount Ceph to
  your local machine.
- The `spikeinterface-gui` Python package (included in the project
  environment).

### Step 1: Launch the GUI

Open `aeon/dj_pipeline/scripts/launch_si_gui.py` in your editor and fill in
the `key` dictionary with your sorting task's primary key fields:

```python
# Look up the exact values from the SpikeSorting table:
from aeon.dj_pipeline import spike_sorting
spike_sorting.SpikeSorting & {"experiment_name": "abcGolden01-aeonx1"}

# Then fill in the key with values from that query.
# These 6 fields are the minimum needed to uniquely identify a sorting result:
key = {
    "experiment_name": "abcGolden01-aeonx1",
    "insertion_number": 1,
    "block_start": "2026-05-05 15:15:51",  # Replace with actual block start
    "block_end": "2026-05-05 15:45:51",    # Replace with actual block end
    "electrode_group": "all",
    "paramset_id": "400",
}
```

The full primary key for a sorting task (all 9 fields) is:

| Field | Description |
|-------|-------------|
| `experiment_name` | Experiment identifier (e.g., `"abcGolden01-aeonx1"`) |
| `subject` | Subject name |
| `insertion_number` | Probe insertion number |
| `block_start` | Block start datetime |
| `block_end` | Block end datetime |
| `probe_type` | Probe hardware type (e.g., `"neuropixels2.0"`) |
| `electrode_config_name` | Electrode configuration name |
| `electrode_group` | Electrode group label (e.g., `"all"`) |
| `paramset_id` | Sorting parameter set ID (e.g., `"400"` for Kilosort 4) |

The 6-field key above works because DataJoint can resolve the remaining fields
(`subject`, `probe_type`, `electrode_config_name`) if the restriction uniquely
identifies a single entry.

Then run the script. The GUI opens with the raw sorting analyzer loaded. You
can review waveforms, autocorrelograms, and quality metrics for each unit.

### Step 2: Review and label units

In the GUI:

- Label units as **good**, **mua** (multi-unit activity), or **noise**.
- Merge units that appear to be the same neuron split across two IDs.
- Remove units that are clearly noise artifacts.
- Periodically click **"Save in analyzer"** to save your progress to disk.
  This protects against data loss if the GUI crashes.

### Step 3: Save the curation to the database

After you close the GUI, open `aeon/dj_pipeline/scripts/save_curation.py` and
fill in the same `key` dictionary. Then run it to save your curation:

```python
from aeon.dj_pipeline import spike_sorting_curation

curation_id = spike_sorting_curation.save_manual_curation(
    key, description="First pass curation"
)
```

This assigns a unique `curation_id` (starting at 1), copies the temporary
`curation_data.json` file to a permanent `curation_data_id{N}.json` file, and
creates entries in the `ManualCuration` table.

### Step 4: Make the curation official

Designate your saved curation as the official version:

```python
spike_sorting_curation.make_curation_official(key, curation_id)
```

Then apply it:

```python
from aeon.dj_pipeline import spike_sorting_curation as curation

curation.ApplyOfficialCuration.populate(display_progress=True)
```

### What happens when manual curation is applied

Unlike auto-approval, applying a real manual curation **deletes** the existing
`SortedSpikes` entry (which had `curation_id=-1`) and all of its downstream
dependents:

- `SortedSpikes.Unit`
- `Waveform` (and `UnitWaveform`, `ChannelWaveform`)
- `SortingQuality` (and `Metric`)
- `SyncedSpikes`

After `ApplyOfficialCuration.populate()` finishes, you must re-populate the
downstream tables in order:

```python
from aeon.dj_pipeline import spike_sorting

# Re-populate SortedSpikes with curated data
spike_sorting.SortedSpikes.populate(display_progress=True)

# Re-populate waveforms and quality metrics
spike_sorting.Waveform.populate(display_progress=True)
spike_sorting.SortingQuality.populate(display_progress=True)

# Re-populate synced spikes (depends on SortedSpikes)
spike_sorting.SyncedSpikes.populate(display_progress=True)
```

These re-populate calls now use the curated analyzer instead of the raw one, so
all downstream data reflects your manual curation decisions.

### Iterating on a curation

You can create multiple curations for the same sorting task. To base a new
curation on an existing one (instead of starting from raw), pass a
`parent_curation_id` when launching the GUI:

```python
spike_sorting_curation.launch_spikeinterface_gui(key, parent_curation_id=1)
```

This loads your previous curation as the starting point, so you can refine it
further without redoing all the work. The system tracks the parent-child
relationship in the database for lineage.

### Restoring raw sorting

If you need to undo an official curation and go back to the raw sorting
results:

```python
spike_sorting_curation.restore_raw_sorting(key)
```

This removes the `OfficialCuration` entry and deletes all curated data from the
database. The worker manager then re-populates everything using the raw
analyzer. The curated analyzer files remain on disk for reference -- nothing is
deleted from the filesystem.

---

## DB Steps Summary

Regardless of which path you take, the database flow is the same:

1. **`ManualCuration`** -- Record what curation was done (or that none was
   done, in the auto-approval case).
2. **`OfficialCuration`** -- Designate which `ManualCuration` entry is the
   official one for this sorting task.
3. **`ApplyOfficialCuration.populate()`** -- Apply the official curation. For
   auto-approval this just updates `curation_id`; for manual curation this
   deletes and re-creates downstream entries with curated data.

---

## Reference

For full technical details on the curation system -- table schemas, file
organization, path resolution, and all available functions -- see
[`docs/specs/SPEC_SPIKE_SORTING_CURATION.md`](../specs/SPEC_SPIKE_SORTING_CURATION.md).

The reference scripts used above live in the repository at:

- `aeon/dj_pipeline/scripts/launch_si_gui.py` -- Launch the SpikeInterface GUI
- `aeon/dj_pipeline/scripts/save_curation.py` -- Save, make official, or
  restore curations
