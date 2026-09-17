"""Builds a controlled behaviour + ephys scenario for SpikeTrains tests.

Timing is chosen so the three cases that can break re-chunking all land where an
assertion can see them::

    behavioural chunks  [08:00-09:00)   [09:00-10:00)   [10:00-11:00)
    ephys chunks         08:00-08:30     09:00-09:20     (none)
                         08:30-09:00     09:40-10:00
    block A  08:00-09:30  units {1, 2}   -- covers 08:00-09:20
    block B  09:00-10:00  units {1, 3}   -- covers 09:00-09:20, 09:40-10:00

So:

- **08:00 chunk** — one block, full coverage, roster {1, 2}. The ordinary case.
- **09:00 chunk** — *both* blocks, and an ephys gap 09:20-09:40. Coverage is two
  intervals. Unit 2 was only ever sorted by block A, which stops at 09:20, so its
  denominator is 1200 s against 2400 s for units 1 and 3. That is the cross-block
  case, and it is the one that silently halves a firing rate if got wrong.
- **10:00 chunk** — no ephys at all, so it must not be computable.

One spike sits at exactly 09:00:00.000000 to pin the half-open rule: it belongs to
the 09:00 chunk, not the 08:00 one.

Block-scoped unit ids are deliberately different from global ones (A: 101, 102;
B: 201, 203) so that confusing the two shows up as a failure rather than a
coincidence.
"""

from datetime import datetime

import numpy as np

DAY = datetime(2026, 5, 11)


def _at(hour, minute=0):
    return DAY.replace(hour=hour, minute=minute)


CHUNKS = [(_at(8), _at(9)), (_at(9), _at(10)), (_at(10), _at(11))]
EPHYS_CHUNKS = [
    (_at(8), _at(8, 30)),
    (_at(8, 30), _at(9)),
    (_at(9), _at(9, 20)),
    (_at(9, 40), _at(10)),
]
BLOCKS = {
    "A": {"window": (_at(8), _at(9, 30)), "units": {1: 101, 2: 102}},
    "B": {"window": (_at(9), _at(10)), "units": {1: 201, 3: 203}},
}
BOUNDARY_SPIKE = _at(9)

PROBE_TYPE = "synthetic-np2"
CONFIG_NAME = "synthetic-config"
ELECTRODES = list(range(4))
SUBJECT = "synthetic-mouse"
PARAMSET_ID = "synthetic"
MATCHING_PARAMSET = 1


def _spikes_in(window, seed, n=8):
    """Deterministic spike times inside ``window``, as ``datetime64[ns]``."""
    rng = np.random.default_rng(seed)
    span = (window[1] - window[0]).total_seconds()
    offsets = np.sort(rng.uniform(0, span, n))
    return np.datetime64(window[0]) + (offsets * 1e9).astype("timedelta64[ns]")


def _blocks_covering(ephys_chunk):
    """Which blocks' windows contain this ephys chunk."""
    return [
        name
        for name, block in BLOCKS.items()
        if block["window"][0] <= ephys_chunk[0] and ephys_chunk[1] <= block["window"][1]
    ]


def build_scenario(experiment_name):
    """Insert the whole scenario and return what the tests need to assert against."""
    from aeon.dj_pipeline import acquisition, ephys, subject

    epoch_start = CHUNKS[0][0]

    # --- subject, epoch, behavioural chunks -------------------------------------
    subject.Subject.insert1(
        {"subject": SUBJECT, "sex": "U", "subject_birth_date": DAY.date(), "subject_description": ""},
        skip_duplicates=True,
    )
    acquisition.Experiment.Subject.insert1(
        {"experiment_name": experiment_name, "subject": SUBJECT}, skip_duplicates=True
    )
    acquisition.Epoch.insert1(
        {"experiment_name": experiment_name, "epoch_start": epoch_start, "directory_type": "raw"},
        skip_duplicates=True,
    )
    acquisition.Chunk.insert(
        [
            {
                "experiment_name": experiment_name,
                "chunk_start": start,
                "chunk_end": end,
                "directory_type": "raw",
                "epoch_start": epoch_start,
            }
            for start, end in CHUNKS
        ],
        skip_duplicates=True,
    )

    # --- probe, config, insertion ------------------------------------------------
    ephys.ProbeType.insert1({"probe_type": PROBE_TYPE}, skip_duplicates=True)
    ephys.ProbeType.Electrode.insert(
        [
            {
                "probe_type": PROBE_TYPE,
                "electrode": e,
                "shank": 0,
                "x_coord": float(e),
                "y_coord": float(e) * 20,
            }
            for e in ELECTRODES
        ],
        skip_duplicates=True,
    )
    ephys.Probe.insert1({"probe": "synthetic-probe", "probe_type": PROBE_TYPE}, skip_duplicates=True)
    ephys.ElectrodeConfig.insert1(
        {"probe_type": PROBE_TYPE, "electrode_config_name": CONFIG_NAME}, skip_duplicates=True
    )
    ephys.ElectrodeConfig.Electrode.insert(
        [
            {"probe_type": PROBE_TYPE, "electrode_config_name": CONFIG_NAME, "electrode": e}
            for e in ELECTRODES
        ],
        skip_duplicates=True,
    )
    insertion = {"experiment_name": experiment_name, "subject": SUBJECT, "insertion_number": 1}
    ephys.ProbeInsertion.insert1({**insertion, "probe": "synthetic-probe"}, skip_duplicates=True)

    # --- ephys epoch + chunks ----------------------------------------------------
    ephys.EphysEpoch.insert1(
        {"experiment_name": experiment_name, "epoch_start": epoch_start, "epoch_dir": "synthetic"},
        skip_duplicates=True,
    )
    ephys.EphysChunk.insert(
        [
            {
                **insertion,
                "chunk_start": start,
                "chunk_end": end,
                "epoch_start": epoch_start,
            }
            for start, end in EPHYS_CHUNKS
        ],
        skip_duplicates=True,
    )

    # --- blocks ------------------------------------------------------------------
    for name, block in BLOCKS.items():
        start, end = block["window"]
        ephys.EphysBlock.insert1(
            {**insertion, "block_start": start, "block_end": end}, skip_duplicates=True
        )
        ephys.EphysBlockInfo.insert1(
            {
                **insertion,
                "block_start": start,
                "block_end": end,
                "block_duration": (end - start).total_seconds() / 3600,
                "probe_type": PROBE_TYPE,
                "electrode_config_name": CONFIG_NAME,
            },
            skip_duplicates=True,
            allow_direct_insert=True,
        )
        ephys.EphysBlockInfo.Chunk.insert(
            [
                {**insertion, "block_start": start, "block_end": end, "chunk_start": c[0]}
                for c in EPHYS_CHUNKS
                if name in _blocks_covering(c)
            ],
            skip_duplicates=True,
            allow_direct_insert=True,
        )

    _build_sorting_chain(experiment_name, insertion)

    return {
        "experiment_name": experiment_name,
        "subject": SUBJECT,
        "insertion_number": 1,
        "insertion_key": insertion,
        "covered_chunk_starts": [CHUNKS[0][0], CHUNKS[1][0]],
        "uncovered_chunk_start": CHUNKS[2][0],
        "boundary_chunk_start": CHUNKS[1][0],
        "span_end": CHUNKS[2][1],
        "units_by_block": {n: set(b["units"]) for n, b in BLOCKS.items()},
        "partial_unit": 2,  # block A only, so it stops at 09:20
        "full_units": {1, 3},
        "expected_spike_counts": _expected_counts(),
    }


def _expected_counts():
    """Total spikes the scenario inserts, per unit and overall."""
    per_unit = {}
    for name, block in BLOCKS.items():
        for global_unit in block["units"]:
            for chunk in EPHYS_CHUNKS:
                if _owner(chunk, global_unit) != name:
                    continue
                n = 8 + (1 if chunk[0] == BOUNDARY_SPIKE and global_unit == 1 else 0)
                per_unit[global_unit] = per_unit.get(global_unit, 0) + n
    return {"per_unit": per_unit, "total": sum(per_unit.values())}


def _owner(ephys_chunk, global_unit):
    """Which block owns this (unit, ephys chunk) pair — first block that found it."""
    for name in ("A", "B"):
        if name in _blocks_covering(ephys_chunk) and global_unit in BLOCKS[name]["units"]:
            return name
    return None


def _build_sorting_chain(experiment_name, insertion):
    """Everything from ElectrodeGroup down to UnitMatching.Spikes."""
    from aeon.dj_pipeline import spike_sorting

    now = datetime.now()
    spike_sorting.ElectrodeGroup.insert1(
        {
            "probe_type": PROBE_TYPE,
            "electrode_config_name": CONFIG_NAME,
            "electrode_group": "shank0",
            "electrode_group_description": "synthetic",
            "electrode_count": len(ELECTRODES),
        },
        skip_duplicates=True,
    )
    spike_sorting.SortingParamSet.insert1(
        {"paramset_id": PARAMSET_ID, "sorting_method": "kilosort4", "params": {}},
        skip_duplicates=True,
    )
    spike_sorting.UnitMatchingParamSet.insert1(
        {
            "matching_paramset_id": MATCHING_PARAMSET,
            "matching_method": "spike_time_overlap",
            "seed_block_start": BLOCKS["A"]["window"][0],
            "params": {},
        },
        skip_duplicates=True,
    )

    for name, block in BLOCKS.items():
        start, end = block["window"]
        block_key = {**insertion, "block_start": start, "block_end": end}
        task = {
            **block_key,
            "probe_type": PROBE_TYPE,
            "electrode_config_name": CONFIG_NAME,
            "electrode_group": "shank0",
            "paramset_id": PARAMSET_ID,
        }
        spike_sorting.SortingTask.insert1(task, skip_duplicates=True)
        for table in (
            spike_sorting.PreProcessing,
            spike_sorting.SpikeSorting,
            spike_sorting.PostProcessing,
        ):
            row = {**task, "execution_time": now, "execution_duration": 0.0}
            if table is spike_sorting.PreProcessing:
                row["sorting_output_dir"] = f"synthetic/{name}"
            table.insert1(row, skip_duplicates=True, allow_direct_insert=True)

        spike_sorting.SortedSpikes.insert1(
            {**task, "execution_time": now, "execution_duration": 0.0, "curation_id": -1},
            skip_duplicates=True,
            allow_direct_insert=True,
        )
        spike_sorting.SortedSpikes.Unit.insert(
            [
                {
                    **task,
                    "unit": local,
                    "probe_type": PROBE_TYPE,
                    "electrode_config_name": CONFIG_NAME,
                    "electrode": 0,
                    "unit_quality": "good",
                    "spike_count": 0,
                    "spike_indices": np.array([], dtype=np.int64),
                    "spike_sites": np.array([], dtype=np.int64),
                    "spike_depths": np.array([], dtype=np.float64),
                }
                for local in block["units"].values()
            ],
            skip_duplicates=True,
            allow_direct_insert=True,
        )
        spike_sorting.SyncedSpikes.insert1(task, skip_duplicates=True, allow_direct_insert=True)

        spike_sorting.GlobalUnit.insert(
            [
                {
                    **insertion,
                    "global_unit": g,
                    "matching_paramset_id": MATCHING_PARAMSET,
                    "probe_type": PROBE_TYPE,
                    "electrode": 0,
                }
                for g in block["units"]
            ],
            skip_duplicates=True,
        )

        match_key = {**task, "matching_paramset_id": MATCHING_PARAMSET}
        spike_sorting.UnitMatching.insert1(
            {**match_key, "execution_time": now, "execution_duration": 0.0},
            skip_duplicates=True,
            allow_direct_insert=True,
        )
        spike_sorting.UnitMatching.Unit.insert(
            [{**match_key, "unit": local, "global_unit": g} for g, local in block["units"].items()],
            skip_duplicates=True,
            allow_direct_insert=True,
        )

        spikes_rows = []
        for g in block["units"]:
            for chunk in EPHYS_CHUNKS:
                if _owner(chunk, g) != name:
                    continue
                times = _spikes_in(chunk, seed=g * 10 + chunk[0].hour * 60 + chunk[0].minute)
                if chunk[0] == BOUNDARY_SPIKE and g == 1:
                    times = np.sort(np.append(times, np.datetime64(BOUNDARY_SPIKE)))
                spikes_rows.append(
                    {
                        **match_key,
                        "global_unit": g,
                        "chunk_start": chunk[0],
                        "spike_times": times,
                        "spike_count": len(times),
                    }
                )
        spike_sorting.UnitMatching.Spikes.insert(
            spikes_rows, skip_duplicates=True, allow_direct_insert=True
        )


def add_late_block(experiment_name):
    """Match a further block over already-covered time, making existing rows stale.

    Block C covers the 08:00 behavioural chunk and finds a unit neither A nor B did,
    so a row written before it existed is now built from an incomplete input set.
    """
    from datetime import datetime

    from aeon.dj_pipeline import ephys, spike_sorting

    insertion = {"experiment_name": experiment_name, "subject": SUBJECT, "insertion_number": 1}
    window = (_at(8), _at(9))
    now = datetime.now()

    ephys.EphysBlock.insert1(
        {**insertion, "block_start": window[0], "block_end": window[1]}, skip_duplicates=True
    )
    ephys.EphysBlockInfo.insert1(
        {
            **insertion,
            "block_start": window[0],
            "block_end": window[1],
            "block_duration": 1.0,
            "probe_type": PROBE_TYPE,
            "electrode_config_name": CONFIG_NAME,
        },
        skip_duplicates=True,
        allow_direct_insert=True,
    )
    ephys.EphysBlockInfo.Chunk.insert(
        [
            {**insertion, "block_start": window[0], "block_end": window[1], "chunk_start": c[0]}
            for c in EPHYS_CHUNKS
            if window[0] <= c[0] and c[1] <= window[1]
        ],
        skip_duplicates=True,
        allow_direct_insert=True,
    )

    task = {
        **insertion,
        "block_start": window[0],
        "block_end": window[1],
        "probe_type": PROBE_TYPE,
        "electrode_config_name": CONFIG_NAME,
        "electrode_group": "shank0",
        "paramset_id": PARAMSET_ID,
    }
    spike_sorting.SortingTask.insert1(task, skip_duplicates=True)
    for table in (
        spike_sorting.PreProcessing,
        spike_sorting.SpikeSorting,
        spike_sorting.PostProcessing,
    ):
        row = {**task, "execution_time": now, "execution_duration": 0.0}
        if table is spike_sorting.PreProcessing:
            row["sorting_output_dir"] = "synthetic/C"
        table.insert1(row, skip_duplicates=True, allow_direct_insert=True)

    spike_sorting.SortedSpikes.insert1(
        {**task, "execution_time": now, "execution_duration": 0.0, "curation_id": -1},
        skip_duplicates=True,
        allow_direct_insert=True,
    )
    spike_sorting.SortedSpikes.Unit.insert1(
        {
            **task,
            "unit": 301,
            "probe_type": PROBE_TYPE,
            "electrode_config_name": CONFIG_NAME,
            "electrode": 0,
            "unit_quality": "good",
            "spike_count": 0,
            "spike_indices": np.array([], dtype=np.int64),
            "spike_sites": np.array([], dtype=np.int64),
            "spike_depths": np.array([], dtype=np.float64),
        },
        skip_duplicates=True,
        allow_direct_insert=True,
    )
    spike_sorting.SyncedSpikes.insert1(task, skip_duplicates=True, allow_direct_insert=True)
    spike_sorting.GlobalUnit.insert1(
        {
            **insertion,
            "global_unit": 4,
            "matching_paramset_id": MATCHING_PARAMSET,
            "probe_type": PROBE_TYPE,
            "electrode": 0,
        },
        skip_duplicates=True,
    )

    match_key = {**task, "matching_paramset_id": MATCHING_PARAMSET}
    spike_sorting.UnitMatching.insert1(
        {**match_key, "execution_time": now, "execution_duration": 0.0},
        skip_duplicates=True,
        allow_direct_insert=True,
    )
    spike_sorting.UnitMatching.Unit.insert1(
        {**match_key, "unit": 301, "global_unit": 4},
        skip_duplicates=True,
        allow_direct_insert=True,
    )
