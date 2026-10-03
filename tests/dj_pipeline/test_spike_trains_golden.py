"""SpikeTrains on the golden dataset: real spikes re-chunked onto behavioural hours.

The synthetic suite pins the arithmetic; this pins that it survives contact with real
data — real spike times, a real sorted block, re-chunked onto a wall-clock hour that
aligns with neither block boundary.

What this dataset cannot reach: the behavioural arm records ~07:51:34-08:00, while the
two sorted blocks overlap at 08:39:47-08:59:47. No behavioural chunk covers that span,
so no row here draws on more than one block. Cross-block ownership, partial coverage
and the stale-after-a-late-block path stay with the synthetic suite, which exists to
construct exactly the geometry the real recording happens not to have.

Specialized tier: the fixture chain populates UnitMatching over ~2M spikes, twice.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.specialized


@pytest.fixture(scope="session")
def golden_spike_trains(ephys_unit_matching_populated, ephys_full_pipeline, ctx):
    """Ingest the behavioural arm's chunks, then populate SpikeTrains over them.

    The ephys chain drives ``ephys.EphysEpoch.ingest_epochs``, which leaves no
    ``acquisition.Epoch`` rows, so both behavioural steps are taken here. Both arms
    share one ``experiment_name``, so ``SpikeTrains.key_source`` joins them on it;
    chunk discovery uses CameraNest via ``_ref_device_mapping`` and reads only the
    ``raw`` directory, leaving the ephys arm alone.
    """
    from aeon.dj_pipeline import processed_ephys

    acquisition = ephys_full_pipeline["acquisition"]
    exp_name = ctx.cfg["experiment_name"]

    acquisition.Epoch.ingest_epochs(exp_name)
    acquisition.Chunk.ingest_chunks(exp_name)
    assert acquisition.Chunk & {"experiment_name": exp_name}, (
        f"no behavioural chunks ingested for {exp_name}. Both arms must share one "
        "experiment row, and the behaviour tree must be registered as its 'raw' directory."
    )

    processed_ephys.SpikeTrains.populate(suppress_errors=False)
    rows = (processed_ephys.SpikeTrains & {"experiment_name": exp_name}).to_dicts()
    assert rows, "SpikeTrains populated nothing - key_source found no overlap"
    return {
        "module": processed_ephys,
        "acquisition": acquisition,
        "rows": rows,
        "exp_name": exp_name,
    }


def _fetch_tsgroup(module, row):
    """Fetch the stored TsGroup for one SpikeTrains row."""
    return (module.SpikeTrains & {k: row[k] for k in module.SpikeTrains.primary_key}).fetch1("spikes")


class TestGoldenSpikeTrains:
    """What must hold when real spikes are re-chunked onto behavioural hours."""

    def test_conserves_spikes_against_unit_matching(self, golden_spike_trains, ctx):
        """Test that every matched spike inside a chunk is kept, exactly once.

        Counted straight off ``UnitMatching.Spikes`` with no block restriction. Its
        unique index spans ``(experiment, subject, insertion, global_unit,
        chunk_start)`` across all blocks, so a spike cannot be stored twice and a
        plain sum cannot double-count. That makes this independent of ``make()``'s
        own block selection: if it drops a covering block, the counts diverge here.
        """
        exp_name = golden_spike_trains["exp_name"]
        acquisition = golden_spike_trains["acquisition"]

        source_rows = (ctx.spike_sorting.UnitMatching.Spikes & {"experiment_name": exp_name}).to_dicts()
        assert source_rows, "UnitMatching.Spikes is empty - nothing to conserve"
        all_times = [np.asarray(r["spike_times"], dtype="datetime64[ns]") for r in source_rows]

        for row in golden_spike_trains["rows"]:
            start, end = (acquisition.Chunk & row).fetch1("chunk_start", "chunk_end")
            lo, hi = np.datetime64(start), np.datetime64(end)
            expected = sum(int(((t >= lo) & (t < hi)).sum()) for t in all_times)

            assert row["n_spikes"] == expected, (
                f"chunk {start}: stored {row['n_spikes']} spikes, "
                f"UnitMatching.Spikes has {expected} in [{start}, {end})"
            )

        assert sum(r["n_spikes"] for r in golden_spike_trains["rows"]) > 0, (
            "re-chunked to zero spikes on real data"
        )

    def test_source_blocks_are_exactly_the_matched_overlapping_blocks(self, golden_spike_trains, ctx):
        """Test that source_blocks records every matched block covering the chunk.

        Recomputed here from ``EphysBlock * UnitMatching`` rather than reusing
        ``_covering_blocks``, so a block silently dropped from the provenance record
        shows up as a set difference. This is the substitute for the foreign key the
        table does without, so it has to be exact rather than non-empty.

        Note the golden behavioural arm runs ~07:51:34-08:00, which overlaps only the
        first sorted block; the two blocks overlap at 08:39:47-08:59:47, where there
        is no behavioural data. The multi-block path is therefore unreachable on this
        dataset and stays covered by the synthetic suite.
        """
        acquisition = golden_spike_trains["acquisition"]

        for row in golden_spike_trains["rows"]:
            start, end = (acquisition.Chunk & row).fetch1("chunk_start", "chunk_end")
            insertion = {k: row[k] for k in ("experiment_name", "subject", "insertion_number")}
            matched = (
                ctx.ephys.EphysBlock * ctx.spike_sorting.UnitMatching
                & insertion
                & f'block_start < "{end}" AND block_end > "{start}"'
            ).to_dicts()
            expected = {f"{b['block_start']}/{b['block_end']}" for b in matched}

            assert set(row["source_blocks"]) == expected, (
                f"chunk {start}: source_blocks {sorted(row['source_blocks'])} "
                f"!= matched blocks overlapping the window {sorted(expected)}"
            )
            assert expected, f"chunk {start} has a row but no matched covering block"

    def test_times_are_harp_absolute_and_inside_the_support(self, golden_spike_trains):
        """Test that times are Harp seconds since 1904 and lie within time_support.

        A chunk-relative or Unix-epoch offset would still round-trip and still plot;
        it would only be wrong once joined against behaviour.
        """
        module = golden_spike_trains["module"]
        tsgroup = _fetch_tsgroup(module, golden_spike_trains["rows"][0])

        assert tsgroup.time_support.start[0] > 3.8e9, "times are not on the Harp epoch"
        for unit in tsgroup.index:
            times = np.asarray(tsgroup[unit].t)
            if times.size:
                assert times.min() >= tsgroup.time_support.start[0]
                assert times.max() <= tsgroup.time_support.end[-1]

    def test_roster_and_coverage_match_the_object(self, golden_spike_trains):
        """Test that the summary columns agree with the stored TsGroup.

        ``n_units`` counts the roster including units sorted over the chunk but
        silent in it, so it must match the TsGroup's index rather than the number
        of units that fired.
        """
        module = golden_spike_trains["module"]
        for row in golden_spike_trains["rows"]:
            assert 0.0 < row["coverage_frac"] <= 1.0, f"coverage_frac {row['coverage_frac']} out of range"
            assert row["n_partial_units"] <= row["n_units"]

            tsgroup = _fetch_tsgroup(module, row)
            assert len(tsgroup.index) == row["n_units"]
            assert int(sum(len(tsgroup[u]) for u in tsgroup.index)) == row["n_spikes"]

    def test_stale_is_empty_after_a_clean_populate(self, golden_spike_trains):
        """Test that nothing reports stale straight after populating.

        ``stale()`` substitutes for the foreign key this table deliberately lacks,
        so a false positive on untouched data would make it useless.
        """
        module = golden_spike_trains["module"]
        assert module.SpikeTrains.stale() == []
