"""Analysis-facing ephys products on the behavioural chunk grain.

``SpikeTrains`` re-chunks curated, HARP-synced spike trains from the ephys rig's
``EphysChunk`` grain to ``acquisition.Chunk``, so spikes and behaviour join on
``(experiment_name, chunk_start)`` with no time arithmetic at the call site.

The table has **no foreign key to the sorted data**. That is deliberate: a
behavioural hour can be covered by several ``UnitMatching`` rows, so keying on them
would fragment the object for every user, forever. The cost is that nothing
invalidates a row automatically — ``source_blocks`` records what went in and
``SpikeTrains.stale()`` finds rows whose inputs have moved on. See
``docs/specs/SPEC_SPIKE_TRAINS.md``.
"""

import datajoint as dj

from aeon.dj_pipeline import acquisition, ephys, get_schema_name, spike_sorting

schema = dj.Schema(get_schema_name("processed_ephys"))


@schema
class SpikeTrains(dj.Computed):
    definition = """
    # Curated, HARP-synced spike trains for one behavioural chunk, as a pynapple TsGroup
    -> acquisition.Chunk                 # experiment_name, chunk_start (BEHAVIOURAL grain)
    -> ephys.ProbeInsertion              # subject, insertion_number
    ---
    n_units: int32                       # units in the roster, including silent ones
    n_spikes: int64                      # total spikes across all units
    coverage_frac: float32               # covered seconds / (chunk_end - chunk_start)
    n_partial_units: int32               # units sorted for less than the full chunk; 0 normally
    source_blocks: json                  # contributing EphysBlock starts + matching paramset
    spikes: <pynapple@dj_store>          # TsGroup; HARP seconds since 1904-01-01
    """

    @property
    def key_source(self):
        """Behavioural chunks that some matched ephys chunk overlaps, per insertion.

        Half-open overlap: an ephys chunk ending exactly at ``chunk_start`` belongs
        to the previous behavioural chunk. Restricted to blocks ``UnitMatching`` has
        run for, so a chunk whose ephys is sorted but unmatched stays uncomputable
        rather than producing a row with no units.
        """
        matched = (
            (spike_sorting.UnitMatching * ephys.EphysBlockInfo.Chunk)
            .proj()
            .join(
                ephys.EphysChunk.proj(eph_start="chunk_start", eph_end="chunk_end"),
                semantic_check=False,
            )
        )
        overlap = "eph_start < chunk_end AND eph_end > chunk_start"
        return dj.U("experiment_name", "chunk_start", "subject", "insertion_number") & (
            acquisition.Chunk.join(matched, semantic_check=False) & overlap
        )
