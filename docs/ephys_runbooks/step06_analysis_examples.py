"""06 -- Analysis Examples
=======================
Starter examples showing how to query spike sorting output and create
common visualizations. These patterns are building blocks for custom
analysis -- not a complete analysis toolkit.

After the pipeline is fully populated (steps 1-5), the main outputs
are:

    SyncedSpikes.Unit  -- spike times in the HARP (behavioral) clock,
                          one row per unit per block. This is the primary
                          table for analysis because spike times are
                          aligned with behavioral events.
    SortedSpikes.Unit  -- spike times in the ONIX (hardware) clock.
                          Use this if you only need relative timing
                          within a single block, or if SyncedSpikes
                          has not been populated yet.
    GlobalUnit         -- persistent unit IDs across blocks. Use these
                          to track the same neuron across multiple
                          recording sessions.

This script demonstrates:
    1. Fetching spike times from the database
    2. Raster plots (units x time)
    3. Peri-event time histograms (PETH)
    4. Discovering behavioral event tables for your experiment

Run from the repo root on an HPC compute node (Ceph must be visible):

    uv run python docs/ephys_runbooks/step06_analysis_examples.py
"""

# --------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------
# (none at module level -- all imports are deferred inside functions
# to avoid DB side effects when the module is imported)

# --------------------------------------------------------------------------
# Configuration -- edit these for your experiment
# --------------------------------------------------------------------------

EXPERIMENT_NAME = "abcGolden01-aeonx1"


# --------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------


def fetch_spike_times(experiment_name, unit_id=None):
    """Fetch spike times from SyncedSpikes (preferred) or SortedSpikes.

    This is the most common query pattern for analysis: fetch spike times
    for one or all units, grouped by (block, unit). The returned dict
    maps each (block_start, unit) pair to a numpy array of spike times.

    SyncedSpikes is preferred because its spike times are in the HARP
    clock -- the same clock used by behavioral events (rewards, beam
    breaks, etc.). If SyncedSpikes has not been populated, the function
    falls back to SortedSpikes (ONIX clock).

    DataJoint query pattern used here:
        table & restriction  -- filters rows matching the restriction
        .to_dicts()          -- returns list of dicts

    Args:
        experiment_name: The experiment to fetch from.
        unit_id: If given, fetch only this unit. If None, fetch all units.

    Returns:
        dict mapping (block_start, unit) -> spike_times numpy array.
    """
    # Deferred imports -- no DB side effects at module level.
    import numpy as np

    from aeon.dj_pipeline import spike_sorting

    restriction = {"experiment_name": experiment_name}

    # Try SyncedSpikes first (preferred -- HARP-aligned times).
    # Fall back to SortedSpikes if SyncedSpikes is not yet populated.
    try:
        unit_table = spike_sorting.SyncedSpikes.Unit
        source_name = "SyncedSpikes"
        if not (spike_sorting.SyncedSpikes & restriction):
            raise ValueError("no SyncedSpikes entries")
    except Exception:
        unit_table = spike_sorting.SortedSpikes.Unit
        source_name = "SortedSpikes"

    # Build the query. The & operator restricts the table to rows
    # matching the given dict. You can chain multiple restrictions:
    #   table & {"col1": val1} & {"col2": val2}
    query = unit_table & restriction
    if unit_id is not None:
        query = query & {"unit": unit_id}

    # fetch() returns columns as separate arrays (positional) or as a
    # list of dicts (as_dict=True). The dict form is easier to iterate.
    entries = query.to_dicts()

    # Build the result dict keyed by (block_start, unit).
    spike_dict = {}
    for entry in entries:
        key = (entry["block_start"], entry["unit"])
        spike_dict[key] = np.asarray(entry["spike_times"])

    print(f"Source: {source_name}")
    print(f"Fetched {len(spike_dict)} (block, unit) entries")
    if spike_dict:
        total_spikes = sum(len(v) for v in spike_dict.values())
        print(f"Total spikes: {total_spikes:,}")

    return spike_dict


def plot_raster(spike_times_dict, title="Raster Plot", save_path=None):
    """Simple raster plot: units on y-axis, spike times on x-axis.

    Each dot represents one spike. Units are sorted by unit number for
    consistent ordering across plots.

    NOTE: On headless HPC nodes (no display), plt.show() will fail.
    Pass save_path to write the figure to disk instead, or set the
    matplotlib backend before importing pyplot:
        import matplotlib
        matplotlib.use("Agg")

    Args:
        spike_times_dict: Dict mapping (block_start, unit) -> spike_times.
            Typically the output of fetch_spike_times().
        title: Plot title string.
        save_path: If given, save figure to this path instead of showing.
            Use this on headless HPC nodes.
    """
    import matplotlib

    # Use non-interactive backend if saving to file (headless-safe).
    # MUST be set before importing pyplot — the backend is locked on import.
    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    # Sort by unit number for consistent y-axis ordering.
    sorted_keys = sorted(spike_times_dict.keys(), key=lambda x: x[1])

    for y_idx, key in enumerate(sorted_keys):
        times = spike_times_dict[key]
        # s=0.1 makes tiny dots so dense firing is visible.
        ax.scatter(times, [y_idx] * len(times), s=0.1, c="black")

    # Label y-axis ticks with actual unit IDs.
    unit_labels = [str(k[1]) for k in sorted_keys]
    ax.set_yticks(range(len(sorted_keys)))
    ax.set_yticklabels(unit_labels, fontsize=6)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Unit")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Raster plot saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_peth(
    spike_times,
    event_times,
    window=(-0.5, 1.0),
    bin_size=0.01,
    title="Peri-Event Time Histogram",
    save_path=None,
):
    """Peri-event time histogram: align spikes to behavioral events.

    For each event, finds all spikes within the time window relative to
    that event. The histogram shows the spike count distribution across
    time bins, revealing whether neural activity is modulated by the
    event.

    Common events to align to:
        - Pellet deliveries (reward)
        - Arena entry/exit times
        - Patch visits

    Args:
        spike_times: 1-D array of spike times (from one unit).
        event_times: 1-D array of event times (same clock as spikes).
        window: Tuple (pre, post) in seconds relative to each event.
            Default: 0.5 s before to 1.0 s after.
        bin_size: Histogram bin width in seconds. Default: 10 ms.
        title: Plot title string.
        save_path: If given, save figure to this path instead of showing.
    """
    import matplotlib
    import numpy as np

    # MUST be set before importing pyplot — the backend is locked on import.
    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # For each event, compute spike times relative to that event and
    # keep only spikes within the window. This is the core PETH
    # computation -- vectorized per event, concatenated across events.
    # Spike times from SyncedSpikes are datetime64[ns], so subtraction
    # produces timedelta64[ns]. Convert to float seconds for binning.
    aligned = []
    for event in event_times:
        relative = spike_times - event
        # Convert timedelta64 to float seconds if needed.
        if hasattr(relative.dtype, "kind") and relative.dtype.kind == "m":
            relative = relative / np.timedelta64(1, "s")
        mask = (relative >= window[0]) & (relative <= window[1])
        aligned.extend(relative[mask])

    if not aligned:
        print("No spikes found within the event window. Try a wider window.")
        return

    bins = np.arange(window[0], window[1] + bin_size, bin_size)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(aligned, bins=bins, color="steelblue", edgecolor="none")
    ax.axvline(0, color="red", linestyle="--", label="Event")
    ax.set_xlabel("Time relative to event (s)")
    ax.set_ylabel("Spike count")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"PETH saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def fetch_behavioral_events(experiment_name):
    """Show how to find and query behavioral event tables.

    Behavioral events live in core pipeline tables (acquisition, streams,
    tracking). The exact tables available depend on your experiment type
    and rig configuration. This function prints what is available and
    shows the query patterns you would use.

    For the ephys runbook, the key insight is: spike times from SyncedSpikes
    are on the HARP clock, and behavioral events are also on the HARP
    clock. This means you can directly compare them -- no additional
    alignment is needed.

    Common behavioral event sources:
        acquisition.Epoch       -- experiment epochs (start/end times)
        acquisition.Chunk       -- 1-hour data chunks
        streams.*               -- auto-generated device stream tables
                                   (e.g., pellet beam breaks, patch states)

    NOTE: The streams module is auto-generated per experiment. The
    available tables depend on the DevicesSchema configured for your
    experiment. Use the pattern below to discover what is available.

    Args:
        experiment_name: The experiment to inspect.
    """
    # Deferred imports -- no DB side effects at module level.
    from aeon.dj_pipeline import acquisition

    restriction = {"experiment_name": experiment_name}

    # ------------------------------------------------------------------
    # 1. Check that the experiment exists
    # ------------------------------------------------------------------
    exp = acquisition.Experiment & restriction
    if not exp:
        print(f"Experiment '{experiment_name}' not found in the database.")
        return

    print(f"Experiment: {experiment_name}")
    exp_info = exp.fetch1()
    print(f"  Type: {exp_info.get('experiment_type', 'N/A')}")

    # ------------------------------------------------------------------
    # 2. Show available epochs (time boundaries of the experiment)
    # ------------------------------------------------------------------
    epochs = (acquisition.Epoch & restriction).to_dicts()
    print(f"\n  Epochs: {len(epochs)}")
    for ep in epochs[:5]:  # Show first 5
        print(f"    {ep['epoch_start']} (dir: {ep.get('epoch_dir', '')})")
    if len(epochs) > 5:
        print(f"    ... and {len(epochs) - 5} more")

    # ------------------------------------------------------------------
    # 3. Show available chunks (1-hour data segments)
    # ------------------------------------------------------------------
    chunk_count = len(acquisition.Chunk & restriction)
    print(f"\n  Chunks: {chunk_count}")

    # ------------------------------------------------------------------
    # 4. Point to streams tables
    # ------------------------------------------------------------------
    print(
        "\n  Stream tables (auto-generated per experiment):"
        "\n  -----------------------------------------------"
        "\n  The streams module contains device-specific tables generated"
        "\n  from your experiment's DevicesSchema. To explore them:"
        "\n"
        "\n    from aeon.dj_pipeline import streams"
        "\n    # List all stream table classes:"
        "\n    stream_tables = [name for name in dir(streams)"
        "\n                     if not name.startswith('_')]"
        "\n    print(stream_tables)"
        "\n"
        "\n  Common stream tables for foraging experiments:"
        "\n    streams.UndergroundFeederBeamBreak  -- pellet delivery events"
        "\n    streams.UndergroundFeederManualDelivery"
        "\n    streams.WeightScaleWeightRaw / WeightFiltered"
        "\n"
        "\n  Query pattern (same as any DataJoint table):"
        "\n    events = (streams.SomeTable & restriction).fetch(as_dict=True)"
    )

    # ------------------------------------------------------------------
    # 5. Alignment reminder
    # ------------------------------------------------------------------
    print(
        "\n  Clock alignment:"
        "\n  -----------------"
        "\n  All stream/behavioral timestamps use the HARP clock."
        "\n  SyncedSpikes.Unit spike_times are also in the HARP clock."
        "\n  You can directly subtract or compare them -- no extra"
        "\n  alignment step is needed."
        "\n"
        "\n  Example: align spikes to pellet deliveries"
        "\n    pellet_times = (streams.UndergroundFeederBeamBreak"
        "\n                    & restriction).fetch('timestamps')"
        "\n    plot_peth(unit_spike_times, pellet_times)"
    )


# --------------------------------------------------------------------------
# Where to go from here
# --------------------------------------------------------------------------
#
# This script covers basic query patterns and starter visualizations.
# For more advanced analysis:
#
# Analysis repositories:
#   - foragingABC_analysis (SainsburyWellcomeCentre/foragingABC_analysis)
#     Contains Adrian's analysis notebooks for foraging experiments,
#     including behavioral decoding and neural population analyses.
#
# DataJoint queries for custom analysis:
#   - Join spike data with behavioral tables:
#       (spike_sorting.SyncedSpikes.Unit * streams.SomeTable & restriction)
#   - Aggregate across blocks using GlobalUnit:
#       (spike_sorting.GlobalUnit * spike_sorting.UnitMatching.Spikes
#        & restriction)
#   - Filter units by quality metrics:
#       good_units = (spike_sorting.SortingQuality.Unit
#                     & "snr > 5" & "isi_violations_ratio < 0.01")
#
# SpikeInterface documentation:
#   - https://spikeinterface.readthedocs.io/
#   - Sorting analyzer extensions (waveforms, PCA, quality metrics)
#   - Comparison and benchmarking tools
#
# DataJoint documentation:
#   - https://datajoint.com/docs/
#   - Query operators: &, -, *, .proj(), .aggr()
#   - Fetch options: as_dict, order_by, limit, format


# --------------------------------------------------------------------------
# Run standalone
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Step 6: Analysis Examples")
    print("=" * 60)

    print("\n--- Fetch spike times ---")
    spike_dict = fetch_spike_times(EXPERIMENT_NAME)

    if spike_dict:
        print("\n--- Raster plot ---")
        plot_raster(spike_dict)

        # Example PETH using the first unit's spikes and synthetic events.
        # In real analysis, you would use actual behavioral event times
        # (e.g., pellet deliveries) instead of synthetic ones.
        print("\n--- PETH example (synthetic events) ---")
        import numpy as np

        first_key = next(iter(spike_dict))
        example_spikes = spike_dict[first_key]
        # Create evenly spaced synthetic events for demonstration.
        # Replace with real event times from fetch_behavioral_events().
        # np.linspace doesn't work with datetime64, so use arange with
        # a timedelta step instead.
        t_min, t_max = example_spikes.min(), example_spikes.max()
        duration = t_max - t_min
        step = duration // 20
        events = np.arange(t_min + step, t_max - step, step)
        plot_peth(example_spikes, events)

    print("\n--- Behavioral events ---")
    fetch_behavioral_events(EXPERIMENT_NAME)

    print("\n" + "=" * 60)
    print("  Step 6 complete.")
    print("=" * 60)
