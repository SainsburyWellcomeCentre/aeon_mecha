"""Unit matching QC plots for the ephys v2 pipeline.

Generates quality-control visualizations to evaluate unit tracking stability
across ephys blocks. Queries UnitMatching, GlobalUnit, and related tables.

Saves figures as PNGs and generates an HTML report for easy viewing.

Usage:
  uv run python -m aeon.dj_pipeline.scripts.ephys_v2_qc_plots
  uv run python -m aeon.dj_pipeline.scripts.ephys_v2_qc_plots --output-dir ./qc_output
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import datajoint as dj
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for HPC
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from aeon.dj_pipeline import ephys, spike_sorting, get_schema_name

# ---------------------------------------------------------------------------
# Configuration (same as ephys_v2_setup.py)
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "social-ephys0.1-aeon3"
SUBJECT = "test-subject-001"
INSERTION_NUMBER = 1

# Colors
CMAP_LONGEVITY = plt.cm.viridis
COLOR_NEW = "#2ecc71"
COLOR_MATCHED = "#3498db"
COLOR_LOST = "#e74c3c"


def fetch_unit_matching_data():
    """Fetch all unit matching data into DataFrames."""
    insertion_key = {
        "experiment_name": EXPERIMENT_NAME,
        "subject": SUBJECT,
        "insertion_number": INSERTION_NUMBER,
    }

    # All blocks
    blocks = pd.DataFrame(
        (ephys.EphysBlock & insertion_key).fetch(
            "block_start", "block_end", as_dict=True
        )
    ).sort_values("block_start").reset_index(drop=True)
    blocks["block_idx"] = range(len(blocks))
    blocks["block_label"] = blocks["block_start"].apply(
        lambda t: t.strftime("%H:%M") if hasattr(t, "strftime") else str(t)[:5]
    )

    # UnitMatching.Unit — which global_unit appears in which block
    unit_entries = pd.DataFrame(
        (spike_sorting.UnitMatching.Unit & insertion_key).fetch(
            "block_start", "block_end", "unit", "global_unit",
            "match_confidence", as_dict=True
        )
    )

    # Spike counts per global unit per block (sum across chunks)
    spike_entries = pd.DataFrame(
        (spike_sorting.UnitMatching.Spikes & insertion_key).fetch(
            "block_start", "block_end", "global_unit", "spike_count",
            as_dict=True
        )
    )

    return blocks, unit_entries, spike_entries


def compute_unit_block_matrix(blocks, unit_entries):
    """Build a (global_unit × block) presence matrix."""
    global_units = sorted(unit_entries["global_unit"].unique())
    n_units = len(global_units)
    n_blocks = len(blocks)

    # Map block_start → block index
    block_start_to_idx = dict(zip(blocks["block_start"], blocks["block_idx"]))

    # Presence matrix: 1 if unit present in block, 0 otherwise
    presence = np.zeros((n_units, n_blocks), dtype=int)
    for _, row in unit_entries.iterrows():
        uid = global_units.index(row["global_unit"])
        bidx = block_start_to_idx.get(row["block_start"])
        if bidx is not None:
            presence[uid, bidx] = 1

    # Longevity = number of blocks each unit appears in
    longevity = presence.sum(axis=1)

    return global_units, presence, longevity


def plot_unit_gantt(blocks, global_units, presence, longevity, output_dir):
    """Gantt-style chart: horizontal lines per unit spanning their blocks.

    Units sorted by longevity (most stable at top), then by first appearance.
    """
    n_units = len(global_units)
    n_blocks = len(blocks)
    max_longevity = longevity.max()

    # Sort: primary by longevity (descending), secondary by first block (ascending)
    first_block = np.array([np.argmax(presence[i]) for i in range(n_units)])
    sort_idx = np.lexsort((first_block, -longevity))

    fig, ax = plt.subplots(figsize=(10, max(4, n_units * 0.06 + 2)))

    for plot_row, uid in enumerate(sort_idx):
        active_blocks = np.where(presence[uid] == 1)[0]
        lon = longevity[uid]
        color = CMAP_LONGEVITY(lon / max_longevity)

        # Draw a solid bar spanning full width of first to last active block
        first_b = active_blocks.min()
        last_b = active_blocks.max()
        left_edge = first_b - 0.45
        right_edge = last_b + 0.45
        ax.barh(
            plot_row, width=right_edge - left_edge, left=left_edge,
            height=0.6, color=color, edgecolor="none",
        )

    # Axis formatting
    ax.set_xlim(-0.5, n_blocks - 0.5)
    ax.set_xticks(range(n_blocks))
    ax.set_xticklabels(blocks["block_label"], fontsize=10)
    ax.set_xlabel("Block start time", fontsize=11)
    ax.set_ylim(-0.5, n_units - 0.5)
    ax.set_ylabel(f"Global units (n={n_units})", fontsize=11)
    ax.set_title("Unit Tracking Across Blocks", fontsize=13, fontweight="bold")
    ax.invert_yaxis()

    # Remove y-tick labels (too many units)
    ax.set_yticks([])

    # Legend for longevity
    handles = []
    for lon_val in range(1, max_longevity + 1):
        handles.append(mpatches.Patch(
            color=CMAP_LONGEVITY(lon_val / max_longevity),
            label=f"{lon_val} block{'s' if lon_val > 1 else ''}"
        ))
    ax.legend(handles=handles, title="Longevity", loc="upper right", fontsize=9)

    plt.tight_layout()
    path = output_dir / "unit_gantt.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_unit_heatmap(blocks, global_units, presence, longevity, output_dir):
    """Heatmap: blocks × global units (sorted by longevity)."""
    n_units = len(global_units)
    n_blocks = len(blocks)

    first_block = np.array([np.argmax(presence[i]) for i in range(n_units)])
    sort_idx = np.lexsort((first_block, -longevity))
    sorted_presence = presence[sort_idx]

    fig, ax = plt.subplots(figsize=(max(6, n_blocks * 1.5 + 2), max(6, n_units * 0.12 + 2)))
    im = ax.imshow(sorted_presence, aspect="auto", cmap="Blues", interpolation="nearest")

    ax.set_xticks(range(n_blocks))
    ax.set_xticklabels(blocks["block_label"], fontsize=10)
    ax.set_xlabel("Block start time", fontsize=11)
    ax.set_ylabel(f"Global units (n={n_units})", fontsize=11)
    ax.set_title("Unit Presence Heatmap", fontsize=13, fontweight="bold")
    ax.set_yticks([])

    plt.tight_layout()
    path = output_dir / "unit_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_unit_yield(blocks, global_units, presence, longevity, output_dir):
    """Stacked bar chart: matched vs new vs lost per block."""
    n_blocks = len(blocks)

    new_counts = []
    matched_counts = []
    lost_counts = []

    for bidx in range(n_blocks):
        current_units = set(np.where(presence[:, bidx] == 1)[0])

        if bidx == 0:
            new_counts.append(len(current_units))
            matched_counts.append(0)
            lost_counts.append(0)
        else:
            prev_units = set(np.where(presence[:, bidx - 1] == 1)[0])
            matched = current_units & prev_units
            new = current_units - prev_units
            lost = prev_units - current_units
            matched_counts.append(len(matched))
            new_counts.append(len(new))
            lost_counts.append(len(lost))

    x = np.arange(n_blocks)
    width = 0.6

    fig, ax = plt.subplots(figsize=(max(6, n_blocks * 2), 5))
    ax.bar(x, matched_counts, width, label="Matched", color=COLOR_MATCHED)
    ax.bar(x, new_counts, width, bottom=matched_counts, label="New", color=COLOR_NEW)
    ax.bar(x, [-c for c in lost_counts], width, label="Lost", color=COLOR_LOST)

    ax.set_xticks(x)
    ax.set_xticklabels(blocks["block_label"], fontsize=10)
    ax.set_xlabel("Block start time", fontsize=11)
    ax.set_ylabel("Unit count", fontsize=11)
    ax.set_title("Unit Yield Per Block", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.axhline(0, color="black", linewidth=0.5)

    # Annotate bars
    for i in range(n_blocks):
        total = matched_counts[i] + new_counts[i]
        ax.text(i, total + 0.5, str(total), ha="center", va="bottom", fontsize=10, fontweight="bold")
        if lost_counts[i] > 0:
            ax.text(i, -lost_counts[i] - 0.5, f"-{lost_counts[i]}", ha="center", va="top",
                    fontsize=9, color=COLOR_LOST)

    plt.tight_layout()
    path = output_dir / "unit_yield.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_longevity_histogram(longevity, n_blocks, output_dir):
    """Histogram: how many units survived N blocks."""
    fig, ax = plt.subplots(figsize=(6, 5))

    bins = np.arange(0.5, n_blocks + 1.5, 1)
    counts, _, bars = ax.hist(longevity, bins=bins, color=COLOR_MATCHED,
                               edgecolor="white", linewidth=1.5)

    # Color bars by longevity
    max_lon = n_blocks
    for bar, b in zip(bars, range(1, n_blocks + 1)):
        bar.set_facecolor(CMAP_LONGEVITY(b / max_lon))

    # Annotate counts
    for i, c in enumerate(counts):
        if c > 0:
            ax.text(i + 1, c + 0.3, str(int(c)), ha="center", va="bottom",
                    fontsize=11, fontweight="bold")

    ax.set_xticks(range(1, n_blocks + 1))
    ax.set_xlabel("Number of blocks tracked", fontsize=11)
    ax.set_ylabel("Number of global units", fontsize=11)
    ax.set_title("Unit Longevity Distribution", fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = output_dir / "longevity_histogram.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_spike_count_consistency(blocks, global_units, presence, longevity,
                                  spike_entries, output_dir):
    """Line plot: spike count per global unit across blocks.

    Only shows units tracked across 2+ blocks. Lines colored by longevity.
    """
    n_blocks = len(blocks)
    block_start_to_idx = dict(zip(blocks["block_start"], blocks["block_idx"]))
    max_longevity = longevity.max()

    # Aggregate spike counts per (global_unit, block)
    spike_per_block = spike_entries.groupby(
        ["global_unit", "block_start"]
    )["spike_count"].sum().reset_index()

    # Only units with longevity >= 2
    multi_block_units = [i for i, l in enumerate(longevity) if l >= 2]
    if not multi_block_units:
        print("  Skipped spike_count_consistency (no multi-block units)")
        return None

    fig, ax = plt.subplots(figsize=(max(6, n_blocks * 2), 6))

    for uid_idx in multi_block_units:
        gu = global_units[uid_idx]
        lon = longevity[uid_idx]
        color = CMAP_LONGEVITY(lon / max_longevity)

        unit_spikes = spike_per_block[spike_per_block["global_unit"] == gu]
        xs = [block_start_to_idx[bs] for bs in unit_spikes["block_start"]]
        ys = unit_spikes["spike_count"].values

        ax.plot(xs, ys, "o-", color=color, alpha=0.6, markersize=4, linewidth=1.2)

    ax.set_xticks(range(n_blocks))
    ax.set_xticklabels(blocks["block_label"], fontsize=10)
    ax.set_xlabel("Block start time", fontsize=11)
    ax.set_ylabel("Spike count (per block)", fontsize=11)
    ax.set_title("Spike Count Consistency Across Blocks", fontsize=13, fontweight="bold")
    ax.set_yscale("log")

    # Legend
    handles = []
    for lon_val in range(2, max_longevity + 1):
        handles.append(mpatches.Patch(
            color=CMAP_LONGEVITY(lon_val / max_longevity),
            label=f"{lon_val} blocks"
        ))
    ax.legend(handles=handles, title="Longevity", loc="upper right", fontsize=9)

    plt.tight_layout()
    path = output_dir / "spike_count_consistency.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def generate_html_report(output_dir, plot_paths, blocks, global_units, longevity):
    """Generate a simple HTML report bundling all plots."""
    n_blocks = len(blocks)
    n_units = len(global_units)
    max_lon = longevity.max()
    lon_counts = {i: int((longevity == i).sum()) for i in range(1, n_blocks + 1)}

    block_times = ", ".join(blocks["block_label"].tolist())

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Ephys v2 Unit Matching QC — {EXPERIMENT_NAME}</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; background: #fafafa; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 40px; }}
        .summary {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .summary table {{ border-collapse: collapse; width: 100%; }}
        .summary td, .summary th {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #ecf0f1; }}
        .summary th {{ color: #7f8c8d; font-weight: 600; width: 200px; }}
        img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); margin: 15px 0; }}
        .timestamp {{ color: #95a5a6; font-size: 0.85em; }}
    </style>
</head>
<body>
    <h1>Unit Matching QC Report</h1>
    <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

    <div class="summary">
    <table>
        <tr><th>Experiment</th><td>{EXPERIMENT_NAME}</td></tr>
        <tr><th>Subject</th><td>{SUBJECT}</td></tr>
        <tr><th>Insertion</th><td>{INSERTION_NUMBER}</td></tr>
        <tr><th>Blocks</th><td>{n_blocks} ({block_times})</td></tr>
        <tr><th>Global units</th><td>{n_units}</td></tr>
        <tr><th>Max longevity</th><td>{max_lon} blocks</td></tr>
        <tr><th>Longevity breakdown</th><td>{', '.join(f'{v} units × {k} blocks' for k, v in sorted(lon_counts.items(), reverse=True) if v > 0)}</td></tr>
    </table>
    </div>
"""

    plot_info = [
        ("Unit Tracking (Gantt)", "unit_gantt.png",
         "Each horizontal line is a global unit spanning the blocks it was detected in. "
         "Units sorted by longevity (most stable at top). Color = number of blocks tracked."),
        ("Unit Presence Heatmap", "unit_heatmap.png",
         "Binary presence matrix — blue = unit detected in block."),
        ("Unit Yield Per Block", "unit_yield.png",
         "How many units were matched from the previous block, how many are new, "
         "and how many were lost. First block is all new by definition."),
        ("Longevity Distribution", "longevity_histogram.png",
         "How many global units were tracked across 1, 2, 3, ... blocks."),
        ("Spike Count Consistency", "spike_count_consistency.png",
         "Total spike count per block for each multi-block unit. "
         "Large deviations may indicate suspect matching. Log scale."),
    ]

    for title, filename, description in plot_info:
        path = output_dir / filename
        if path.exists():
            html += f"""
    <h2>{title}</h2>
    <p>{description}</p>
    <img src="{filename}" alt="{title}">
"""

    html += """
</body>
</html>"""

    report_path = output_dir / "qc_report.html"
    report_path.write_text(html)
    print(f"  Saved: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Ephys v2 unit matching QC plots")
    parser.add_argument("--output-dir", type=str, default="./qc_output",
                        help="Directory to save plots and report")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching unit matching data for {EXPERIMENT_NAME}...")
    blocks, unit_entries, spike_entries = fetch_unit_matching_data()

    if unit_entries.empty:
        print("ERROR: No UnitMatching data found. Run the pipeline first.")
        sys.exit(1)

    n_blocks = len(blocks)
    global_units, presence, longevity = compute_unit_block_matrix(blocks, unit_entries)

    print(f"\n  {len(global_units)} global units across {n_blocks} blocks")
    for lon in range(n_blocks, 0, -1):
        count = (longevity == lon).sum()
        if count > 0:
            print(f"  {count} units tracked across {lon} block{'s' if lon > 1 else ''}")

    print(f"\nGenerating plots → {output_dir}/")
    plot_paths = []

    plot_paths.append(plot_unit_gantt(blocks, global_units, presence, longevity, output_dir))
    plot_paths.append(plot_unit_heatmap(blocks, global_units, presence, longevity, output_dir))
    plot_paths.append(plot_unit_yield(blocks, global_units, presence, longevity, output_dir))
    plot_paths.append(plot_longevity_histogram(longevity, n_blocks, output_dir))
    plot_paths.append(plot_spike_count_consistency(
        blocks, global_units, presence, longevity, spike_entries, output_dir))

    print("\nGenerating HTML report...")
    report_path = generate_html_report(output_dir, plot_paths, blocks, global_units, longevity)

    print(f"\nDone! View report at: {report_path}")
    print(f"To copy to your laptop: scp -r aeon-hpc:{output_dir.resolve()} .")


if __name__ == "__main__":
    main()
