"""Utilities for tracking data processing."""

import numpy as np
import pandas as pd
from tqdm import tqdm


def clean_swaps(df: pd.DataFrame, region_df: pd.DataFrame) -> pd.DataFrame:
    """ Swap correction for dual mouse tracking.

    Filters out-of-bounds points first,
    then does identity assignment with majority voting to fix track swaps.

    - Pre-cleaning: removes points outside arena/nest bounds
    - Early frames: filled with raw data before first complete observation
    - Swap detection: uses frame-to-frame distance minimization
    - Identity correction: SLEAP-style majority vote within continuous segments of cleaning
    - Swap marking: sets identity_likelihood=NaN on locally swapped frames

    Parameters:
    - df: DataFrame with tracking data (x, y, identity_name columns)
    - region_df: DataFrame containing arena region data with columns:
        - region_name: name of the region (e.g., 'ArenaCenter', 'ArenaOuterRadius', 'NestRegion')
        - region_data: dictionary containing region-specific data

    Returns:
    - DataFrame with cleaned tracking data, same structure as input but corrected x,y coords
      and identity_likelihood=NaN on locally swapped frames

    Example:
        To retrieve region_df from the database:
        ```python
        active_region_query = acquisition.EpochConfig.ActiveRegion & (acquisition.Chunk & chunk_key)
        region_df = active_region_query.fetch(format="frame")
        ```
    """

    # Helper to extract region values
    def get_region_value(region_name):
        mask = region_df.index.get_level_values("region_name") == region_name
        if mask.any():
            return region_df.loc[mask, "region_data"].iloc[0]
        return None

    # Parse arena geometry
    arena_center = get_region_value("ArenaCenter")
    arena_outer_radius = get_region_value("ArenaOuterRadius")
    nest_region = get_region_value("NestRegion")

    if arena_center is None or arena_outer_radius is None:
        raise ValueError(
            "Could not find ArenaCenter or ArenaOuterRadius in region data"
        )

    # Coords + radius
    center_x = float(arena_center["X"])
    center_y = float(arena_center["Y"])
    outer_radius = float(arena_outer_radius) + 10

    # Nest boundary points
    nest_x_coords = []
    nest_y_coords = []
    if nest_region and "ArrayOfPoint" in nest_region:
        for point in nest_region["ArrayOfPoint"]:
            nest_x_coords.append(float(point["X"]))
            nest_y_coords.append(float(point["Y"]))

    # Filter out-of-bounds points before cleaning
    if len(nest_x_coords) > 0:
        # Arena circle check
        dist2 = (df["x"] - center_x) ** 2 + (df["y"] - center_y) ** 2
        inside_arena = dist2 <= outer_radius**2

        # Nest bounding box with padding
        nx_min, nx_max = min(nest_x_coords) - 10, max(nest_x_coords) + 10
        ny_min, ny_max = min(nest_y_coords) - 10, max(nest_y_coords) + 10
        inside_nest = df["x"].between(nx_min, nx_max) & df["y"].between(ny_min, ny_max)

        # Keep arena OR nest points
        df = df[inside_arena | inside_nest]
    else:
        # Arena only if no nest
        dist2 = (df["x"] - center_x) ** 2 + (df["y"] - center_y) ** 2
        inside_arena = dist2 <= outer_radius**2
        df = df[inside_arena]

    # Swap correction starts here
    # 1) setup data for processing
    df = df.sort_index()
    ids = df["identity_name"].unique()
    if len(ids) != 2:
        raise ValueError(
            "Expected exactly two identities, found: {}".format(ids)
        )

    # 2) Prep for merge later
    df2 = df.reset_index()
    time_col = df2.columns[0]  # timestamp column name

    # 3) Reshape to 2×T arrays
    wide = df2.pivot(index=time_col, columns='identity_name', values=['x', 'y'])
    times = wide.index.values
    T = len(times)
    x_raw = np.vstack([
        wide['x'][ids[0]].values,
        wide['x'][ids[1]].values
    ])
    y_raw = np.vstack([
        wide['y'][ids[0]].values,
        wide['y'][ids[1]].values
    ])

    # 4) Init cleaned arrays + swap tracking
    x_clean = np.full_like(x_raw, np.nan)
    y_clean = np.full_like(y_raw, np.nan)
    swapped_flags = np.zeros(T, dtype=bool)

    # 5) Find first complete frame
    valid = np.isfinite(x_raw).all(axis=0)
    if not valid.any():
        raise RuntimeError("No frame with both subjects present")
    first_i = np.argmax(valid)

    # Helper to flush & apply local vote
    def _flush_segment(start, end, votes_same, votes_swap):
        if votes_swap > votes_same:
            x_clean[:, start:end] = x_clean[::-1, start:end]
            y_clean[:, start:end] = y_clean[::-1, start:end]
            swapped_flags[start:end] = ~swapped_flags[start:end]

    # 6) Local-segment tracking
    seg_start = first_i
    votes_same = 0
    votes_swap = 0

    # Initialize on first full-detect frame
    last_x = x_raw[:, first_i].copy()
    last_y = y_raw[:, first_i].copy()
    x_clean[:, first_i] = last_x
    y_clean[:, first_i] = last_y
    votes_same += 1

    for t in tqdm(range(first_i + 1, T), desc="Cleaning frames"):
        present = np.isfinite(x_raw[:, t])
        n_det = present.sum()

        # Precompute for two detections
        if n_det == 2:
            inter_d = np.hypot(
                x_raw[0, t] - x_raw[1, t],
                y_raw[0, t] - y_raw[1, t]
            )
            dx = x_raw[:, t][:, None] - last_x[None, :]
            dy = y_raw[:, t][:, None] - last_y[None, :]
            dist = np.hypot(dx, dy)
            d_same = dist[0, 0] + dist[1, 1]
            d_swap = dist[0, 1] + dist[1, 0]
            d0 = dist[0].min()
            d1 = dist[1].min()

        # Detect continuity break: blindly trusting raw or frames dropped
        break_cond = (
            n_det == 0 or
            (n_det == 2 and inter_d < 100) or
            (n_det == 2 and min(d_same, d_swap) > 90 and min(d0, d1) > 90)
        )
        if break_cond:
            _flush_segment(seg_start, t, votes_same, votes_swap)
            seg_start = t
            votes_same = votes_swap = 0

        # 6a) zero detections → drop both
        if n_det == 0:
            continue

        # 6b) one detection → assign to closest ≤90px
        if n_det == 1:
            i = np.where(present)[0][0]
            d0_ = np.hypot(
                x_raw[i, t] - last_x[0],
                y_raw[i, t] - last_y[0]
            )
            d1_ = np.hypot(
                x_raw[i, t] - last_x[1],
                y_raw[i, t] - last_y[1]
            )
            if min(d0_, d1_) <= 90:
                j = 0 if d0_ <= d1_ else 1
                x_clean[j, t] = x_raw[i, t]
                y_clean[j, t] = y_raw[i, t]
                votes_same += 1
                last_x[j], last_y[j] = x_raw[i, t], y_raw[i, t]
            continue

        # 6c) two detections too close → trust raw
        if inter_d < 100:
            x_clean[:, t] = x_raw[:, t]
            y_clean[:, t] = y_raw[:, t]
            votes_same += 1
            last_x[:] = x_raw[:, t]
            last_y[:] = y_raw[:, t]
            continue

        # 6d) both assignments >90px → keep only the closer detection (or drop)
        if min(d_same, d_swap) > 90:
            if min(d0, d1) > 90:
                continue
            i_k = 0 if d0 <= d1 else 1
            j = int(np.argmin(dist[i_k]))
            x_clean[j, t] = x_raw[i_k, t]
            y_clean[j, t] = y_raw[i_k, t]
            votes_same += 1
            last_x[j], last_y[j] = x_raw[i_k, t], y_raw[i_k, t]
            continue

        # 6e) normal same-vs-swap assignment
        if d_same <= d_swap:
            x_clean[:, t] = x_raw[:, t]
            y_clean[:, t] = y_raw[:, t]
            votes_same += 1
        else:
            x_clean[:, t] = x_raw[::-1, t]
            y_clean[:, t] = y_raw[::-1, t]
            votes_swap += 1
            swapped_flags[t] = True

        last_x[:] = x_clean[:, t]
        last_y[:] = y_clean[:, t]

    # 7) Flush the final segment
    _flush_segment(seg_start, T, votes_same, votes_swap)

    # 8) Build cleaned DataFrame
    cleaned = pd.DataFrame({
        time_col: np.repeat(times, 2),
        'identity_name': np.tile(ids, T),
        'x': x_clean.ravel(order='F'),
        'y': y_clean.ravel(order='F'),
    })

    # 9) Merge back with original data (drop old x, y)
    df2_noxy = df2.drop(columns=['x', 'y'])
    result = (
        df2_noxy
        .merge(cleaned, on=[time_col, 'identity_name'], how='left')
        .set_index(time_col)
        .sort_index()
    )

    # 10) Mark swapped frames in likelihood
    if 'identity_likelihood' in result.columns:
        mask = result.index.isin(times[swapped_flags])
        result.loc[mask, 'identity_likelihood'] = np.nan

    # 11) Final cleanup: drop any rows where x or y is NaN
    result = result.dropna(subset=['x', 'y'], how='any')

    return result
