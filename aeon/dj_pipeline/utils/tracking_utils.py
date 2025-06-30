"""Utilities for tracking data processing."""

import numpy as np
import pandas as pd

NUM_IDS = 2  # Number of identities expected in the tracking data
MAX_DIST_BETWEEN_FRAMES = 90  # Maximum distance between frames
MIN_INTER_SUBJ_DIST = 100  # Subjects must be at least this far apart
MAX_SWAP_COST = 100  # Maximum cost distance to consider a swap


def filter_valid_points(df: pd.DataFrame, region_df: pd.DataFrame) -> pd.DataFrame:
    """Filter points in the DataFrame that are within the arena or nest region.

    Args:
        df (pandas.DataFrame): DataFrame containing tracking data with columns 'x' and 'y'.
        region_df (pandas.DataFrame): DataFrame containing arena region data with columns:
            - region_name: name of the region (e.g., 'ArenaCenter', 'ArenaOuterRadius', 'NestRegion')
            - region_data: dictionary containing region-specific data

    Returns:
        pandas.DataFrame: DataFrame with only valid (in-arena or in-nest) points

    """

    def get_region_value(region_name):
        mask = region_df.index.get_level_values("region_name") == region_name
        if mask.any():
            return region_df.loc[mask, "region_data"].iloc[0]  # type: ignore
        return None

    # --- Extract arena and nest geometry ---
    arena_center = get_region_value("ArenaCenter")
    arena_outer_radius = get_region_value("ArenaOuterRadius")
    nest_region = get_region_value("NestRegion")

    if arena_center is None or arena_outer_radius is None:
        raise ValueError("Could not find ArenaCenter or ArenaOuterRadius in region data")

    center_x = float(arena_center["X"])
    center_y = float(arena_center["Y"])
    outer_radius = float(arena_outer_radius) + 10  # extra padding

    coords = df[["x", "y"]].to_numpy()
    dx = coords[:, 0] - center_x
    dy = coords[:, 1] - center_y
    dist2 = dx**2 + dy**2
    inside_arena = dist2 <= outer_radius**2

    inside_nest = np.zeros(len(df), dtype=bool)
    if nest_region and "ArrayOfPoint" in nest_region:
        nest_coords = np.array([(float(p["X"]), float(p["Y"])) for p in nest_region["ArrayOfPoint"]])
        if len(nest_coords) > 0:
            x_min, x_max = nest_coords[:, 0].min() - 10, nest_coords[:, 0].max() + 10
            y_min, y_max = nest_coords[:, 1].min() - 10, nest_coords[:, 1].max() + 10
            inside_nest = (
                (coords[:, 0] >= x_min)
                & (coords[:, 0] <= x_max)
                & (coords[:, 1] >= y_min)
                & (coords[:, 1] <= y_max)
            )

    return df.loc[inside_arena | inside_nest]


def clean_swaps(df: pd.DataFrame, region_df: pd.DataFrame) -> pd.DataFrame:
    """Swap correction for dual mouse tracking.

    Filters out-of-bounds points first,
    then does identity assignment with majority voting to fix track swaps.

    - Pre-cleaning: removes points outside arena/nest bounds
    - Early frames: filled with raw data before first complete observation
    - Swap detection: uses frame-to-frame distance minimization
    - Identity correction: SLEAP-style majority vote within continuous segments of cleaning
    - Swap marking: sets identity_likelihood=NaN on locally swapped frames

    Args:
        df (pandas.DataFrame): DataFrame containing tracking data with columns 'x' and 'y'.
        region_df (pandas.DataFrame): DataFrame containing arena region data with columns:
            - region_name: name of the region (e.g., 'ArenaCenter', 'ArenaOuterRadius', 'NestRegion')
            - region_data: dictionary containing region-specific data

    Returns:
        pandas.DataFrame: DataFrame with cleaned tracking data having the same structure as
            the input but corrected x,y coords and identity_likelihood=NaN on locally swapped frames

    Examples:
        To retrieve region_df from the database:

        >>> active_region_query = acquisition.EpochConfig.ActiveRegion & (acquisition.Chunk & chunk_key)
        >>> region_df = active_region_query.fetch(format="frame")

    """
    # Select only points within the arena or nest region
    df = filter_valid_points(df, region_df)

    # Swap correction
    # 1) setup data for processing
    df = df.sort_index()
    ids = sorted(df["identity_name"].unique())  # enforce order
    if len(ids) != NUM_IDS:
        raise ValueError(f"Expected exactly two identities, found: {ids}")

    # 2) Prep for merge later
    df2 = df.reset_index()
    time_col = df2.columns[0]  # timestamp column name

    # 3) Reshape to 2×T arrays
    wide = df2.pivot(index=time_col, columns="identity_name", values=["x", "y"])
    times = wide.index.values
    T = len(times)
    x_raw = wide["x"][ids].to_numpy().T
    y_raw = wide["y"][ids].to_numpy().T

    # 4) Init cleaned arrays + swap tracking
    x_clean = np.full_like(x_raw, np.nan)
    y_clean = np.full_like(y_raw, np.nan)
    swapped_flags = np.zeros(T, dtype=bool)

    # 5) Find first complete frame and keep raw data before it
    valid = np.isfinite(x_raw).all(axis=0)
    if not valid.any():
        raise RuntimeError("No frame with both subjects present")
    first_i = np.argmax(valid)
    if first_i > 0:
        x_clean[:, :first_i] = x_raw[:, :first_i]
        y_clean[:, :first_i] = y_raw[:, :first_i]

    # 6) Local-segment tracking
    # Initialize on first full-detect frame
    seg_start = first_i
    votes_same = 1
    votes_swap = 0
    last_x = x_raw[:, first_i].copy()
    last_y = y_raw[:, first_i].copy()
    x_clean[:, first_i] = last_x
    y_clean[:, first_i] = last_y

    # --- Helper functions ---
    def _flush_segment(start, end, votes_same, votes_swap):
        """Flush the current segment and apply local vote."""
        if votes_swap > votes_same:
            x_clean[:, start:end] = x_clean[::-1, start:end]
            y_clean[:, start:end] = y_clean[::-1, start:end]
            swapped_flags[start:end] = ~swapped_flags[start:end]

    def _assign_single_detection(t, src_idx, dest_idx):
        """Assign values from src_idx to dest_idx at time t."""
        x_clean[dest_idx, t] = x_raw[src_idx, t]
        y_clean[dest_idx, t] = y_raw[src_idx, t]
        last_x[dest_idx] = x_raw[src_idx, t]
        last_y[dest_idx] = y_raw[src_idx, t]

    def _assign_full_frame(t, x_vals, y_vals):
        """Assign full frame values to both identities at time t."""
        x_clean[:, t] = x_vals
        y_clean[:, t] = y_vals
        last_x[:] = x_vals
        last_y[:] = y_vals

    def _update_votes(t):
        """Determine if a swap occurred at time t and update vote counts accordingly."""
        if np.allclose(x_raw[:, t], x_clean[:, t], equal_nan=True) and np.allclose(
            y_raw[:, t], y_clean[:, t], equal_nan=True
        ):
            nonlocal votes_same
            votes_same += 1
        else:
            nonlocal votes_swap
            votes_swap += 1

    for t in range(first_i + 1, T):
        present = np.isfinite(x_raw[:, t])
        n_det = present.sum()

        # zero detections → drop both
        if n_det == 0:
            _flush_segment(seg_start, t, votes_same, votes_swap)
            seg_start = t
            votes_same = votes_swap = 0
            continue

        # 1 detection → assign to closest
        if n_det == 1:
            src_idx = np.where(present)[0][0]
            dist_to_0 = np.hypot(x_raw[src_idx, t] - last_x[0], y_raw[src_idx, t] - last_y[0])
            dist_to_1 = np.hypot(x_raw[src_idx, t] - last_x[1], y_raw[src_idx, t] - last_y[1])
            if min(dist_to_0, dist_to_1) <= MAX_DIST_BETWEEN_FRAMES:
                dest_idx = 0 if dist_to_0 <= dist_to_1 else 1
                _assign_single_detection(t, src_idx, dest_idx)
                _update_votes(t)
            continue

        # 2 detections
        # compute distances and assignment costs
        inter_d = np.hypot(x_raw[0, t] - x_raw[1, t], y_raw[0, t] - y_raw[1, t])
        dx = x_raw[:, t][:, None] - last_x[None, :]
        dy = y_raw[:, t][:, None] - last_y[None, :]
        dist_mat = np.hypot(dx, dy)
        cost_same = dist_mat[0, 0] + dist_mat[1, 1]
        cost_swap = dist_mat[0, 1] + dist_mat[1, 0]
        min_dist_id0 = dist_mat[0].min()
        min_dist_id1 = dist_mat[1].min()

        # Define discontinuity conditions
        break_too_close = inter_d < MIN_INTER_SUBJ_DIST  # two detections are too close
        break_too_costly = min(cost_same, cost_swap) > MAX_SWAP_COST
        break_both_far = min(min_dist_id0, min_dist_id1) > MAX_DIST_BETWEEN_FRAMES  # both assignments >90px

        if break_too_close or (break_too_costly and break_both_far):
            _flush_segment(seg_start, t, votes_same, votes_swap)
            seg_start = t
            votes_same = votes_swap = 0

        # Re-evaluate after reset
        if break_too_close:  # keep original assignment
            _assign_full_frame(t, x_raw[:, t], y_raw[:, t])
            votes_same += 1
            continue

        # Re-evaluate after reset
        if break_too_costly:
            # If both assignments are too far, skip
            if break_both_far:
                continue
            dest_idx = 0 if min_dist_id0 <= min_dist_id1 else 1
            src_idx = int(np.argmin(dist_mat[dest_idx]))
            _assign_single_detection(t, src_idx, dest_idx)
            _update_votes(t)
            continue

        if cost_same <= cost_swap:
            x_vals = x_raw[:, t]
            y_vals = y_raw[:, t]
            votes_same += 1
        else:
            x_vals = x_raw[::-1, t]
            y_vals = y_raw[::-1, t]
            swapped_flags[t] = True
            votes_swap += 1
        _assign_full_frame(t, x_vals, y_vals)

    # 7) Flush the final segment
    _flush_segment(seg_start, T, votes_same, votes_swap)

    # 8) Build cleaned DataFrame
    cleaned = pd.DataFrame(
        {
            time_col: np.repeat(times, 2),
            "identity_name": np.tile(ids, T),
            "x": x_clean.ravel(order="F"),
            "y": y_clean.ravel(order="F"),
        }
    )

    # 9) Merge back with original data (drop old x, y)
    df2_noxy = df2.drop(columns=["x", "y"])
    result = (
        df2_noxy.merge(cleaned, on=[time_col, "identity_name"], how="right")
        .set_index(time_col)
        .sort_index()
    )

    # 10) Mark swapped frames in likelihood
    if "identity_likelihood" in result.columns:
        mask = result.index.isin(times[swapped_flags])
        result.loc[mask, "identity_likelihood"] = np.nan

    # 11) Final cleanup: drop any rows where x or y is NaN
    result = result.dropna(subset=["x", "y"], how="any")

    return result
