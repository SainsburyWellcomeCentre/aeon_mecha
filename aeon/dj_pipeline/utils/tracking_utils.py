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
    - Identity correction: SLEAP-style majority vote at the end
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
    wide = df2.pivot(index=time_col, columns="identity_name", values=["x", "y"])
    times = wide.index.values
    T = len(times)
    x_raw = np.vstack([wide["x"][ids[0]].values, wide["x"][ids[1]].values])
    y_raw = np.vstack([wide["y"][ids[0]].values, wide["y"][ids[1]].values])

    # 4) Init cleaned arrays + swap tracking
    x_clean = np.full_like(x_raw, np.nan)
    y_clean = np.full_like(y_raw, np.nan)
    swapped_flags = np.zeros(T, dtype=bool)  # Track local swaps

    # 5) Find first complete frame
    valid = np.isfinite(x_raw).all(axis=0)
    first_i = np.argmax(valid)

    # Fill early frames with raw data
    if first_i > 0:
        x_clean[:, :first_i] = x_raw[:, :first_i]
        y_clean[:, :first_i] = y_raw[:, :first_i]

    # Start tracking from first complete frame
    last_x = x_raw[:, first_i].copy()
    last_y = y_raw[:, first_i].copy()
    x_clean[:, first_i] = last_x
    y_clean[:, first_i] = last_y

    # Vote counter for final identity assignment
    track_votes = np.zeros((2, 2), dtype=np.int64)

    # Count first frame if valid
    if valid[first_i]:
        track_votes[0, 0] += 1
        track_votes[1, 1] += 1

    # 6) Main loop: swap correction + voting
    for t in tqdm(range(first_i + 1, T), desc="Cleaning frames"):
        # Missing data: carry forward last known positions
        if not np.isfinite(x_raw[:, t]).all():
            for k in (0, 1):
                if np.isfinite(x_raw[k, t]):
                    x_clean[k, t] = last_x[k]
                    y_clean[k, t] = last_y[k]
            continue

        # Mice too close: keep original assignment
        inter_mouse_dist = np.hypot(
            x_raw[0, t] - x_raw[1, t], y_raw[0, t] - y_raw[1, t]
        )
        if inter_mouse_dist < 100:
            x_clean[:, t] = x_raw[:, t]
            y_clean[:, t] = y_raw[:, t]
            track_votes[0, 0] += 1
            track_votes[1, 1] += 1
            last_x = x_raw[:, t].copy()
            last_y = y_raw[:, t].copy()
            continue

        # Distance-based swap decision
        dx = x_raw[:, t][:, None] - last_x[None, :]
        dy = y_raw[:, t][:, None] - last_y[None, :]
        dist = np.hypot(dx, dy)
        d_same = dist[0, 0] + dist[1, 1]
        d_swap = dist[0, 1] + dist[1, 0]

        # Too far even with best assignment: carry forward
        if min(d_same, d_swap) > 90:
            x_clean[:, t] = last_x
            y_clean[:, t] = last_y
            continue

        # Assign based on shortest total distance
        if d_same <= d_swap:
            x_clean[:, t] = x_raw[:, t]
            y_clean[:, t] = y_raw[:, t]
            track_votes[0, 0] += 1
            track_votes[1, 1] += 1
        else:
            x_clean[:, t] = x_raw[::-1, t]
            y_clean[:, t] = y_raw[::-1, t]
            track_votes[0, 1] += 1
            track_votes[1, 0] += 1
            swapped_flags[t] = True  # Mark local swap

        # Update reference for next frame
        last_x = x_clean[:, t].copy()
        last_y = y_clean[:, t].copy()

    # 7) Global identity correction via majority vote
    need_swap = track_votes[0, 1] > track_votes[0, 0]
    if need_swap:
        x_clean = x_clean[::-1, :]
        y_clean = y_clean[::-1, :]
        # Flip swap flags too
        swapped_flags = swapped_flags[::-1]

    # 8) Build output dataframe
    cleaned = pd.DataFrame(
        {
            time_col: np.repeat(times, 2),
            "identity_name": np.tile(ids, T),
            "x": x_clean.ravel(order="F"),
            "y": y_clean.ravel(order="F"),
        }
    )

    # 9) Merge back with original data (drop old x,y)
    df2_noxy = df2.drop(columns=["x", "y"])
    result = (
        df2_noxy.merge(cleaned, on=[time_col, "identity_name"], how="left")
        .set_index(time_col)
        .sort_index()
    )

    # 10) Mark swapped frames in likelihood
    if "identity_likelihood" in result.columns:
        mask = result.index.isin(times[swapped_flags])
        result.loc[mask, "identity_likelihood"] = np.nan

    return result
