import numpy as np
import pandas as pd
from tqdm import tqdm


def clean_swaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fast swap‐correction that returns the original df's columns,
    but with x,y replaced by the cleaned tracks. Includes identity correction
    based on SLEAP's majority vote *inside* the main loop, so we only loop once over T.

    Steps:
      0) sort & get identities
      1) reset_index so we can merge back later
      2) pivot x,y into 2×T arrays
      3) prepare cleaned arrays
      4) find first fully‐observed column and initialize
      5) loop over t=first_i+1..T-1:
            • swap‐correction logic (same as before)
            • **immediately** update track_votes whenever x_raw[:,t] & x_clean[:,t] are both finite
      6) decide final swap based on accumulated votes
      7) rebuild cleaned‐coord DataFrame & merge back
    """

    # 0) sort & get identities
    df = df.sort_index()
    ids = df['identity_name'].unique()
    assert len(ids) == 2, "Need exactly two identities"

    # 1) reset_index so we can merge back later
    df2 = df.reset_index()
    time_col = df2.columns[0]  # timestamp column name

    # 2) pivot x,y into 2×T arrays
    wide = df2.pivot(index=time_col, columns='identity_name', values=['x','y'])
    times = wide.index.values
    T = len(times)
    x_raw = np.vstack([wide['x'][ids[0]].values,
                       wide['x'][ids[1]].values])
    y_raw = np.vstack([wide['y'][ids[0]].values,
                       wide['y'][ids[1]].values])

    # 3) prepare cleaned arrays
    x_clean = np.full_like(x_raw, np.nan)
    y_clean = np.full_like(y_raw, np.nan)

    # 4) find first fully‐observed column
    valid = np.isfinite(x_raw).all(axis=0)
    first_i = np.argmax(valid)
    last_x = x_raw[:, first_i].copy()
    last_y = y_raw[:, first_i].copy()
    x_clean[:, first_i] = last_x
    y_clean[:, first_i] = last_y

    # Initialize vote‐matrix: [cleaned_track, original_identity]
    track_votes = np.zeros((2, 2), dtype=np.int64)

    # If the very first frame is fully observed, count those votes now:
    if valid[first_i]:
        # At t=first_i, x_clean[:,first_i] == x_raw[:,first_i], so track 0 ← orig 0, track 1 ← orig 1
        track_votes[0, 0] += 1
        track_votes[1, 1] += 1

    # 5) loop and swap‐correct *and* vote in one pass
    for t in tqdm(range(first_i + 1, T), desc="Cleaning frames"):
        # 5a) if not all raw‐points are finite, just carry‐forward
        if not np.isfinite(x_raw[:, t]).all():
            for k in (0, 1):
                if np.isfinite(x_raw[k, t]):
                    x_clean[k, t] = last_x[k]
                    y_clean[k, t] = last_y[k]
            # No "vote" in incomplete frames
            continue

        # 5b) if the two mice are very close, keep original order
        inter_mouse_dist = np.hypot(x_raw[0, t] - x_raw[1, t],
                                    y_raw[0, t] - y_raw[1, t])
        if inter_mouse_dist < 100:
            x_clean[:, t] = x_raw[:, t]
            y_clean[:, t] = y_raw[:, t]
            # immediate vote: clean track 0 came from raw‐0, clean 1 from raw‐1
            track_votes[0, 0] += 1
            track_votes[1, 1] += 1
            last_x = x_raw[:, t].copy()
            last_y = y_raw[:, t].copy()
            continue

        # 5c) compute distances to previous frame to decide swap
        dx = x_raw[:, t][:, None] - last_x[None, :]
        dy = y_raw[:, t][:, None] - last_y[None, :]
        dist = np.hypot(dx, dy)
        d_same = dist[0, 0] + dist[1, 1]
        d_swap = dist[0, 1] + dist[1, 0]

        # 5d) if even the best assignment is too big, carry‐forward
        if min(d_same, d_swap) > 90:
            x_clean[:, t] = last_x
            y_clean[:, t] = last_y
            # No vote, because x_clean did not actually come from x_raw this frame
            continue

        # 5e) pick same vs swapped assignment
        if d_same <= d_swap:
            x_clean[:, t] = x_raw[:, t]
            y_clean[:, t] = y_raw[:, t]
            # vote: clean₀ came from raw₀, clean₁ from raw₁
            track_votes[0, 0] += 1
            track_votes[1, 1] += 1
        else:
            x_clean[:, t] = x_raw[::-1, t]
            y_clean[:, t] = y_raw[::-1, t]
            # vote: clean₀ came from raw₁, clean₁ from raw₀
            track_votes[0, 1] += 1
            track_votes[1, 0] += 1

        # 5f) update last_x/last_y
        last_x = x_clean[:, t].copy()
        last_y = y_clean[:, t].copy()

    # 6) Final identity swap based on majority vote
    need_swap = track_votes[0, 1] > track_votes[0, 0]
    if need_swap:
        print(f"Swapping final tracks based on SLEAP majority vote")
        print(f"Track votes:\n{track_votes}")
        x_clean = x_clean[::-1, :]
        y_clean = y_clean[::-1, :]

    # 7) build cleaned coord table, naming columns x,y
    cleaned = pd.DataFrame({
        time_col: np.repeat(times, 2),
        'identity_name': np.tile(ids, T),
        'x': x_clean.ravel(order='F'),
        'y': y_clean.ravel(order='F'),
    })

    # 8) drop the old x,y and merge back everything else
    df2_noxy = df2.drop(columns=['x', 'y'])
    result = (
        df2_noxy
        .merge(cleaned, on=[time_col, 'identity_name'], how='right')
        .set_index(time_col)
        .sort_index()
    )

    return result
