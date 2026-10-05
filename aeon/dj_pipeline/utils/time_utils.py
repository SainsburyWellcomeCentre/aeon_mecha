"""Time/datetime utilities for the DJ pipeline."""

import datetime

datetime_formats = [
    "%Y-%m-%dT%H%M%SZ",    # new format: 2026-05-16T102123
    "%Y-%m-%dT%H-%M-%S",  # old format: 2026-05-16T10-21-23
]

def parse_epoch_timestamp(name: str) -> datetime.datetime:
    """Parse an epoch directory name into a naive datetime.

    Handles both formats:

    - Old (hyphenated): ``2026-04-15T09-03-01``
    - New (compact ISO 8601): ``2026-04-15T090301Z``
    """
    date_str, time_str = name.split("T")
    return datetime.datetime.fromisoformat(
        date_str + "T" + time_str.replace("-", ":")
    ).replace(tzinfo=None)


def compute_chunk_time_model(clock_path, all_timestamps):
    """Compute the sync model for one chunk of ephys data.

    Output mirrors `HarpSyncAlignment` from aeon_api.
    """
    import numpy as np

    clock_binary = np.memmap(clock_path, dtype=np.int64, mode='r')
    n_samples = len(clock_binary)

    clock_start = clock_binary[0]
    clock_end = clock_binary[-1]

    # Find the sync info interval which covers the ephys chunk
    initial_ts_index = np.searchsorted(all_timestamps['Value.Clock'].values, clock_start, side='left')
    final_ts_index = np.searchsorted(all_timestamps['Value.Clock'].values, clock_end, side='right')

    # If clock times lie outside the range of sync info, we need to take the boundary indices
    initial_ts_index = max(0, initial_ts_index)
    final_ts_index = min(final_ts_index, len(all_timestamps) - 1)

    if final_ts_index <= initial_ts_index:
        return None

    slope, intercept = np.polyfit(
        all_timestamps.iloc[initial_ts_index:final_ts_index]['Value.Clock'].values,
        all_timestamps.iloc[initial_ts_index:final_ts_index]['Value.HarpTime'].values,
        deg=1,
    )
    harp_start = intercept + clock_start*slope
    harp_end = intercept + clock_end*slope

    model_info = {
        'clock_start': clock_start,
        'clock_end': clock_end,
        'harp_start': harp_start,
        'harp_end': harp_end,
        'n_samples': n_samples,
        'slope': slope,
        'intercept': intercept
    }

    return model_info
