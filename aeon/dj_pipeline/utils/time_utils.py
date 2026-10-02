"""Time/datetime utilities for the DJ pipeline."""

import datetime


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
    """
    Compute's the sync model for one chunk of ephys data; 
    output matches `HarpSyncAlignment` from aeon_api.
    """
    import numpy as np

    clock_binary = np.memmap(clock_path, dtype=np.int64, mode='r')
    n_samples = len(clock_binary)

    clock_start = clock_binary[0]
    clock_end = clock_binary[-1]

    first_ts_index = np.searchsorted(all_timestamps['Value.Clock'].values, clock_start, side='left')
    final_ts_index = np.searchsorted(all_timestamps['Value.Clock'].values, clock_end, side='right')

    initial_ephy_second, final_ephys_second = all_timestamps['Value.HarpTime'].iloc[[first_ts_index, final_ts_index]]

    initial_index_for_interpolation = np.searchsorted(all_timestamps['Value.HarpTime'].values, initial_ephy_second)
    final_index_for_interpolation = np.searchsorted(all_timestamps['Value.HarpTime'].values, final_ephys_second) + 1

    slope, intercept = np.polyfit(
        all_timestamps.iloc[initial_index_for_interpolation:final_index_for_interpolation]['Value.Clock'].values,
        all_timestamps.iloc[initial_index_for_interpolation:final_index_for_interpolation]['Value.HarpTime'].values,
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
