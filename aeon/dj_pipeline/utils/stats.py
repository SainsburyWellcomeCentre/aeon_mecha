"""Pure numpy/pandas stat helpers with no datajoint dependency.

These helpers compute JSON summary stats (min, max, mean, dtype, count)
for stream timestamps and arbitrary data columns.
The resulting fields provide a quick overview of the data represented by
the `stream_df` column, which uses the <aeon_stream> codec for lazy
DataFrame loading.
"""

import numpy as np


def column_stats(values):
    """Compute JSON-serializable summary stats for a numpy array."""
    stats = {"dtype": str(values.dtype), "count": int(len(values))}
    if len(values) > 0 and np.issubdtype(values.dtype, np.number):
        finite = values[np.isfinite(values)]
        if len(finite) > 0:
            stats["min"] = float(np.min(finite))
            stats["max"] = float(np.max(finite))
            stats["mean"] = round(float(np.mean(finite)), 4)
    return stats


def timestamp_stats(index):
    """Compute JSON-serializable summary stats for a datetime index."""
    if len(index) == 0:
        return {"count": 0}
    stats = {
        "min": str(index.min()),
        "max": str(index.max()),
        "count": int(len(index)),
    }
    if len(index) > 1:
        diffs = np.diff(index.values) / np.timedelta64(1, "ns")
        median_diff_ns = float(np.median(diffs))
        stats["sampling_rate_hz"] = round(1e9 / median_diff_ns, 2) if median_diff_ns > 0 else None
    return stats
