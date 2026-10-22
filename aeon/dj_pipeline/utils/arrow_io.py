"""Column-oriented access to Bonsai ONIX Arrow IPC files.

``read_bonsai_onix_arrow`` (spike_sorting_utils) is optimised for time-slicing a
384-channel recording. These helpers do the opposite: whole-column reads and
metadata-only lookups, for the clock streams that share the probe's file.
"""

from pathlib import Path

import numpy as np


def _open(path: Path):
    """Return (memory_mapped_source, RecordBatchFileReader). Caller closes the source."""
    import pyarrow as pa
    from pyarrow.ipc import RecordBatchFileReader

    source = pa.memory_map(str(path), "r")
    return source, RecordBatchFileReader(source)


def _require_column(reader, column: str, path: Path) -> None:
    if column not in reader.schema.names:
        raise KeyError(
            f"Column {column!r} not in {path}. Available: {list(reader.schema.names)[:8]}"
            f"{'...' if len(reader.schema.names) > 8 else ''}"
        )


def read_arrow_column(path: Path, column: str) -> np.ndarray:
    """Return one whole column as a numpy array."""
    path = Path(path)
    source, reader = _open(path)
    try:
        _require_column(reader, column, path)
        batches = [reader.get_batch(i).column(column) for i in range(reader.num_record_batches)]
        if not batches:
            # Match the column's declared dtype: np.empty(0) is float64, and a float
            # clock flowing into extend_clock would silently change the timebase.
            return np.empty(0, dtype=reader.schema.field(column).type.to_pandas_dtype())
        import pyarrow as pa

        return pa.chunked_array(batches).to_numpy(zero_copy_only=False)
    finally:
        source.close()


def arrow_num_samples(path: Path) -> int:
    """Total rows, summed from batch metadata.

    The final batch of a Bonsai file is short whenever acquisition stopped
    mid-batch, so this cannot be derived from batch count x batch size.

    Costs ~6 s on a 6.75 GB probe file: pyarrow exposes no per-batch row count
    without deserialising the batch. That is paid deliberately -- a sample count
    inferred from a uniform-batch assumption would shift every spike timestamp
    on a file that does not meet it.
    """
    path = Path(path)
    source, reader = _open(path)
    try:
        return sum(reader.get_batch(i).num_rows for i in range(reader.num_record_batches))
    finally:
        source.close()


def read_arrow_clock_bounds(path: Path, column: str = "Clock") -> tuple[int, int]:
    """Return (first, last) value of ``column`` without scanning the file.

    Only the first and last record batches are touched, so this stays cheap on
    the 6.75 GB probe files.
    """
    path = Path(path)
    source, reader = _open(path)
    try:
        _require_column(reader, column, path)
        n = reader.num_record_batches
        if n == 0:
            raise ValueError(f"{path} has no record batches")
        first = reader.get_batch(0).column(column)
        last = first if n == 1 else reader.get_batch(n - 1).column(column)
        if len(first) == 0 or len(last) == 0:
            raise ValueError(f"{path} has no record batches with rows")
        return int(first[0].as_py()), int(last[-1].as_py())
    finally:
        source.close()
