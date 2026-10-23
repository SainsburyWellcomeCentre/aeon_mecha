"""Column-oriented access to Bonsai ONIX Arrow IPC files.

``read_bonsai_onix_arrow`` (spike_sorting_utils) is optimised for time-slicing a
384-channel recording. These helpers do the opposite: a whole-column read and a
cheap bounds lookup, for the clock streams that share the probe's file.
"""

from pathlib import Path

import numpy as np


class EmptyArrowChunk(ValueError):
    """An arrow file that is well formed but holds no rows.

    Distinct from a malformed file: callers that can legitimately skip an empty
    chunk match on this type rather than on the wording of a message.
    """


def _open(path: Path):
    """Return (memory_mapped_source, RecordBatchFileReader). Caller closes the source."""
    import pyarrow as pa
    from pyarrow.ipc import RecordBatchFileReader

    source = pa.memory_map(str(path), "r")
    try:
        return source, RecordBatchFileReader(source)
    except Exception:
        # The caller's try/finally only starts once this returns, so a truncated or
        # non-arrow file would leak the map on every one of the hundreds of files
        # an epoch walk opens.
        source.close()
        raise


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

        whole_column = pa.chunked_array(batches)
        if whole_column.null_count:
            # to_numpy upcasts an integer column with nulls to float64 + NaN, so a
            # single null clock would silently change the timebase and lose precision
            # on large uint64 ticks -- the same hazard the empty branch above guards.
            raise ValueError(
                f"Column {column!r} in {path} has {whole_column.null_count} null value(s). "
                f"Converting would upcast it to float64 and change the timebase."
            )
        return whole_column.to_numpy(zero_copy_only=False)
    finally:
        source.close()


def _first_nonempty(reader, column: str, order):
    """First batch column in ``order`` that has rows, or None."""
    for i in order:
        col = reader.get_batch(i).column(column)
        if len(col):
            return col
    return None


def read_arrow_clock_bounds(path: Path, column: str = "Clock") -> tuple[int, int]:
    """Return (first, last) value of ``column`` without scanning the file.

    Only the first and last record batches are touched, so this stays cheap on
    the 6.75 GB probe files.
    """
    path = Path(path)
    source, reader = _open(path)
    try:
        _require_column(reader, column, path)
        n_batches = reader.num_record_batches
        if n_batches == 0:
            raise EmptyArrowChunk(f"{path} has no record batches")
        # A writer that flushes at stop can leave an empty batch at either end, so
        # walk past those rather than reporting a file with data as having none.
        # In the common case this touches exactly the first and last batch.
        first = _first_nonempty(reader, column, range(n_batches))
        last = _first_nonempty(reader, column, range(n_batches - 1, -1, -1))
        if first is None or last is None:
            raise EmptyArrowChunk(f"{path} has record batches but none with rows")
        return int(first[0].as_py()), int(last[-1].as_py())
    finally:
        source.close()
