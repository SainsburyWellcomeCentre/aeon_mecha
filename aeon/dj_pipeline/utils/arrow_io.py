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


def _open(path: Path, column: str | None = None):
    """Return (memory_mapped_source, RecordBatchFileReader). Caller closes the source.

    With ``column``, the reader decompresses only that field. ``get_batch`` otherwise
    decompresses the whole batch, so pulling one clock out of a 386-column probe file
    costs all 384 amplifier channels too -- ~24x the time, measured.
    """
    import pyarrow as pa
    from pyarrow.ipc import IpcReadOptions, RecordBatchFileReader

    source = pa.memory_map(str(path), "r")
    try:
        reader = RecordBatchFileReader(source)
        if column is not None:
            names = reader.schema.names
            if column not in names:
                raise KeyError(
                    f"Column {column!r} not in {path}. Available: {list(names)[:8]}"
                    f"{'...' if len(names) > 8 else ''}"
                )
            # Reading the footer costs nothing; only batches are compressed.
            reader = RecordBatchFileReader(
                source, options=IpcReadOptions(included_fields=[names.index(column)])
            )
        return source, reader
    except Exception:
        # The caller's try/finally only starts once this returns, so a truncated or
        # non-arrow file would leak the map on every one of the hundreds of files
        # an epoch walk opens.
        source.close()
        raise


def open_for_row_counts(path: Path):
    """Open a reader that decompresses only the first field, for ``num_rows`` walks.

    An empty ``included_fields`` is read as no filter, so one column is the floor.
    """
    import pyarrow as pa
    from pyarrow.ipc import RecordBatchFileReader

    source = pa.memory_map(str(Path(path)), "r")
    try:
        first = RecordBatchFileReader(source).schema.names[0]
    except Exception:
        source.close()
        raise
    source.close()
    return _open(Path(path), first)


def read_arrow_column(path: Path, column: str) -> np.ndarray:
    """Return one whole column as a numpy array."""
    path = Path(path)
    source, reader = _open(path, column)
    try:
        batches = [reader.get_batch(i).column(column) for i in range(reader.num_record_batches)]
        if not batches:
            # Match the column's declared dtype: np.empty(0) is float64, and a float
            # clock flowing into extend_clock would silently change the timebase.
            return np.empty(0, dtype=reader.schema.field(column).type.to_pandas_dtype())
        import pyarrow as pa

        whole_column = pa.chunked_array(batches)
        # to_numpy upcasts an integer column with nulls to float64 + NaN, which on a
        # clock changes the timebase and loses precision on large uint64 ticks.
        if whole_column.null_count:
            raise ValueError(f"Column {column!r} in {path} has {whole_column.null_count} null value(s).")
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
    source, reader = _open(path, column)
    try:
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
