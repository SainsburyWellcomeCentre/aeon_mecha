"""Unit tests for arrow_io: column and metadata access on Bonsai ONIX arrow files."""

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _write_arrow(path, batch_sizes, clock_step=7):
    """Write an arrow file with Clock + one data column, batched as given."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    total = sum(batch_sizes)
    clock = (np.arange(total, dtype=np.uint64) + 1) * clock_step
    data = np.arange(total, dtype=np.uint16)
    schema = pa.schema([("AmplifierData0", pa.uint16()), ("Clock", pa.uint64())])
    off = 0
    with ipc.new_file(path, schema) as w:
        for n in batch_sizes:
            w.write_batch(
                pa.RecordBatch.from_arrays(
                    [pa.array(data[off : off + n]), pa.array(clock[off : off + n])], schema=schema
                )
            )
            off += n
    return clock, data


class TestArrowIO:
    """Column reads, exact sample counts, and cheap clock bounds."""

    def test_read_column_returns_full_column(self, tmp_path):
        from aeon.dj_pipeline.utils.arrow_io import read_arrow_column

        f = tmp_path / "t.arrow"
        clock, _ = _write_arrow(f, [500, 500, 137])
        assert np.array_equal(read_arrow_column(f, "Clock"), clock)

    def test_num_samples_counts_short_final_batch(self, tmp_path):
        from aeon.dj_pipeline.utils.arrow_io import arrow_num_samples

        f = tmp_path / "t.arrow"
        _write_arrow(f, [500, 500, 137])
        assert arrow_num_samples(f) == 1137

    def test_clock_bounds_match_the_full_column(self, tmp_path):
        from aeon.dj_pipeline.utils.arrow_io import read_arrow_clock_bounds

        f = tmp_path / "t.arrow"
        clock, _ = _write_arrow(f, [500, 500, 137])
        assert read_arrow_clock_bounds(f) == (int(clock[0]), int(clock[-1]))

    def test_clock_bounds_reads_only_two_batches(self, tmp_path, monkeypatch):
        """Bounds come from batch metadata -- a 155M-sample file must not be scanned."""
        import pyarrow.ipc as ipc

        from aeon.dj_pipeline.utils.arrow_io import read_arrow_clock_bounds

        f = tmp_path / "t.arrow"
        _write_arrow(f, [100] * 50)
        calls = []
        original = ipc.RecordBatchFileReader.get_batch
        monkeypatch.setattr(
            ipc.RecordBatchFileReader,
            "get_batch",
            lambda self, i: (calls.append(i), original(self, i))[1],
        )
        read_arrow_clock_bounds(f)
        assert len(calls) <= 2, f"read {len(calls)} batches, expected at most 2"

    def test_missing_column_raises_naming_available_columns(self, tmp_path):
        from aeon.dj_pipeline.utils.arrow_io import read_arrow_column

        f = tmp_path / "t.arrow"
        _write_arrow(f, [10])
        with pytest.raises(KeyError) as exc:
            read_arrow_column(f, "HubClock")
        assert "HubClock" in str(exc.value) and "Clock" in str(exc.value)

    def test_empty_file_raises(self, tmp_path):
        import pyarrow as pa
        import pyarrow.ipc as ipc

        from aeon.dj_pipeline.utils.arrow_io import read_arrow_clock_bounds

        f = tmp_path / "empty.arrow"
        with ipc.new_file(f, pa.schema([("Clock", pa.uint64())])):
            pass
        with pytest.raises(ValueError, match="no record batches"):
            read_arrow_clock_bounds(f)
