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


class TestReadAmplifierRecordingArrow:
    """read_amplifier_recording dispatches on suffix; arrow bypasses resolve_ephys_file."""

    def _arrow_probe_file(self, tmp_path, batch_sizes, n_channels=4):
        import pyarrow as pa
        import pyarrow.ipc as ipc

        total = sum(batch_sizes)
        cols = {f"AmplifierData{i}": np.full(total, i, dtype=np.uint16) for i in range(n_channels)}
        cols["HubClock"] = np.arange(total, dtype=np.uint64)
        cols["Clock"] = np.arange(total, dtype=np.uint64) * 7
        schema = pa.schema([(k, pa.from_numpy_dtype(v.dtype)) for k, v in cols.items()])
        f = tmp_path / "NeuropixelsV2_ProbeA_0.arrow"
        off = 0
        with ipc.new_file(f, schema) as w:
            for n in batch_sizes:
                w.write_batch(
                    pa.RecordBatch.from_arrays(
                        [pa.array(v[off : off + n]) for v in cols.values()], schema=schema
                    )
                )
                off += n
        return f

    def test_sample_count_includes_short_final_batch(self, tmp_path):
        """Review Focus #1: a uniform-batch assumption would lose the tail."""
        from aeon.dj_pipeline.utils.ephys_utils import read_amplifier_recording

        f = self._arrow_probe_file(tmp_path, [500, 500, 137])
        rec = read_amplifier_recording(f, num_channels=4, fs_hz=30000.0)
        assert rec.get_num_samples() == 1137

    def test_clock_columns_are_not_channels(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import read_amplifier_recording

        f = self._arrow_probe_file(tmp_path, [100], n_channels=4)
        rec = read_amplifier_recording(f, num_channels=4, fs_hz=30000.0)
        assert rec.get_num_channels() == 4, "Clock/HubClock must not be read as channels"

    def test_traces_match_source_columns(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import read_amplifier_recording

        f = self._arrow_probe_file(tmp_path, [100], n_channels=3)
        rec = read_amplifier_recording(f, num_channels=3, fs_hz=30000.0)
        traces = rec.get_traces(start_frame=0, end_frame=5)
        assert traces.shape == (5, 3)
        # column i was filled with the constant i
        assert [int(traces[0, i]) for i in range(3)] == [0, 1, 2]

    def test_gains_are_applied(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import read_amplifier_recording

        f = self._arrow_probe_file(tmp_path, [50])
        rec = read_amplifier_recording(f, num_channels=4, fs_hz=30000.0, gain_to_uV=2.5)
        assert rec.get_channel_gains()[0] == 2.5
