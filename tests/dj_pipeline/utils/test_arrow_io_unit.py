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
        # Lower bound too: if the implementation moves to read_all()/read_next_batch,
        # the patch records nothing and '<= 2' would pass while the file is scanned.
        assert 1 <= len(calls) <= 2, f"read {len(calls)} batches, expected 1 or 2"

    def test_missing_column_raises_naming_available_columns(self, tmp_path):
        from aeon.dj_pipeline.utils.arrow_io import read_arrow_column

        f = tmp_path / "t.arrow"
        _write_arrow(f, [10])
        with pytest.raises(KeyError) as exc:
            read_arrow_column(f, "HubClock")
        message = str(exc.value)
        assert "HubClock" in message
        # Asserting "Clock" here would be satisfied by "HubClock" itself; the point
        # is that the message lists what IS available, so name a column only the
        # available list can supply.
        assert "AmplifierData0" in message, message

    def test_empty_file_raises(self, tmp_path):
        import pyarrow as pa
        import pyarrow.ipc as ipc

        from aeon.dj_pipeline.utils.arrow_io import read_arrow_clock_bounds

        f = tmp_path / "empty.arrow"
        with ipc.new_file(f, pa.schema([("Clock", pa.uint64())])):
            pass
        with pytest.raises(ValueError, match="no record batches"):
            read_arrow_clock_bounds(f)

    def test_empty_file_column_keeps_its_declared_dtype(self, tmp_path):
        """A float64 empty clock would silently change the timebase in extend_clock."""
        import pyarrow as pa
        import pyarrow.ipc as ipc

        from aeon.dj_pipeline.utils.arrow_io import read_arrow_column

        f = tmp_path / "empty.arrow"
        with ipc.new_file(f, pa.schema([("Clock", pa.uint64())])):
            pass
        out = read_arrow_column(f, "Clock")
        assert len(out) == 0
        assert out.dtype == np.uint64

    def test_bounds_walk_past_empty_batches_at_either_end(self, tmp_path):
        """A writer that flushes at stop leaves an empty batch; the file still has data."""
        import pyarrow as pa
        import pyarrow.ipc as ipc

        from aeon.dj_pipeline.utils.arrow_io import read_arrow_clock_bounds

        f = tmp_path / "padded.arrow"
        schema = pa.schema([("Clock", pa.uint64())])
        with ipc.new_file(f, schema) as w:
            for values in ([], [10, 20, 30], []):
                w.write_batch(
                    pa.RecordBatch.from_arrays([pa.array(values, type=pa.uint64())], schema=schema)
                )
        assert read_arrow_clock_bounds(f) == (10, 30)

    def test_all_batches_empty_raises(self, tmp_path):
        import pyarrow as pa
        import pyarrow.ipc as ipc

        from aeon.dj_pipeline.utils.arrow_io import read_arrow_clock_bounds

        f = tmp_path / "all_empty.arrow"
        schema = pa.schema([("Clock", pa.uint64())])
        with ipc.new_file(f, schema) as w:
            w.write_batch(pa.RecordBatch.from_arrays([pa.array([], type=pa.uint64())], schema=schema))
        with pytest.raises(ValueError, match="none with rows"):
            read_arrow_clock_bounds(f)

    def test_null_in_a_column_raises_rather_than_upcasting(self, tmp_path):
        """A null turns an integer column into float64 + NaN, silently.

        On Clock that changes the timebase and loses precision on large ticks, which
        is the whole reason the empty-file branch preserves the declared dtype.
        """
        import pyarrow as pa
        import pyarrow.ipc as ipc

        from aeon.dj_pipeline.utils.arrow_io import read_arrow_column

        f = tmp_path / "nulls.arrow"
        schema = pa.schema([("Clock", pa.uint64())])
        with ipc.new_file(f, schema) as w:
            w.write_batch(
                pa.RecordBatch.from_arrays([pa.array([1, None, 3], type=pa.uint64())], schema=schema)
            )
        with pytest.raises(ValueError, match="null value"):
            read_arrow_column(f, "Clock")

    def test_empty_chunk_raises_a_type_callers_can_catch(self, tmp_path):
        """find_overlapping_bno055_chunks catches EmptyArrowChunk to skip empty chunks.

        Asserting only ValueError would leave that contract unpinned, so narrowing the
        type back to a plain ValueError would silently turn the skip into a hard fail.
        """
        import pyarrow as pa
        import pyarrow.ipc as ipc

        from aeon.dj_pipeline.utils.arrow_io import EmptyArrowChunk, read_arrow_clock_bounds

        f = tmp_path / "empty.arrow"
        with ipc.new_file(f, pa.schema([("Clock", pa.uint64())])):
            pass
        with pytest.raises(EmptyArrowChunk):
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

    def _varying_probe_file(self, tmp_path, batch_sizes, n_channels=2):
        """Cell (row, ch) holds row*10 + ch, so any row or channel shift changes values."""
        import pyarrow as pa
        import pyarrow.ipc as ipc

        total = sum(batch_sizes)
        rows = np.arange(total, dtype=np.uint16)
        cols = {f"AmplifierData{i}": rows * 10 + i for i in range(n_channels)}
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

    def test_traces_across_a_batch_boundary(self, tmp_path):
        """A read straddling two batches is where stride arithmetic first goes wrong.

        get_traces addresses batches by a constant stride, so an off-by-one there
        shifts every spike time without changing the sample count.
        """
        from aeon.dj_pipeline.utils.ephys_utils import read_amplifier_recording

        f = self._varying_probe_file(tmp_path, [40, 40, 13])
        rec = read_amplifier_recording(f, num_channels=2, fs_hz=30000.0)
        traces = rec.get_traces(start_frame=38, end_frame=43)
        assert traces.tolist() == [[r * 10, r * 10 + 1] for r in range(38, 43)]

    def test_traces_wholly_inside_the_short_final_batch(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import read_amplifier_recording

        f = self._varying_probe_file(tmp_path, [40, 40, 13])
        rec = read_amplifier_recording(f, num_channels=2, fs_hz=30000.0)
        traces = rec.get_traces(start_frame=85, end_frame=93)
        assert traces.tolist() == [[r * 10, r * 10 + 1] for r in range(85, 93)]

    def test_traces_to_end_of_a_file_with_no_short_tail(self, tmp_path):
        """A chunk with no short tail puts end_frame exactly on a batch boundary.

        A full 18M-sample chunk divides evenly, so deriving the last batch from
        end_frame rather than end_frame - 1 indexes one past the final batch here.
        """
        from aeon.dj_pipeline.utils.ephys_utils import read_amplifier_recording

        f = self._varying_probe_file(tmp_path, [40, 40])
        rec = read_amplifier_recording(f, num_channels=2, fs_hz=30000.0)
        traces = rec.get_traces(start_frame=0, end_frame=80)
        assert traces.tolist() == [[r * 10, r * 10 + 1] for r in range(80)]


class TestStreamNameSurvivesSerialisation:
    """The recording must rebuild with the stream it was created for.

    PreProcessing pickles the recording and parallel workers rebuild it from
    ``_kwargs``. Leaving ``stream_name`` out of that dict makes the rebuilt copy fall
    back to the default stream, so a clock read would come back as amplifier data with
    nothing to notice — this is the #630 defect that had to be fixed before the arrow
    clock path could be built on it.
    """

    def _write(self, path, n_rows=8):
        import pyarrow as pa
        import pyarrow.ipc as ipc

        cols = {
            "AmplifierData0": np.arange(n_rows, dtype=np.uint16),
            "Clock": np.arange(100, 100 + n_rows, dtype=np.uint64),
        }
        schema = pa.schema([(k, pa.from_numpy_dtype(v.dtype)) for k, v in cols.items()])
        with ipc.new_file(path, schema) as w:
            w.write_batch(pa.RecordBatch.from_arrays([pa.array(v) for v in cols.values()], schema=schema))
        return path

    def test_non_default_stream_survives_a_round_trip(self, tmp_path):
        import spikeinterface as si

        from aeon.dj_pipeline.utils.spike_sorting_utils import read_bonsai_onix_arrow

        f = self._write(tmp_path / "NeuropixelsV2_ProbeA_0.arrow")
        rec = read_bonsai_onix_arrow(f, sampling_frequency=30000.0, stream_name="Clock")
        assert rec.get_num_channels() == 1

        rebuilt = si.load(rec.to_dict())
        assert list(rebuilt.channel_ids) == list(rec.channel_ids), (
            f"rebuilt with {list(rebuilt.channel_ids)}, not the Clock stream it was made for"
        )
        np.testing.assert_array_equal(
            rebuilt.get_traces(start_frame=0, end_frame=8),
            rec.get_traces(start_frame=0, end_frame=8),
        )


class TestArrowChannelIntegrity:
    """Channel count and order come from the arrow schema; both must be checked.

    PreProcessing maps electrodes positionally
    (``si_recording.channel_ids[electrodes_df.channel_idx.values]``), so a column
    count that disagrees with the registered ElectrodeConfig, or columns in
    non-numeric order, mislabels every channel with no error.
    """

    def _write(self, path, names):
        import pyarrow as pa
        import pyarrow.ipc as ipc

        cols = {n: np.zeros(4, dtype=np.uint16) for n in names}
        cols["Clock"] = np.arange(4, dtype=np.uint64)
        schema = pa.schema([(k, pa.from_numpy_dtype(v.dtype)) for k, v in cols.items()])
        with ipc.new_file(path, schema) as w:
            w.write_batch(pa.RecordBatch.from_arrays([pa.array(v) for v in cols.values()], schema=schema))

    def test_channel_count_mismatch_raises(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import read_amplifier_recording

        f = tmp_path / "NeuropixelsV2_ProbeA_0.arrow"
        self._write(f, [f"AmplifierData{i}" for i in range(4)])
        with pytest.raises(ValueError, match="AmplifierData columns but the ElectrodeConfig"):
            read_amplifier_recording(f, num_channels=8, fs_hz=30000.0)

    def test_non_numeric_channel_order_raises(self, tmp_path):
        """Lexicographic order (0, 1, 10, 2, ...) would silently transpose channels."""
        from aeon.dj_pipeline.utils.ephys_utils import read_amplifier_recording

        f = tmp_path / "NeuropixelsV2_ProbeA_0.arrow"
        self._write(f, ["AmplifierData0", "AmplifierData1", "AmplifierData10", "AmplifierData2"])
        with pytest.raises(ValueError, match="not in numeric order"):
            read_amplifier_recording(f, num_channels=4, fs_hz=30000.0)

    def test_non_uint16_samples_are_rejected(self, tmp_path):
        """uint8/uint32 pass every other check and then get 16-bit gain applied."""
        import pyarrow as pa
        import pyarrow.ipc as ipc

        from aeon.dj_pipeline.utils.ephys_utils import read_amplifier_recording

        f = tmp_path / "NeuropixelsV2_ProbeA_0.arrow"
        cols = {f"AmplifierData{i}": np.zeros(4, dtype=np.uint32) for i in range(2)}
        cols["Clock"] = np.arange(4, dtype=np.uint64)
        schema = pa.schema([(k, pa.from_numpy_dtype(v.dtype)) for k, v in cols.items()])
        with ipc.new_file(f, schema) as w:
            w.write_batch(pa.RecordBatch.from_arrays([pa.array(v) for v in cols.values()], schema=schema))
        with pytest.raises(ValueError, match="expected uint16"):
            read_amplifier_recording(f, num_channels=2, fs_hz=30000.0)


class TestLeadingEmptyBatch:
    """batch_len comes from batch 0, so an empty batch 0 leaves the stride undefined.

    read_arrow_clock_bounds tolerates an empty batch at either end because it only
    reads first and last values -- no stride arithmetic is involved. get_traces is
    the opposite: it addresses batch i at i*batch_len, which a leading empty batch
    makes false for every batch after it. The two readers differ on purpose; what
    must not happen is batch_len reaching get_traces as zero.
    """

    def _write(self, path, batch_sizes):
        import pyarrow as pa
        import pyarrow.ipc as ipc

        total = sum(batch_sizes)
        rows = np.arange(total, dtype=np.uint16)
        schema = pa.schema([("AmplifierData0", pa.uint16()), ("Clock", pa.uint64())])
        off = 0
        with ipc.new_file(path, schema) as w:
            for n in batch_sizes:
                w.write_batch(
                    pa.RecordBatch.from_arrays(
                        [
                            pa.array(rows[off : off + n]),
                            pa.array(np.arange(off, off + n, dtype=np.uint64)),
                        ],
                        schema=schema,
                    )
                )
                off += n
        return path

    def test_empty_first_batch_is_rejected_by_name(self, tmp_path):
        """[0, 100] slipped through: 0 != 0 is false and the last batch is exempt.

        get_traces then died on 'integer division or modulo by zero', naming neither
        the file nor the cause.
        """
        from aeon.dj_pipeline.utils.ephys_utils import read_amplifier_recording

        f = self._write(tmp_path / "NeuropixelsV2_ProbeA_0.arrow", [0, 100])
        with pytest.raises(ValueError, match="first record batch is empty"):
            read_amplifier_recording(f, num_channels=1, fs_hz=30000.0)

    def test_trailing_empty_batch_is_accepted(self, tmp_path):
        """A writer flushing at stop costs nothing: the stride still holds."""
        from aeon.dj_pipeline.utils.ephys_utils import read_amplifier_recording

        f = self._write(tmp_path / "NeuropixelsV2_ProbeA_0.arrow", [100, 100, 0])
        rec = read_amplifier_recording(f, num_channels=1, fs_hz=30000.0)
        assert rec.get_num_samples() == 200
        assert rec.get_traces(start_frame=98, end_frame=102).ravel().tolist() == [98, 99, 100, 101]


class TestNonUniformBatchGuard:
    """get_traces addresses batches by a constant; a file that breaks it must not pass.

    ``start_frame // batch_len`` assumes every batch but the last has the same row
    count. ``get_num_samples`` sums true per-batch rows, so a non-uniform file keeps
    a correct length while every trace read is offset -- a recording that looks
    healthy with every spike time wrong.
    """

    def _write(self, path, batch_sizes):
        import pyarrow as pa
        import pyarrow.ipc as ipc

        total = sum(batch_sizes)
        cols = {
            "AmplifierData0": np.arange(total, dtype=np.uint16),
            "Clock": np.arange(total, dtype=np.uint64),
        }
        schema = pa.schema([(k, pa.from_numpy_dtype(v.dtype)) for k, v in cols.items()])
        off = 0
        with ipc.new_file(path, schema) as w:
            for n in batch_sizes:
                w.write_batch(
                    pa.RecordBatch.from_arrays(
                        [pa.array(v[off : off + n]) for v in cols.values()], schema=schema
                    )
                )
                off += n

    def test_uniform_with_short_final_batch_is_accepted(self, tmp_path):
        """The real golden files are exactly this shape."""
        from aeon.dj_pipeline.utils.ephys_utils import read_amplifier_recording

        f = tmp_path / "NeuropixelsV2_ProbeA_0.arrow"
        self._write(f, [100, 100, 37])
        rec = read_amplifier_recording(f, num_channels=1, fs_hz=30000.0)
        assert rec.get_num_samples() == 237

    def test_non_uniform_middle_batch_raises(self, tmp_path):
        from aeon.dj_pipeline.utils.ephys_utils import read_amplifier_recording

        f = tmp_path / "NeuropixelsV2_ProbeA_0.arrow"
        self._write(f, [100, 70, 100])
        with pytest.raises(ValueError, match="addresses batches by a constant stride"):
            read_amplifier_recording(f, num_channels=1, fs_hz=30000.0)
