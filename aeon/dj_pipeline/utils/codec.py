"""Codecs for lazy-loading data referenced from MySQL.

The stream codecs store JSON references; on fetch, they reconstruct readers and
return DataFrames built from the raw files on disk.

- ``AeonStreamCodec`` (``<aeon_stream>``) — HARP-clocked time-indexed streams,
  loaded via ``io_api.load(start, end)``.
- ``OnixStreamCodec`` (``<aeon_onix_stream>``) — ONIX-clocked stream groups
  (e.g., Bno055 IMU). Structural-only: loads + prefix-renames + concats. Does
  NOT apply HARP sync regression — that's exposed via ``OnixImuChunk.synced_df``.
- ``XArrayNetCDFCodec`` (``<xarray@store>``) — an ``xarray.Dataset`` persisted as a
  NetCDF-4 file in a ``protocol: file`` store, reopened lazily on fetch.
- ``PynappleCodec`` (``<pynapple@store>``) — a pynapple object persisted as a
  ``.npz`` in a ``protocol: file`` store. ``TsGroup`` decodes through a fast path
  equivalent to ``nap.load_file`` but several times quicker; every other type goes
  through ``nap.load_file`` directly. ``pynapple`` is an optional extra.

The pynapple payload is an uncompressed ``.npz`` by design: one file per value, which
is what DataJoint's external store tracks. ``savez_compressed`` is 55-70x slower to
write; zarr is 40-60x slower to *open*, isn't what ``nap.load_file`` reads, and is a
directory the codec would have to manage itself. Reading one member (``keys``,
``_metadata``) still costs only kilobytes.

A zip leaves its members byte-unaligned, so nothing here is lazy: ``np.load(mmap_mode=...)``
silently does nothing on one, and a value passed in lazily (a zarr-backed ``TsdFrame``
built with ``load_array=False``) comes back as a plain ndarray. Laziness would need a
non-npz backend.
"""

import os
from typing import Any

import datajoint as dj
import numpy as np
import pandas as pd
import xarray as xr
from datajoint.builtin_codecs import SchemaCodec
from datajoint.errors import DataJointError


class AeonStreamCodec(dj.Codec):
    """Codec for lazy-loading stream data from raw files.

    Used for the `stream_df` column in auto-generated stream tables.
    On insert, stores a self-contained JSON reference with enough info
    to reconstruct the stream reader and load data at fetch time.
    On fetch, returns the full pd.DataFrame from raw files.

    Stored JSON format::

        {
            "stream_type": "Encoder",
            "experiment_name": "abcBehav0-aeon3",
            "device_name": "Feeder1",
            "chunk_start": "2025-11-18 10:13:15",
            "chunk_end": "2025-11-18 11:00:00",
            "epoch_start": "2025-11-18 10:13:15"
        }
    """

    name = "aeon_stream"

    _REQUIRED_KEYS = {
        "stream_type",
        "experiment_name",
        "device_name",
        "chunk_start",
        "chunk_end",
        "epoch_start",
    }

    def get_dtype(self, is_store: bool) -> str:
        """Return JSON as the storage type."""
        return "json"

    def encode(self, value, *, key=None, store_name=None):
        """Validate and store the stream reference dict as JSON."""
        if not isinstance(value, dict):
            raise TypeError(f"AeonStreamCodec expects a dict, got {type(value).__name__}")
        missing = self._REQUIRED_KEYS - value.keys()
        if missing:
            raise ValueError(f"AeonStreamCodec missing required keys: {missing}")
        return value

    def decode(self, stored, *, key=None):
        """Load stream data from raw files using the stored reference."""
        from swc.aeon.io import api as io_api

        from aeon.dj_pipeline import acquisition
        from aeon.dj_pipeline.utils.load_metadata import get_stream_reader_for_epoch

        data_dirs = acquisition.Experiment.get_data_directories(
            {"experiment_name": stored["experiment_name"]}
        )
        stream_reader = get_stream_reader_for_epoch(
            stored["experiment_name"],
            stored["device_name"],
            stored["stream_type"],
            stored["epoch_start"],
        )
        return io_api.load(
            root=data_dirs,
            reader=stream_reader,
            start=pd.Timestamp(stored["chunk_start"]),
            end=pd.Timestamp(stored["chunk_end"]),
        )


class OnixStreamCodec(dj.Codec):
    """Structural-only codec for ONIX-clocked stream groups (e.g., Bno055 IMU).

    Stores a self-contained JSON reference. On fetch, loads the referenced
    binaries via the dotmap reader hierarchy in ``aeon/schema/ephys.py``,
    prefix-renames stream columns by their source class, concats on the shared
    sample index, and returns an **ONIX-clock-indexed** DataFrame.

    The codec deliberately does NOT apply the HARP sync regression — that's
    the caller's responsibility, exposed as ``OnixImuChunk.synced_df``.

    Stored JSON format::

        {
            "experiment_name": "...",
            "epoch_start": "...",
            "sync_start": "...",
            "device_name": "NeuropixelsV2Beta",
            "stream_group": "Bno055"
        }
    """

    name = "aeon_onix_stream"

    _REQUIRED_KEYS = {
        "experiment_name",
        "epoch_start",
        "sync_start",
        "device_name",
        "stream_group",
    }

    def get_dtype(self, is_store: bool) -> str:
        """Return JSON as the storage type."""
        return "json"

    def encode(self, value, *, key=None, store_name=None):
        """Validate and store the stream reference dict as JSON."""
        if not isinstance(value, dict):
            raise TypeError(f"OnixStreamCodec expects a dict, got {type(value).__name__}")
        missing = self._REQUIRED_KEYS - value.keys()
        if missing:
            raise ValueError(f"OnixStreamCodec missing required keys: {missing}")
        return value

    def decode(self, stored, *, key=None):
        """Load + merge the referenced ONIX stream group as an ONIX-indexed DataFrame.

        Reloads the same Bno055 chunks the populate-time row was built from
        (via ``chunk_indices``), then filters to ``[onix_ts_start, onix_ts_end]``.
        """
        # Lazy imports to avoid circular references at module load time.
        from aeon.dj_pipeline import acquisition, ephys
        from aeon.dj_pipeline.utils.onix_imu import (
            IMU_COLUMNS,
            find_overlapping_bno055_chunks,
            load_and_merge_bno055,
        )

        if stored["stream_group"] != "Bno055":
            raise NotImplementedError(
                f"stream_group={stored['stream_group']!r} not supported. Only 'Bno055' is wired today."
            )

        sm_key = {
            "experiment_name": stored["experiment_name"],
            "epoch_start": pd.Timestamp(stored["epoch_start"]),
            "sync_start": pd.Timestamp(stored["sync_start"]),
        }

        epoch_dir = (ephys.EphysEpoch & sm_key).fetch1("epoch_dir")
        raw_dir = acquisition.Experiment.get_data_directory(
            {"experiment_name": stored["experiment_name"]}, "raw-ephys"
        )
        if raw_dir is None:
            raise FileNotFoundError(
                f"No raw-ephys data directory registered for experiment {stored['experiment_name']!r}"
            )
        device_dir = raw_dir / epoch_dir / stored["device_name"]

        # Prefer chunk_indices captured at populate time; fall back to a
        # fresh overlap scan for backward-compat with older rows.
        chunk_indices = stored.get("chunk_indices")
        onix_ts_start = stored.get("onix_ts_start")
        onix_ts_end = stored.get("onix_ts_end")
        if onix_ts_start is None or onix_ts_end is None:
            ts_start_raw, ts_end_raw = (ephys.EphysSyncModel & sm_key).fetch1(
                "onix_ts_start", "onix_ts_end"
            )
            onix_ts_start = int(ts_start_raw)
            onix_ts_end = int(ts_end_raw)
        if chunk_indices is None:
            chunk_indices = find_overlapping_bno055_chunks(
                device_dir,
                stored["device_name"],
                int(onix_ts_start),
                int(onix_ts_end),
            )

        if not chunk_indices:
            return pd.DataFrame(columns=list(IMU_COLUMNS), index=pd.Index([], dtype=np.uint64))

        df = pd.concat([load_and_merge_bno055(device_dir, stored["device_name"], n) for n in chunk_indices])
        return df[(df.index >= int(onix_ts_start)) & (df.index <= int(onix_ts_end))]


class XArrayNetCDFCodec(SchemaCodec):
    """Store an xarray.Dataset as NetCDF-4 at {schema}/{table}/{pk}/{field}_<token>.nc.

    Usable as ``<xarray@store>`` (the ``@`` store modifier is required); ``protocol:
    file`` stores only, no object stores. Unlike ``NpyCodec``, a ``.nc`` is a single
    file opened lazily from disk, so it is written/read directly by local path
    (``backend._full_path``) rather than buffered through ``put_buffer``/``get_buffer``.

    Fully generic: it knows nothing about any particular dataset schema or domain
    format — it only round-trips an ``xarray.Dataset`` to and from NetCDF. Only
    ``Dataset`` is accepted; a ``DataArray`` is rejected (call ``.to_dataset()``
    first) so that insert and fetch stay symmetric.
    """

    name = "xarray"

    def validate(self, value: Any) -> None:
        """Accept only an xarray.Dataset; a DataArray must be converted by the caller."""
        if not isinstance(value, xr.Dataset):
            hint = " — call .to_dataset() first" if isinstance(value, xr.DataArray) else ""
            raise DataJointError(f"<xarray> requires an xarray.Dataset, got {type(value).__name__}{hint}")

    def _local_path(self, path: str, store_name: str | None, config) -> str:
        """Resolve a store-relative path to an absolute local filesystem path."""
        backend = self._get_backend(store_name, config=config)
        if backend.protocol != "file":
            raise DataJointError("<xarray> supports only `protocol: file` stores")
        return backend._full_path(path)

    def encode(self, value: xr.Dataset, *, key: dict | None = None, store_name: str | None = None) -> dict:
        """Write the Dataset to a NetCDF-4 file and return JSON metadata."""
        schema, table, field, primary_key = self._extract_context(key)
        config = (key or {}).get("_config")
        path, _token = self._build_path(
            schema, table, field, primary_key, ext=".nc", store_name=store_name, config=config
        )
        local_path = self._local_path(path, store_name, config)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        value.to_netcdf(local_path, engine="netcdf4")
        return {
            "path": path,
            "store": store_name,
            "dims": dict(value.sizes),
            "data_vars": list(value.data_vars),
        }

    def decode(self, stored: dict, *, key: dict | None = None) -> xr.Dataset:
        """Reopen the stored NetCDF file as a lazy xarray.Dataset (no dask)."""
        config = (key or {}).get("_config")
        local_path = self._local_path(stored["path"], stored.get("store"), config)
        return xr.open_dataset(local_path, engine="netcdf4")


def _narrow_int(values: np.ndarray) -> np.ndarray:
    """Return the narrowest signed-int view of ``values`` that holds its full range.

    ``argsort(kind="stable")`` radix-sorts one pass per byte, so an int16 key is
    ~4x less work than int64. Only the sort key is narrowed; the values reaching
    the output stay int64. Both bounds are checked — pynapple permits negative
    unit keys, and a cast that wrapped would reorder spikes silently.
    """
    if not values.size:
        return values
    lo, hi = int(values.min()), int(values.max())
    for dtype in (np.int16, np.int32):
        info = np.iinfo(dtype)
        if info.min <= lo and hi <= info.max:
            return values.astype(dtype, copy=False)
    return values


def _to_members(value: Any) -> dict[str, np.ndarray]:
    """Build the mapping pynapple's ``save()`` hands to ``np.savez``, without a file.

    Every ``save()`` ends in ``np.savez(filename, **members)``, so assembling the
    same mapping lets the in-DB form skip the npz container entirely - 2-10x
    smaller, and no filesystem round trip. ``TestPynappleMemberParity`` pins this
    to pynapple's own output, because it reads ``_metadata`` directly.
    """
    kind = type(value).__name__
    members: dict[str, np.ndarray] = {"type": np.array([kind])}

    if kind == "IntervalSet":
        members["start"] = np.asarray(value.start)
        members["end"] = np.asarray(value.end)
        members["_metadata"] = np.array(dict(value._metadata), dtype=object)
        return members

    support = value.time_support
    members["start"] = np.asarray(support.start)
    members["end"] = np.asarray(support.end)

    if kind == "TsGroup":
        times = [value[unit].t for unit in value.index]
        index = [np.full(len(value[unit]), unit, dtype=np.int64) for unit in value.index]
        members["t"] = np.concatenate(times) if times else np.empty(0, dtype=np.float64)
        members["index"] = np.concatenate(index) if index else np.empty(0, dtype=np.int64)
        members["keys"] = np.asarray(value.index, dtype=np.int64)
        # `rate` is derived from the support, so pynapple drops it before writing.
        members["_metadata"] = np.array(dict(value._metadata.drop("rate")), dtype=object)
        return members

    members["t"] = np.asarray(value.t)
    if kind != "Ts":
        members["d"] = np.asarray(value.values)
    if kind == "TsdFrame":
        members["columns"] = np.asarray(value.columns, dtype=object)
        members["_metadata"] = np.array(dict(value._metadata), dtype=object)
    return members


def _tsgroup_from_npz(local_path: str):
    """Rebuild a TsGroup from a pynapple .npz without the per-unit mask loop.

    ``TsGroup._from_npz_reader`` runs ``index == key`` once per unit, O(units x
    events); one stable argsort plus offset slicing is O(n log n). Bit-identical,
    and the stored file is unchanged, so stock ``nap.load_file`` still reads it.
    ``TestPynappleFastPath`` pins the two together.
    """
    import pynapple as nap

    with np.load(local_path, allow_pickle=True) as npz:
        names = set(npz.files)
        times, index, keys = npz["t"], npz["index"], npz["keys"]
        start, end = npz["start"], npz["end"]
        values = npz["d"] if "d" in names else None
        metadata = npz["_metadata"].item() if "_metadata" in names else {}

    support = nap.IntervalSet(start=start, end=end)
    order = np.argsort(_narrow_int(index), kind="stable")  # stable keeps per-unit time order
    times = times[order]
    index = index[order]
    lo = np.searchsorted(index, keys, side="left")
    hi = np.searchsorted(index, keys, side="right")

    if values is None:
        data = {int(k): nap.Ts(t=times[lo[i] : hi[i]], time_support=support) for i, k in enumerate(keys)}
    else:
        values = values[order]
        data = {
            int(k): nap.Tsd(t=times[lo[i] : hi[i]], d=values[lo[i] : hi[i]], time_support=support)
            for i, k in enumerate(keys)
        }
    # Members already carry the group support, so bypass_check is safe. Passing it
    # without that would compute `rate` from each member's own support instead.
    return nap.TsGroup(data, time_support=support, bypass_check=True, metadata=metadata)


_SUMMARY_EXTRAS = {
    "Tsd": lambda v: {"dtype": str(v.values.dtype)},
    "TsdFrame": lambda v: {
        "dtype": str(v.values.dtype),
        "n_columns": int(v.shape[1]),
        "columns": [str(c) for c in v.columns],
    },
    "TsdTensor": lambda v: {"dtype": str(v.values.dtype), "shape": [int(n) for n in v.shape]},
    "IntervalSet": lambda v: {"total_seconds": float(v.tot_length())},
    "TsGroup": lambda v: {"n_events": int(sum(len(v[u]) for u in v.index))},
}
"""Per-type additions to the stored JSON summary. Additive and optional: a type
absent from this table still gets ``kind``/``n_rows``/``t_start``/``t_end``."""


class PynappleCodec(SchemaCodec):
    """Store a pynapple object as .npz at {schema}/{table}/{pk}/{field}_<token>.npz.

    Usable as ``<pynapple@store>``; the ``@`` store modifier is required, and only
    ``protocol: file`` stores are supported. ``obj.save()`` and ``nap.load_file()``
    are path-only, so the file is written and read directly by local path rather
    than buffered through ``put_buffer``/``get_buffer``.

    Domain-agnostic: it round-trips a pynapple object and knows nothing about what
    the object means. ``pynapple`` is an optional extra, imported lazily inside the
    methods, so a schema that declares no ``<pynapple@…>`` column never needs it.

    The stored JSON summary is queryable without decoding anything — ``proj`` on a
    JSON path returns a scalar and never opens the ``.npz``::

        Table.proj(kind='data->>"$.kind"', n='data->>"$.n_events"')
        Table & {"data.kind": "TsGroup"}

    ``describe()`` shows only ``<pynapple@store>``, so the concrete pynapple class
    is discoverable this way rather than from the schema.
    """

    name = "pynapple"

    def validate(self, value: Any) -> None:
        """Accept anything pynapple can round-trip through its own ``.npz``.

        Tests the capability rather than an enumerated list of the six current
        containers, so a type pynapple adds later works without touching this codec.
        ``Folder`` is excluded by the reader check — it has ``save`` but takes
        ``(name, obj)`` and has no ``_from_npz_reader``. The six names stay in the
        message because a type error should say what was expected.
        """
        if not (hasattr(value, "save") and hasattr(type(value), "_from_npz_reader")):
            raise DataJointError(
                "<pynapple> requires a pynapple object (Ts, Tsd, TsdFrame, TsdTensor, "
                f"IntervalSet, TsGroup), got {type(value).__name__}"
            )

    def _local_path(self, path: str, store_name: str | None, config) -> str:
        """Resolve a store-relative path to an absolute local filesystem path."""
        backend = self._get_backend(store_name, config=config)
        if backend.protocol != "file":
            raise DataJointError("<pynapple> supports only `protocol: file` stores")
        return backend._full_path(path)

    @staticmethod
    def _summary(value: Any) -> dict:
        """Queryable summary, so a caller can size a query without opening the file.

        ``kind``/``n_rows``/``t_start``/``t_end`` are always present, so one query
        works across every column of this type; ``n_rows`` counts the primary entity
        (samples, intervals, or units). ``_SUMMARY_EXTRAS`` adds per-type detail, and
        a type absent from it still gets the four core keys.
        """
        kind = type(value).__name__
        # An IntervalSet *is* its own support; everything else carries one.
        support = getattr(value, "time_support", value)
        summary = {
            "kind": kind,
            "n_rows": int(len(value)),
            "t_start": float(support.start[0]) if len(support) else None,
            "t_end": float(support.end[-1]) if len(support) else None,
        }
        if extras := _SUMMARY_EXTRAS.get(kind):
            summary.update(extras(value))
        return summary

    def encode(self, value: Any, *, key: dict | None = None, store_name: str | None = None) -> dict:
        """Write the pynapple object to a .npz file and return JSON metadata."""
        schema, table, field, primary_key = self._extract_context(key)
        config = (key or {}).get("_config")
        path, _token = self._build_path(
            schema, table, field, primary_key, ext=".npz", store_name=store_name, config=config
        )
        local_path = self._local_path(path, store_name, config)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        value.save(local_path)
        return {"path": path, "store": store_name, **self._summary(value)}

    def decode(self, stored: dict, *, key: dict | None = None) -> Any:
        """Reopen the stored .npz as a pynapple object.

        ``TsGroup`` takes a faster reconstruction that is equivalent to
        ``nap.load_file``; every other type goes through it directly.
        """
        import pynapple as nap

        config = (key or {}).get("_config")
        local_path = self._local_path(stored["path"], stored.get("store"), config)
        if stored.get("kind") == "TsGroup":
            return _tsgroup_from_npz(local_path)
        return nap.load_file(local_path)
