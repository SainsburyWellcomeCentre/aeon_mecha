"""NetCDF (xarray) SchemaCodec for DataJoint — mirrors datajoint's NpyCodec.

Persists an ``xarray.Dataset`` as a single NetCDF-4 ``.nc`` file at a
schema-addressed path (``{schema}/{table}/{pk}/{field}_<token>.nc``) in a
``protocol: file`` store, and reopens it lazily on fetch.

Only ``Dataset`` is accepted — a ``DataArray`` is rejected (call
``.to_dataset()`` first). This keeps insert and fetch symmetric: you never store
a ``DataArray`` and get a ``Dataset`` back.

The codec is fully generic: it knows nothing about any particular dataset schema
or domain format — it only round-trips an ``xarray.Dataset`` to and from NetCDF.

Differences vs ``NpyCodec``:
    - A ``.nc`` is a single file we open lazily from disk, so for ``protocol: file``
      stores we write/open it directly by local path (``backend._full_path``)
      rather than buffering bytes through ``put_buffer``/``get_buffer``.
    - No object-store (S3) support — the codec asserts ``protocol == "file"``.

Laziness uses xarray's own lazy indexing (``chunks=None``); no ``dask`` currently.
Accessing a variable pulls it fully into RAM as numpy (``.load()`` realizes it).
"""

from __future__ import annotations

import os
from typing import Any

import xarray as xr
from datajoint.builtin_codecs import SchemaCodec
from datajoint.errors import DataJointError


class XArrayNetCDFCodec(SchemaCodec):
    """Store an xarray.Dataset as NetCDF-4 at {schema}/{table}/{pk}/{field}_<token>.nc.

    Usable as ``<xarray@store>`` (the ``@`` store modifier is required). Fetch
    returns a lazy ``xarray.Dataset``. File stores only.
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
        value.to_netcdf(local_path, engine="h5netcdf")
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
        return xr.open_dataset(local_path, engine="h5netcdf", chunks=None)
