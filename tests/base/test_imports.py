"""Test basic imports."""


def test_api_imports():
    from swc.aeon.io import api as io_api  # noqa: PLC0415

    assert hasattr(io_api, "load")

