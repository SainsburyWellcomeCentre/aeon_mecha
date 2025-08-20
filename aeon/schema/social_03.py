"""Schema definition for social_03 experiments-specific data streams."""

from swc.aeon.io import reader
from swc.aeon.schema import Stream
from aeon.dj_pipeline.utils import paths


class Pose(Stream):
    def __init__(self, path):
        """Initializes the Pose stream."""
        model_root = paths.get_repository_path('ceph_aeon') / "aeon" / "data" / "ingest"
        super().__init__(reader.Pose(f"{path}_222*", model_root=model_root.as_posix()))


class EnvironmentActiveConfiguration(Stream):
    def __init__(self, path):
        """Initializes the EnvironmentActiveConfiguration stream."""
        super().__init__(reader.JsonList(f"{path}_ActiveConfiguration_*", columns=["name"]))
