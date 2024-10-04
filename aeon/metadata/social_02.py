from dataclasses import dataclass

import numpy as np
from dotmap import DotMap
from numpy.typing import NDArray
from pandas import DataFrame


@dataclass
class ArenaMetadata:
    """Represents information about the physical arena."""

    radius_cm: float
    radius_px: float
    center_px: NDArray[np.float64]
    pixel_to_cm: float

    def __init__(self, region: DotMap, radius_cm: float = 100.0) -> None:
        self.radius_cm = radius_cm
        self.radius_px = float(region.ArenaOuterRadius)
        self.center_px = _parse_point(region.ArenaCenter)
        self.pixel_to_cm = self.radius_cm / self.radius_px

    def point_to_cm(self, point: NDArray[np.float64] | DataFrame):
        """Converts the coordinates of a point or series of points from pixels to centimeters."""
        return (point - self.center_px) * self.pixel_to_cm


@dataclass
class PatchMetadata:
    """Represents information about the feeder patch location and size."""

    radius_cm: float
    roi_px: NDArray[np.float64]

    def __init__(self, patch_device: DotMap, patch_region: DotMap) -> None:
        self.radius_cm = float(patch_device.Radius)
        self.roi_px = _parse_roi(patch_region)


@dataclass
class NestMetadata:
    """Represents information about the nest location."""

    roi_px: NDArray[np.float64]

    def __init__(self, nest_region: DotMap) -> None:
        self.roi_px = _parse_roi(nest_region)


@dataclass
class RfidMetadata:
    """Represents information about a single RFID antenna."""

    location_px: NDArray[np.float64]

    def __init__(self, rfid_device: DotMap) -> None:
        self.location_px = _parse_point(rfid_device.Location)


@dataclass
class VideoControllerMetadata:
    """Represents information about the sampling frequency of video streams."""

    global_fps: float
    local_fps: float

    def __init__(self, video_controller: DotMap) -> None:
        self.global_fps = float(video_controller.GlobalTriggerFrequency)
        self.local_fps = float(video_controller.LocalTriggerFrequency)


@dataclass
class ExperimentMetadata:
    """Aggregates all metadata information about the experimental environment."""

    arena: ArenaMetadata
    video: VideoControllerMetadata
    patch1: PatchMetadata
    patch2: PatchMetadata
    patch3: PatchMetadata
    nest: NestMetadata
    patch1_rfid: RfidMetadata
    patch2_rfid: RfidMetadata
    patch3_rfid: RfidMetadata
    nest_rfid1: RfidMetadata
    nest_rfid2: RfidMetadata
    gate_rfid: RfidMetadata

    def __init__(self, metadata: DotMap) -> None:
        # read arena metadata
        self.arena = ArenaMetadata(metadata.ActiveRegion)

        # read video metadata
        self.video = VideoControllerMetadata(metadata.Devices.VideoController)

        # read patch metadata
        self.patch1 = PatchMetadata(metadata.Devices.Patch1, metadata.ActiveRegion.Patch1Region)
        self.patch2 = PatchMetadata(metadata.Devices.Patch2, metadata.ActiveRegion.Patch2Region)
        self.patch3 = PatchMetadata(metadata.Devices.Patch3, metadata.ActiveRegion.Patch3Region)

        # read nest metadata
        self.nest = NestMetadata(metadata.ActiveRegion.NestRegion)

        # read RFID metadata
        self.patch1_rfid = RfidMetadata(metadata.Devices.Patch1Rfid)
        self.patch2_rfid = RfidMetadata(metadata.Devices.Patch2Rfid)
        self.patch3_rfid = RfidMetadata(metadata.Devices.Patch3Rfid)
        self.nest_rfid1 = RfidMetadata(metadata.Devices.NestRfid1)
        self.nest_rfid2 = RfidMetadata(metadata.Devices.NestRfid2)
        self.gate_rfid = RfidMetadata(metadata.Devices.GateRfid)


def _parse_point(point: DotMap) -> NDArray[np.float64]:
    return np.array((float(point.X), float(point.Y)))


def _parse_roi(region: DotMap) -> NDArray[np.float64]:
    return np.array([(float(p.X), float(p.Y)) for p in region.ArrayOfPoint])
