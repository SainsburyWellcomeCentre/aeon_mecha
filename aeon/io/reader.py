import datetime
import json
import math
import os

import numpy as np
import pandas as pd
from dotmap import DotMap
from pathlib import Path

from aeon.io.api import chunk_key
from aeon import util

_SECONDS_PER_TICK = 32e-6
_payloadtypes = {
    1: np.dtype(np.uint8),
    2: np.dtype(np.uint16),
    4: np.dtype(np.uint32),
    8: np.dtype(np.uint64),
    129: np.dtype(np.int8),
    130: np.dtype(np.int16),
    132: np.dtype(np.int32),
    136: np.dtype(np.int64),
    68: np.dtype(np.float32),
}


class Reader:
    """Extracts data from raw files in an Aeon dataset.

    Attributes:
        pattern (str): Pattern used to find raw files,
            usually in the format `<Device>_<DataStream>`.
        columns (str or array-like): Column labels to use for the data.
        extension (str): Extension of data file pathnames.
    """

    def __init__(self, pattern, columns, extension):
        self.pattern = pattern
        self.columns = columns
        self.extension = extension

    def read(self, _):
        """Reads data from the specified file."""
        return pd.DataFrame(columns=self.columns, index=pd.DatetimeIndex([]))


class Harp(Reader):
    """Extracts data from raw binary files encoded using the Harp protocol."""

    def __init__(self, pattern, columns, extension="bin"):
        super().__init__(pattern, columns, extension)

    def read(self, file):
        """Reads data from the specified Harp binary file."""
        data = np.fromfile(file, dtype=np.uint8)
        if len(data) == 0:
            return pd.DataFrame(columns=self.columns, index=pd.DatetimeIndex([]))

        stride = data[1] + 2
        length = len(data) // stride
        payloadsize = stride - 12
        payloadtype = _payloadtypes[data[4] & ~0x10]
        elementsize = payloadtype.itemsize
        payloadshape = (length, payloadsize // elementsize)
        seconds = np.ndarray(length, dtype=np.uint32, buffer=data, offset=5, strides=stride)
        ticks = np.ndarray(length, dtype=np.uint16, buffer=data, offset=9, strides=stride)
        seconds = ticks * _SECONDS_PER_TICK + seconds
        payload = np.ndarray(
            payloadshape, dtype=payloadtype, buffer=data, offset=11, strides=(stride, elementsize)
        )

        if self.columns is not None and payloadshape[1] < len(self.columns):
            data = pd.DataFrame(payload, index=seconds, columns=self.columns[: payloadshape[1]])
            data[self.columns[payloadshape[1] :]] = math.nan
            return data
        else:
            return pd.DataFrame(payload, index=seconds, columns=self.columns)


class Chunk(Reader):
    """Extracts path and epoch information from chunk files in the dataset."""

    def __init__(self, reader=None, pattern=None, extension=None):
        if isinstance(reader, Reader):
            pattern = reader.pattern
            extension = reader.extension
        super().__init__(pattern, columns=["path", "epoch"], extension=extension)

    def read(self, file):
        """Returns path and epoch information for the specified chunk."""
        epoch, chunk = chunk_key(file)
        data = {"path": file, "epoch": epoch}
        return pd.DataFrame(data, index=[chunk], columns=self.columns)


class Metadata(Reader):
    """Extracts metadata information from all epochs in the dataset."""

    def __init__(self, pattern="Metadata"):
        super().__init__(pattern, columns=["workflow", "commit", "metadata"], extension="yml")

    def read(self, file):
        """Returns metadata for the specified epoch."""
        epoch_str = file.parts[-2]
        date_str, time_str = epoch_str.split("T")
        time = datetime.datetime.fromisoformat(date_str + "T" + time_str.replace("-", ":"))
        with open(file) as fp:
            metadata = json.load(fp)
        workflow = metadata.pop("Workflow")
        commit = metadata.pop("Commit", pd.NA)
        data = {"workflow": workflow, "commit": commit, "metadata": [DotMap(metadata)]}
        return pd.DataFrame(data, index=[time], columns=self.columns)


class Csv(Reader):
    """Extracts data from comma-separated (csv) text files, where the first column
    stores the Aeon timestamp, in seconds.
    """

    def __init__(self, pattern, columns, dtype=None, extension="csv"):
        super().__init__(pattern, columns, extension)
        self.dtype = dtype

    def read(self, file):
        """Reads data from the specified CSV text file."""
        return pd.read_csv(file, header=0, names=self.columns, dtype=self.dtype, index_col=0)


class Subject(Csv):
    """Extracts metadata for subjects entering and exiting the environment.

    Columns:
        id (str): Unique identifier of a subject in the environment.
        weight (float): Weight measurement of the subject on entering
            or exiting the environment.
        event (str): Event type. Can be one of `Enter`, `Exit` or `Remain`.
    """

    def __init__(self, pattern):
        super().__init__(pattern, columns=["id", "weight", "event"])


class Log(Csv):
    """Extracts message log data.

    Columns:
        priority (str): Priority level of the message.
        type (str): Type of the log message.
        message (str): Log message data. Can be structured using tab
            separated values.
    """

    def __init__(self, pattern):
        super().__init__(pattern, columns=["priority", "type", "message"])


class Heartbeat(Harp):
    """Extract periodic heartbeat event data.

    Columns:
        second (int): The whole second corresponding to the heartbeat, in seconds.
    """

    def __init__(self, pattern):
        super().__init__(pattern, columns=["second"])


class Encoder(Harp):
    """Extract magnetic encoder data.

    Columns:
        angle (float): Absolute angular position, in radians, of the magnetic encoder.
        intensity (float): Intensity of the magnetic field.
    """

    def __init__(self, pattern):
        super().__init__(pattern, columns=["angle", "intensity"])


class Position(Harp):
    """Extract 2D position tracking data for a specific camera.

    Columns:
        x (float): x-coordinate of the object center of mass.
        y (float): y-coordinate of the object center of mass.
        angle (float): angle, in radians, of the ellipse fit to the object.
        major (float): length, in pixels, of the major axis of the ellipse
            fit to the object.
        minor (float): length, in pixels, of the minor axis of the ellipse
            fit to the object.
        area (float): number of pixels in the object mass.
        id (float): unique tracking ID of the object in a frame.
    """

    def __init__(self, pattern):
        super().__init__(pattern, columns=["x", "y", "angle", "major", "minor", "area", "id"])


class BitmaskEvent(Harp):
    """Extracts event data matching a specific digital I/O bitmask.

    Columns:
        event (str): Unique identifier for the event code.
    """

    def __init__(self, pattern, value, tag):
        super().__init__(pattern, columns=["event"])
        self.value = value
        self.tag = tag

    def read(self, file):
        """Reads a specific event code from digital data and matches it to the
        specified unique identifier.
        """
        data = super().read(file)
        data = data[(data.event & self.value) == self.value]
        data["event"] = self.tag
        return data


class DigitalBitmask(Harp):
    """Extracts event data matching a specific digital I/O bitmask.

    Columns:
        event (str): Unique identifier for the event code.
    """

    def __init__(self, pattern, mask, columns):
        super().__init__(pattern, columns)
        self.mask = mask

    def read(self, file):
        """Reads a specific event code from digital data and matches it to the
        specified unique identifier.
        """
        data = super().read(file)
        state = data[self.columns] & self.mask
        return state[(state.diff() != 0).values] != 0


class Video(Csv):
    """Extracts video frame metadata.

    Columns:
        hw_counter (int): Hardware frame counter value for the current frame.
        hw_timestamp (int): Internal camera timestamp for the current frame.
    """

    def __init__(self, pattern):
        super().__init__(pattern, columns=["hw_counter", "hw_timestamp", "_frame", "_path", "_epoch"])
        self._rawcolumns = ["time"] + self.columns[0:2]

    def read(self, file):
        """Reads video metadata from the specified file."""
        data = pd.read_csv(file, header=0, names=self._rawcolumns)
        data["_frame"] = data.index
        data["_path"] = os.path.splitext(file)[0] + ".avi"
        data["_epoch"] = file.parts[-3]
        data.set_index("time", inplace=True)
        return data


class Pose(Harp):
    """Reader for Harp-binarized tracking data given a model that outputs id, parts, and likelihoods.

    Columns:
        class (int): Int ID of a subject in the environment.
        class_likelihood (float): Likelihood of the subject's identity.
        part (str): Bodypart on the subject.
        part_likelihood (float): Likelihood of the specified bodypart.
        x (float): X-coordinate of the bodypart.
        y (float): Y-coordinate of the bodypart.
    """

    def __init__(self, pattern: str, model_root: str = "/ceph/aeon/aeon/data/processed"):
        """Pose reader constructor."""
        # `pattern` for this reader should typically be '<hpcnode>_<jobid>*'
        super().__init__(pattern, columns=None)
        self._model_root = model_root

    def read(self, file: Path) -> pd.DataFrame:
        """Reads data from the Harp-binarized tracking file."""
        # Get config file from `file`, then bodyparts from config file.
        model_dir = Path(*Path(file.stem.replace("_", "/")).parent.parts[1:])
        config_file_dir = Path(self._model_root) / model_dir
        if not config_file_dir.exists():
            raise FileNotFoundError(f"Cannot find model dir {config_file_dir}")
        config_file = self.get_config_file(config_file_dir)
        parts = self.get_bodyparts(config_file)

        # Using bodyparts, assign column names to Harp register values, and read data in default format.
        columns = ["class", "class_likelihood"]
        for part in parts:
            columns.extend([f"{part}_x", f"{part}_y", f"{part}_likelihood"])
        self.columns = columns
        data = super().read(file)

        # Drop any repeat parts.
        unique_parts, unique_idxs = np.unique(parts, return_index=True)
        repeat_idxs = np.setdiff1d(np.arange(len(parts)), unique_idxs)
        if repeat_idxs:  # drop x, y, and likelihood cols for repeat parts (skip first 5 cols)
            init_rep_part_col_idx = (repeat_idxs - 1) * 3 + 5
            rep_part_col_idxs = np.concatenate([np.arange(i, i + 3) for i in init_rep_part_col_idx])
            keep_part_col_idxs = np.setdiff1d(np.arange(len(data.columns)), rep_part_col_idxs)
            data = data.iloc[:, keep_part_col_idxs]
            parts = unique_parts

        # Set new columns, and reformat `data`.
        n_parts = len(parts)
        part_data_list = [pd.DataFrame()] * n_parts
        new_columns = ["class", "class_likelihood", "part", "x", "y", "part_likelihood"]
        new_data = pd.DataFrame(columns=new_columns)
        for i, part in enumerate(parts):
            part_columns = ["class", "class_likelihood", f"{part}_x", f"{part}_y", f"{part}_likelihood"]
            part_data = pd.DataFrame(data[part_columns])
            part_data.insert(2, "part", part)
            part_data.columns = new_columns
            part_data_list[i] = part_data
        new_data = pd.concat(part_data_list)
        return new_data.sort_index()

    def get_class_names(self, file: Path) -> list[str]:
        """Returns a list of classes from a model's config file."""
        classes = None
        with open(file) as f:
            config = json.load(f)
        if file.stem == "confmap_config":  # SLEAP
            try:
                heads = config["model"]["heads"]
                classes = util.find_nested_key(heads, "class_vectors")["classes"]
            except KeyError as err:
                if not classes:
                    raise KeyError(f"Cannot find class_vectors in {file}.") from err
        return classes

    def get_bodyparts(self, file: Path) -> list[str]:
        """Returns a list of bodyparts from a model's config file."""
        parts = []
        with open(file) as f:
            config = json.load(f)
        if file.stem == "confmap_config":  # SLEAP
            try:
                heads = config["model"]["heads"]
                parts = [util.find_nested_key(heads, "anchor_part")]
                parts += util.find_nested_key(heads, "part_names")
            except KeyError as err:
                if not parts:
                    raise KeyError(f"Cannot find bodyparts in {file}.") from err
        return parts

    @classmethod
    def get_config_file(
        cls,
        config_file_dir: Path,
        config_file_names: None | list[str] = None,
    ) -> Path:
        """Returns the config file from a model's config directory."""
        if config_file_names is None:
            config_file_names = ["confmap_config.json"]  # SLEAP (add for other trackers to this list)
        config_file = None
        for f in config_file_names:
            if (config_file_dir / f).exists():
                config_file = config_file_dir / f
                break
        if config_file is None:
            raise FileNotFoundError(f"Cannot find config file in {config_file_dir}")
        return config_file

    @classmethod
    def class_int2str(cls, data: pd.DataFrame, config_file_dir: Path) -> pd.DataFrame:
        """Converts a class integer in a tracking data dataframe to its associated string (subject id)."""
        config_file = cls.get_config_file(config_file_dir)
        if config_file.stem == "confmap_config":  # SLEAP
            with open(config_file) as f:
                config = json.load(f)
            try:
                heads = config["model"]["heads"]
                classes = util.find_nested_key(heads, "classes")
            except KeyError as err:
                raise KeyError(f"Cannot find classes in {config_file}.") from err
            for i, subj in enumerate(classes):
                data.loc[data["class"] == i, "class"] = subj
        return data


def from_dict(data, pattern=None):
    reader_type = data.get("type", None)
    if reader_type is not None:
        kwargs = {k: v for k, v in data.items() if k != "type"}
        return globals()[reader_type](pattern=pattern, **kwargs)

    return DotMap(
        {k: from_dict(v, f"{pattern}_{k}" if pattern is not None else k) for k, v in data.items()}
    )


def to_dict(dotmap):
    if isinstance(dotmap, Reader):
        kwargs = {k: v for k, v in vars(dotmap).items() if k not in ["pattern"] and not k.startswith("_")}
        kwargs["type"] = type(dotmap).__name__
        return kwargs
    return {k: to_dict(v) for k, v in dotmap.items()}
