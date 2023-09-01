"""Readers for data relevant to Social experiments."""

from pathlib import Path
from typing import List, Union
import json

import pandas as pd

from aeon import util
import aeon.io.reader as _reader


class Pose(_reader.Harp):
    """Reader for Harp-binarized tracking data given a model that outputs id, parts, and likelihoods.

    Columns:
        class (int): Int ID of a subject in the environment.
        class_likelihood (float): Likelihood of the subject's identity.
        part (str): Bodypart on the subject.
        part_likelihood (float): Likelihood of the specified bodypart.
        x (float): X-coordinate of the bodypart.
        y (float): Y-coordinate of the bodypart.
    """
    def __init__(self, pattern: str, extension: str="bin"):
        # `pattern` for this reader should typically be '<hpcnode>_<jobid>*'
        super().__init__(pattern, columns=None, extension=extension)

    def read(self, file: Path, ceph_proc_dir: Path=Path("/ceph/aeon/aeon/data/processed")) -> pd.DataFrame:
        """Reads data from the Harp-binarized tracking file."""
        # Get config file from `file`, then bodyparts from config file.
        model_dir = Path(file.stem.replace("_", "/")).parent
        config_file_dir = ceph_proc_dir / model_dir
        assert config_file_dir.exists(), f"Cannot find model dir {config_file_dir}"
        config_file = get_config_file(config_file_dir)
        parts = self.get_bodyparts(config_file)
        
        # Using bodyparts, assign column names to Harp register values, and read data in default format.
        columns = ["class", "class_likelihood"]
        for part in parts:
            columns.extend([f"{part}_x", f"{part}_y", f"{part}_likelihood"])
        self.columns = columns
        data = super().read(file)
        
        # Set new columns, and reformat `data`.
        n_parts = len(parts)
        part_data_list = [None] * n_parts
        new_columns = ["class", "class_likelihood", "part", "x", "y", "part_likelihood"]
        new_data = pd.DataFrame(columns=new_columns)
        for i, part in enumerate(parts):
            part_columns = ["class", "class_likelihood", f"{part}_x", f"{part}_y", f"{part}_likelihood"]
            part_data = data[part_columns]
            part_data.insert(2, "part", part)
            part_data.columns = new_columns
            part_data_list[i] = part_data
        new_data = pd.concat(part_data_list)
        return new_data.sort_index()

    def get_bodyparts(self, file: Path) -> Union[None, List[str]]:
        """Returns a list of bodyparts from a model's config file."""
        parts = None
        with open(file) as f:
            config = json.load(f)
        if file.stem == "confmap_config":  # SLEAP
            try:
                heads = config["model"]["heads"]
                parts = util.find_nested_key(heads, "part_names")
            except KeyError as err:
                raise KeyError(f"Cannot find bodyparts in {file}.") from err
        return parts


def get_config_file(
    config_file_dir: Path,
    config_file_names: List[str]=[
        "confmap_config.json",  # SLEAP (add others for other trackers to this list)
    ],
):
    """Returns the config file from a model's config directory."""
    config_file = None
    for f in config_file_names:
        if (config_file_dir / f).exists():
            config_file = config_file_dir / f
            break
    assert config_file is not None, f"Cannot find config file in {config_file_dir}"
    return config_file


def class_int2str(data: pd.DataFrame, config_file_dir: Path) -> pd.DataFrame:
    """Converts a class integer in a tracking data dataframe to its associated string (subject id)."""
    config_file = get_config_file(config_file_dir)
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
