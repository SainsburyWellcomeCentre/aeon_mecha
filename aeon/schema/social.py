"""Readers for data relevant to Social experiments."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import aeon.io.reader as _reader
from aeon import util


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

    def read(
        self, file: Path, ceph_proc_dir: str | Path = "/ceph/aeon/aeon/data/processed"
    ) -> pd.DataFrame:
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
                if parts is None:
                    raise KeyError(f"Cannot find bodyparts in {file}.") from err
        return parts


def get_config_file(
    config_file_dir: Path,
    config_file_names: None | list[str] = None,
) -> Path:
    """Returns the config file from a model's config directory."""
    if config_file_names is None:
        config_file_names = ["confmap_config.json"]
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
