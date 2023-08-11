"""Readers for data relevant to Social experiments."""

import json
from pathlib import Path

import pandas as pd

import aeon.io.reader as _reader


class Pose(_reader.Harp):
    """Reader for Harp-binarized tracking data given a model that outputs id, parts, and likelihoods."""
    def __init__(self, pattern):
        self.pattern = pattern
        self.extension = "bin"

    def read(self, file):
        """
        Reads data from the Harp-binarized tracking file
        """
        # Get config file from `file`, then bodyparts from config file.
        ceph_proc_dir = "/ceph/aeon/aeon/data/processed"
        model_dir = file.stem.replace("_", "/")
        config_file_dir = Path(ceph_proc_dir + "/" + model_dir)
        assert config_file_dir.exists(), f"Cannot find model dir {config_file_dir}"
        config_file = get_config_file(config_file_dir)
        parts = self.get_bodyparts(config_file)
        if parts is None:
            raise ValueError(f"Cannot find bodyparts in {config_file}.")
        # With bodyparts, set columns as set by Bonsai, and read data in default format.
        columns = ["class", "class_likelihood"]
        for part in parts:
            columns.extend([f"{part}_x", f"{part}_y", f"{part}_likelihood"])
        self.columns = columns
        data = super().read(file)
        # Set new columns, and reformat `data`.
        columns = ["class", "class_likelihood", "part", "part_likelihood", "x", "y"]
        new_data = pd.DataFrame(columns=columns)
        for part in parts:
            part_data = data[["class", "class_likelihood", f"{part}_likelihood", f"{part}_x", f"{part}_y"]]
            part_data.insert(2, "part", part)
            part_data.columns = columns
            new_data = pd.concat([new_data, part_data])
        return new_data.sort_index()
       
    def get_bodyparts(file):
        """Returns a list of bodyparts from a model's config file."""
        parts = None
        if file.stem == "confmap_config":  # SLEAP
            with open(file) as f:
                config = json.load(f)
            try:
                heads = config["model"]["heads"]
                for model in heads.keys():
                    if heads[model] is not None:
                        parts = heads[model]["confmaps"]["part_names"]
                        break
            except KeyError as err:
               raise KeyError(f"Cannot find bodyparts in {file}.") from err
        return parts


def get_config_file(
    config_file_dir,
    config_file_names=[
        "confmap_config.json",  # SLEAP (add others for other trackers to this list)
    ]
):
    """Returns the config file from a model's config directory."""
    config_file = None
    for f in config_file_names:
        if (config_file_dir / f).exists():
            config_file = config_file_dir / f
            break
    assert config_file is not None, f"Cannot find config file in {config_file_dir}"
    return config_file

def class_int2str(tracking_df, config_file_dir):
    """Converts a class integer in a tracking data dataframe to its associated string (subject id)."""
    config_file = get_config_file(config_file_dir)
    if config_file.stem == "confmap_config":  # SLEAP
        with open(config_file) as f:
            config = json.load(f)
        try:
            heads = config["model"]["heads"]
            for model in heads.keys():
                if heads[model] is not None:
                    classes = heads[model]["class_vectors"]["classes"]
                    break
            for i, subj in enumerate(classes):
                tracking_df.loc[tracking_df["class"] == i, "class"] = subj
            return tracking_df
        except KeyError as err:
            raise KeyError(f"Cannot find classes in {config_file}.") from err
