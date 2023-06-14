"""Readers for data relevant to Social experiments."""

import json

import aeon.io.reader as _reader


class Sleap(_reader.Harp):
    def __init__(self, model_path):
        """Initializes a reader for Harp-binarized Sleap tracking data given a Sleap network model path."""
        columns = ["class", "class_likelihood"]
        # Get bodyparts from Sleap config.
        with open(model_path) as f:
            config = json.load(f)
        bodyparts = config["model"]["heads"]["multi_class_topdown"]["confmaps"]["bodyparts"]
        for part in bodyparts:
            columns.extend([f"{part}_x", f"{part}_y", f"{part}_likelihood"])
        super().__init__(pattern="", columns=columns)


def sleap_class_int2str(tracking_df, model_path):
    """Converts a SLEAP class integer in a tracking data dataframe to its associated string (subject id)."""
    with open(model_path) as f:
        config = json.load(f)
    class_names = config["model"]["heads"]["multi_class_topdown"]["class_vectors"]["classes"]
    for i, subj in enumerate(class_names):
        tracking_df.loc[tracking_df["class"] == i, "class"] = subj
