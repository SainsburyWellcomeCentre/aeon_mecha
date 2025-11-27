from datetime import datetime
import json
import pandas as pd
from dotmap import DotMap

from swc.aeon.io import reader
from swc.aeon.schema import Device, Stream, StreamGroup, core


class Metadata(reader.Reader):
    """Extracts metadata information from all epochs in the dataset."""

    def __init__(self, pattern="Metadata"):
        """Initialize the object with the specified pattern."""
        super().__init__(pattern, columns=["workflow", "commit", "metadata"], extension="json")

    def read(self, file):
        """Returns metadata for the epoch associated with the specified file."""
        epoch_str = file.parts[-2]
        date_str, time_str = epoch_str.split("T")
        time = datetime.fromisoformat(date_str + "T" + time_str.replace("-", ":"))
        with open(file) as fp:
            metadata = json.load(fp)
        workflow = metadata.pop("workflow")
        commit = metadata.pop("commit", pd.NA)
        data = {"workflow": workflow, "commit": commit, "metadata": [metadata]}
        return pd.DataFrame(data, index=pd.Series(time), columns=self.columns)
