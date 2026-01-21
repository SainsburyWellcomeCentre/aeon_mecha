"""Aeon experiment schemas for DataJoint database ingestion."""
from os import PathLike
import pandas as pd
from dotmap import DotMap


social02 = DotMap()
social03 = DotMap()
social04 = DotMap()


# __all__ = ["octagon01", "exp01", "exp02", "social01", "social02", "social03", "social04"]
__all__ = ["social02", "social03", "social04"]
