"""
    Setup file for aeon.
    Use setup.cfg to configure project.
"""
from setuptools import setup

if __name__ == "__main__":
    try:
        setup()
    except:  # noqa
        print(
            "\n\nAn error occurred while building the project, "
            "please ensure you have the most updated version of setuptools, "
            "and wheel with either of the following:\n"
            "   conda install setuptools wheel\n"
            "   pip install -U setuptools wheel\n\n"
        )
        raise
