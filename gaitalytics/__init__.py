"""gaitalytics is a Python package for gait analysis.

It provides tools for processing and analysing gait data, including segmentation,
normalisation, and feature extraction.
The package is designed to be flexible and extensible, functionality to suit their needs

"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("package-name")
except PackageNotFoundError:
    # package is not installed
    pass
