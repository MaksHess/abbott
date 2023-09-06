"""3D image processing."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("abbott")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Max Hess"
__email__ = "max.hess@mls.uzh.ch"
