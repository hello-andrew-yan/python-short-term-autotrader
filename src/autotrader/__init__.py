import logging

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0-dev"

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ["__version__", "logger"]
