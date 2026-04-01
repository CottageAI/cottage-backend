from .base import LLMBackend
from .factory import create_backend
from importlib.metadata import version, PackageNotFoundError

__all__ = ["LLMBackend", "create_backend"]

try:
    __version__ = version("cottage-memory")
except PackageNotFoundError:
    __version__ = "0.0.0"
