from .base import LLMBackend
from .factory import create_backend
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("cottage-backend")
except PackageNotFoundError:
    __version__ = "0.0.0"
    
__all__ = ["LLMBackend", "create_backend"]
