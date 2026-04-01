from abc import ABC, abstractmethod
from typing import Iterator, Any


class LLMBackend(ABC):
    """Abstract interface for any language model backend."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        **kwargs: Any
    ) -> dict[str, Any]:
        """Return a complete non-streaming response."""
        pass

    @abstractmethod
    def stream_chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        **kwargs: Any
    ) -> Iterator[str]:
        """Yield response text chunks as they arrive."""
        pass

    @abstractmethod
    def list_models(self) -> list[str]:
        """Return available model names."""
        pass

    @abstractmethod
    def health(self) -> bool:
        """Return True if the backend is reachable and healthy."""
        pass
    