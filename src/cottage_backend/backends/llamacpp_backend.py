import requests
from typing import Iterator, Any

from ..base import LLMBackend


class LlamaCppBackend(LLMBackend):
    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        self.base_url = base_url.rstrip("/")

    def chat(self, messages, model, **kwargs):
        # call llama-server here
        pass

    def stream_chat(self, messages, model, **kwargs) -> Iterator[str]:
        # streaming implementation here
        pass

    def list_models(self) -> list[str]:
        pass

    def health(self) -> bool:
        pass
    