from cottage_backend.backends.ollama_backend import OllamaBackend
from cottage_backend.backends.llamacpp_backend import LlamaCppBackend


def create_backend(kind: str, base_url: str):
    if kind == "ollama":
        return OllamaBackend(base_url=base_url)
    elif kind == "llamacpp":
        return LlamaCppBackend(base_url=base_url)
    else:
        raise ValueError(f"Unknown backend type: {kind}")
    