import requests
from typing import Iterator, Any

from ..base import LLMBackend


class OllamaBackend(LLMBackend):
    def __init__(self, base_url: str = "http://127.0.0.1:11434"):
        self.base_url = base_url.rstrip("/")

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        **kwargs: Any
    ) -> dict[str, Any]:
        """
        Make a non-streaming chat request to Ollama.
        """
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }

        options = kwargs.get("options")
        if options:
            payload["options"] = options

        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()

        return {
            "content": data.get("message", {}).get("content", ""),
            "raw": data,
        }

    def stream_chat(
    self,
    messages: list[dict[str, str]],
    model: str,
    **kwargs: Any
) -> Iterator[dict[str, Any]]:
        """
        Stream events from Ollama and yield normalized event dictionaries.
        """
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
        }

        options = kwargs.get("options")
        if options:
            payload["options"] = options

        with requests.post(url, json=payload, stream=True, timeout=120) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                data = requests.models.complexjson.loads(line)
                content = data.get("message", {}).get("content", "")
                thinking = data.get("message", {}).get("thinking", False)
                tool_calls = data.get("message", {}).get("tool_calls", [])
                
                if thinking:
                    yield {
                        "type": "thinking",
                        "raw": data,
                    }
                
                if tool_calls:
                    for tool_call in tool_calls:
                        yield {
                            "type": "tool_call",
                            "tool_call": tool_call,
                            "raw": data,
                        }
                        
                if content:
                    yield {
                        "type": "text",
                        "content": content,
                        "raw": data,
                    }

                if data.get("done"):
                    yield {
                        "type": "done",
                        "raw": data,
                    }

    def list_models(self) -> list[str]:
        """
        Return the names of locally available Ollama models.
        """
        url = f"{self.base_url}/api/tags"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        models = data.get("models", [])
        return [model["name"] for model in models]

    def health(self) -> bool:
        """
        Simple health check: if Ollama responds to /api/tags, call it healthy.
        """
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return True
        except requests.RequestException:
            return False
        