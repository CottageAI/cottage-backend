from dataclasses import dataclass


@dataclass
class ChatResponse:
    content: str
    raw: dict


@dataclass
class ModelInfo:
    name: str
    size: str | None = None
    family: str | None = None