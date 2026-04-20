"""Inference engine with adapter management."""

from dataclasses import dataclass, field

@dataclass
class InferenceConfig:
    model_name: str = "distilgpt2"
    device: str = "cuda"
    max_new_tokens: int = 50
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1

    @classmethod
    def rtx4050_7b(cls):
        return cls(model_name="Qwen/Qwen2.5-7B-Instruct", device="cuda", max_new_tokens=200)

@dataclass
class AdapterSlot:
    name: str
    room: str
    version: str
    loaded: bool = False

@dataclass
class InferenceRequest:
    prompt: str = ""
    room: str = "default"
    max_tokens: int = 50
    temperature: float = 0.7
    system_prompt: str = ""

class InferenceRuntime:
    def __init__(self, config: InferenceConfig = None):
        self.config = config or InferenceConfig()
        self.adapters: dict[str, AdapterSlot] = {}
        self.request_count = 0

    def load_adapter(self, name: str, room: str, version: str = "latest"):
        slot = AdapterSlot(name=name, room=room, version=version, loaded=True)
        self.adapters[f"{room}:{name}"] = slot
        return slot

    def unload_adapter(self, room: str, name: str):
        key = f"{room}:{name}"
        if key in self.adapters:
            self.adapters[key].loaded = False

    def build_request(self, prompt: str, room: str = "default", **kwargs) -> InferenceRequest:
        self.request_count += 1
        return InferenceRequest(
            prompt=prompt, room=room,
            max_tokens=kwargs.get("max_tokens", self.config.max_new_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            system_prompt=kwargs.get("system_prompt", ""),
        )

    def active_adapters(self) -> list[AdapterSlot]:
        return [a for a in self.adapters.values() if a.loaded]

    @property
    def stats(self) -> dict:
        return {"requests": self.request_count, "loaded_adapters": len(self.active_adapters()),
                "total_adapters": len(self.adapters), "config": vars(self.config)}
