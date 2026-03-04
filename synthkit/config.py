from dataclasses import dataclass


@dataclass(slots=True)
class SynthKitConfig:
    model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    max_new_tokens: int = 64
    temperature: float = 0.7
    seed: int = 42
