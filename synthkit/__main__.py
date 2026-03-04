from synthkit import SynthKitConfig, augment_texts
from synthkit.providers.mock import MockProvider


if __name__ == "__main__":
    cfg = SynthKitConfig()
    provider = MockProvider()
    result = augment_texts(
        ["The quick brown fox jumps over the lazy dog."],
        operation="paraphrase",
        provider=provider,
        config=cfg,
    )
    print(result[0]["augmented_text"])
