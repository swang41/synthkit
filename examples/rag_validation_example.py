from synthkit import SynthKitConfig, generate_retrieval_validation_set
from synthkit.providers.mock import MockProvider

provider = MockProvider()
config = SynthKitConfig(seed=7)

records = generate_retrieval_validation_set(
    ["Paris is the capital of France and has a population over 2 million."],
    provider=provider,
    config=config,
)

print(records)
