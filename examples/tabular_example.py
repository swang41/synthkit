from synthkit import SynthKitConfig, synthesize_rows
from synthkit.providers.mock import MockProvider

provider = MockProvider()
config = SynthKitConfig(seed=7)

rows = synthesize_rows(
    {"name": "string", "age": "int", "city": "string"},
    num_rows=2,
    provider=provider,
    config=config,
    requirements="age between 18 and 65; city from US major cities",
)

print(rows)
