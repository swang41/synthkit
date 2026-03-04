from synthkit import SynthKitConfig, augment_texts
from synthkit.providers.mock import MockProvider

provider = MockProvider()
config = SynthKitConfig(seed=7)

records = augment_texts(
    ["Our product improves developer productivity."],
    operation="rewrite in a casual tone",
    provider=provider,
    config=config,
)

print(records)
