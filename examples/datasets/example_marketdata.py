import logging
from pathlib import Path

from fintorch.datasets.marketdata import MarketDataset

logging.basicConfig(level=logging.INFO)

data_path = Path("~/.fintorch_data/marketdata-janestreet/").expanduser()


market_data = MarketDataset(root=data_path, force_reload=False, split="train")

for _, batch in market_data:
    print("Loading train batch")
    print(batch.describe())

market_data = MarketDataset(root=data_path, force_reload=False, split="test")

for _, batch in market_data:
    print("Loading test batch")
    print(batch.describe())
