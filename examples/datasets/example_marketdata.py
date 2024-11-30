import logging
from pathlib import Path

from fintorch.datasets.marketdata import MarketDataset
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)

data_path = Path("~/.fintorch_data/marketdata-janestreet/").expanduser()


market_data = MarketDataset(
    root=data_path, force_reload=False, split="train", batch_size=1024
)
# Batch size is determined by the Market_data
dataloader = DataLoader(market_data, batch_size=None, num_workers=28)

for _, batch in dataloader:
    print("Loading train batch")
    print(batch.describe())

market_data = MarketDataset(root=data_path, force_reload=False, split="test")
# Batch size is determined by the Market_data
dataloader = DataLoader(market_data, batch_size=None, num_workers=10)

for _, batch in dataloader:
    print("Loading test batch")
    print(batch.describe())
