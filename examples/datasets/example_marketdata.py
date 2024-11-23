import logging
from pathlib import Path


from fintorch.datasets.marketdata import MarketDataset

logging.basicConfig(level=logging.INFO)

data_path = Path("~/.fintorch_data/marketdata-janestreet/").expanduser()


market_data = MarketDataset(
    root=data_path,
    force_reload=False,
)

# market_data.process()

for _, batch in market_data:
    print("Loading batch")
    print(batch.describe())
