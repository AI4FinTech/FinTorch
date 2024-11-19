import logging
from pathlib import Path

from fintorch.datasets.auctiondata import AuctionDataset

logging.basicConfig(level=logging.INFO)

data_path = Path("~/.fintorch_data/auctiondata-optiver/").expanduser()
auction_data = AuctionDataset(data_path, False)

print(f"Length of the dataset:{len(auction_data)} \n Print first 10 records:")

for i in range(10):
    print(auction_data[i])
