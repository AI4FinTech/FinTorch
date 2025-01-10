import logging
from pathlib import Path

from fintorch.datasets.invoice import InvoiceDataset

logging.basicConfig(level=logging.INFO)

data_path = Path("~/.fintorch_data/invoice-data/").expanduser()
auction_data = InvoiceDataset(data_path, force_reload=False)

print(f"Length of the dataset:{len(auction_data)} \n Print first 10 records:")

for i in range(10):
    print(auction_data[i])
