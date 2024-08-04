import logging
from datetime import date

from fintorch.datasets import stockticker

# Set logging level to INFO
logging.basicConfig(level=logging.INFO)

# Load the stock dataset
tickers = ["AAPL", "MSFT", "GOOG"]

# Create a dictionary mapping from tickers to index
ticker_index = {ticker: index for index, ticker in enumerate(tickers)}

# Load the stock dataset
stockdata = stockticker.StockTicker(
    "~/.fintorch_data/stocktickers/",
    tickers=tickers,
    start_date=date(2015, 1, 1),
    end_date=date(2023, 6, 30),
    mapping=ticker_index,
    force_reload=True,
)
