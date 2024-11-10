import logging
from datetime import date
from pathlib import Path

from fintorch.datasets import stockticker

# Set logging level to INFO
logging.basicConfig(level=logging.INFO)

# Parameters
tickers = ["AAPL", "MSFT", "GOOG"]
data_path = Path("~/.fintorch_data/stocktickers/").expanduser()
start_date = date(2015, 1, 1)
end_date = date(2023, 6, 30)

# Create a dictionary mapping from tickers to index
ticker_index = {ticker: index for index, ticker in enumerate(tickers)}

# Load the stock dataset
stockdata = stockticker.StockTicker(
    data_path,
    tickers=tickers,
    start_date=start_date,
    end_date=end_date,
    mapping=ticker_index,
    force_reload=True,
)

print(stockdata.df_timeseries_dataset.to_pandas().describe())
