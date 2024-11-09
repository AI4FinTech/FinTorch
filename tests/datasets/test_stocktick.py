from datetime import date
from pathlib import Path

import pytest

from fintorch.datasets.stockticker import StockTicker


def test_download():
    tickers = ["AAPL", "MSFT", "GOOG"]
    data_path = Path("~/.fintorch_data/stocktickers/").expanduser()
    start_date = date(2015, 1, 1)
    end_date = date(2023, 6, 30)
    ticker_index = {ticker: index for index, ticker in enumerate(tickers)}
    dataset = StockTicker(
        data_path,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        mapping=ticker_index,
        force_reload=True,
    )

    assert dataset.df_timeseries_dataset.count().shape == (1, 3)
    ds, y, unique_id = dataset.df_timeseries_dataset.count()
    assert ds[0] > 0
    assert y[0] > 0
    assert unique_id[0] > 0
    assert ds[0] == y[0]
    assert y[0] == unique_id[0]


def test_no_tickers():
    tickers = "AAPL"
    data_path = Path("~/.fintorch_data/stocktickers/").expanduser()
    start_date = date(2015, 1, 1)
    end_date = date(2023, 6, 30)
    ticker_index = {ticker: index for index, ticker in enumerate(tickers)}
    with pytest.raises(AssertionError) as exc_info:
        StockTicker(
            data_path,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            mapping=ticker_index,
            force_reload=True,
        )
    assert str(exc_info.value) == "tickers must be a list"


def test_wrong_date():
    tickers = ["AAPL"]
    data_path = Path("~/.fintorch_data/stocktickers/").expanduser()
    end_date = date(2015, 1, 1)
    start_date = date(2023, 6, 30)
    ticker_index = {ticker: index for index, ticker in enumerate(tickers)}
    with pytest.raises(ValueError) as exc_info:
        StockTicker(
            data_path,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            mapping=ticker_index,
            force_reload=True,
        )
    assert str(exc_info.value) == "start_date must be before end_date."


def test_wrong_date_type():
    tickers = ["AAPL"]
    data_path = Path("~/.fintorch_data/stocktickers/").expanduser()
    end_date = date(2015, 1, 1)
    start_date = "wrong"
    ticker_index = {ticker: index for index, ticker in enumerate(tickers)}
    with pytest.raises(AssertionError) as exc_info:
        StockTicker(
            data_path,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            mapping=ticker_index,
            force_reload=True,
        )
    assert str(exc_info.value) == "start_date must be a Date object"
