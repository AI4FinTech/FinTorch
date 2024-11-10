import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from fintorch.datasets.stockticker import StockTicker


def create_mock_stock_data():
    import pandas as pd

    data = pd.read_pickle("tests/datasets/testsdata.pickle")
    return data


@patch("fintorch.datasets.stockticker.yf.download")
def test_download(mock_download):
    # Create temporary directory for test data
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_path = Path(tmp_dir)
        tickers = ["AAPL", "MSFT", "GOOG"]
        start_date = date(2015, 1, 1)
        end_date = date(2023, 6, 30)
        ticker_index = {ticker: index for index, ticker in enumerate(tickers)}

        mock_download.return_value = create_mock_stock_data()

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


@patch("fintorch.datasets.stockticker.yf.download")
def test_no_tickers(mock_download):
    # Create temporary directory for test data
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_path = Path(tmp_dir)
        tickers = "AAPL"
        start_date = date(2015, 1, 1)
        end_date = date(2023, 6, 30)
        ticker_index = {ticker: index for index, ticker in enumerate(tickers)}

        mock_download.return_value = create_mock_stock_data()

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


@patch("fintorch.datasets.stockticker.yf.download")
def test_wrong_date(mock_download):
    # Create temporary directory for test data
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_path = Path(tmp_dir)
        tickers = ["AAPL"]
        end_date = date(2015, 1, 1)
        start_date = date(2023, 6, 30)
        ticker_index = {ticker: index for index, ticker in enumerate(tickers)}
        mock_download.return_value = create_mock_stock_data()
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


@patch("fintorch.datasets.stockticker.yf.download")
def test_wrong_date_type(mock_download):
    # Create temporary directory for test data
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_path = Path(tmp_dir)
        tickers = ["AAPL"]
        end_date = date(2015, 1, 1)
        start_date = "wrong"
        ticker_index = {ticker: index for index, ticker in enumerate(tickers)}
        mock_download.return_value = create_mock_stock_data()
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
