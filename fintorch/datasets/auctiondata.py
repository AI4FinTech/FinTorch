import logging
import os
from datetime import datetime
from typing import Any, List
from zipfile import ZipFile

import polars as pol
import torch
from torch.utils.data import Dataset


class AuctionDataset(Dataset):  # type: ignore
    """
    AuctionDataset Class Documentation

    The `AuctionDataset` class provides a convenient wrapper around the auction dataset used for the NASDAQ daily ten-minute closing auction for the Kaggle competition trading at the close from Optiver. It allows for the automated downloading, processing, and loading of data, enabling efficient handling and use within machine learning workflows, specifically those using PyTorch's Dataset utilities.

    Attributes:
    root (str): The root directory where the dataset will be downloaded, stored, and processed.
    force\_reload (bool): If True, forces a redownload and reprocessing of the dataset even if the required files are already present.

    Methods:
    setupDirectories():
    Creates the necessary directory structure for storing raw and processed data.
    download():
    Downloads the dataset from Kaggle using the Kaggle API.
    process():
    Placeholder for applying transformations to the downloaded data to prepare it for modeling.
    load():
    Loads the processed data for easy access.
    **len**():
    Returns the length of the dataset.
    **getitem**(idx):
    Allows indexing and iteration over data samples.

    Overview:
    The `AuctionDataset` class is designed to load and preprocess historical auction data provided by the Optiver competition on Kaggle. This data is used to predict future price movements of NASDAQ-listed stocks relative to a synthetic index. The dataset includes time series data with multiple features such as stock price, bid/ask information, imbalance data, and reference prices.

    Key Features:
    - Automated Data Download: Automatically downloads the dataset from Kaggle using the Kaggle API. The dataset is sourced from the Optiver "Trading at the Close" competition.
    - Directory Setup: Manages directories for storing both raw and processed data, making it straightforward to maintain organized data structures.
    - Processing Workflow: Provides an interface to transform the raw auction data as needed. This is useful for creating features suitable for training predictive models.
    - Seamless Integration: Extends `torch.utils.data.Dataset`, making it compatible with PyTorch's data loaders and facilitating its integration into machine learning pipelines.

    Usage:
    The class handles the entire lifecycle of the dataset:

    ```
    - Downloading: When instantiated, if required files are missing or if forced reload is requested, the dataset is downloaded directly from Kaggle.
    - Processing: After downloading, the data is processed to apply transformations that make it suitable for model training and evaluation.
    - Loading: Finally, the processed data is loaded into memory, making it available for access by PyTorch.
    ```

    Constructor Parameters:
    root (str): The root directory where the dataset will be downloaded, stored, and processed.
    force_reload (bool, default=False): If True, forces a redownload and reprocessing of the dataset even if the required files are already present.

    Methods Overview:
    setupDirectories():
    Creates the necessary directory structure for storing raw and processed data. Ensures that the root, raw, and processed directories exist.

    ```
    download():
        Uses the Kaggle API to download the dataset. It reads configuration details for authenticating and retrieving files from the Optiver competition.

    process():
        Placeholder function where transformations are applied to the downloaded data to prepare it for modeling.

    load():
        Loads the processed data for easy access when training or evaluating machine learning models.

    __len__() and __getitem__(idx):
        Implementations required by `torch.utils.data.Dataset` to allow indexing and iteration over data samples.
    ```

    Practical Application:
    This class is useful in financial data analysis, specifically for training predictive models using auction data from NASDAQ. It abstracts the tedious parts of dataset handling—such as downloading, file management, and directory creation—allowing data scientists and researchers to focus more on model building and experimentation.


    Train and Test Sets:
    The dataset is split into a training set (`train.csv`) and a test set  (test.csv) . The training set contains historical auction data used for model training, while the test set is used to evaluate model performance. The test set is provided through the competition API and contains similar features but without target labels.

    Competition Data:
    In this competition, Kagglers are challenged to predict the short-term price movements during the 10-minute auction period. All the training data is contained within a single `train.csv` file. Please refer to [https://www.kaggle.com/competitions/optiver-trading-at-the-close/data](https://www.kaggle.com/competitions/optiver-trading-at-the-close/data) & the introduction in the associated notebook for specific column definitions.

    ```
    The dataset includes the following columns:
    - `stock_id`: A unique identifier for the stock. Not all stock IDs exist in every time bucket.
    - `date_id`: A unique identifier for the date. Date IDs are sequential and consistent across all stocks.
    - `imbalance_size`: The amount unmatched at the current reference price (in USD).
    - `imbalance_buy_sell_flag`: An indicator reflecting the direction of auction imbalance.
        - Buy-side imbalance: `1`
        - Sell-side imbalance: `-1`
        - No imbalance: `0`
    - `reference_price`: The price at which paired shares are maximized, the imbalance is minimized, and the distance from the bid-ask midpoint is minimized, in that order. Can also be thought of as equal to the near price bounded between the best bid and ask price.
    - `matched_size`: The amount that can be matched at the current reference price (in USD).
    - `far_price`: The crossing price that will maximize the number of shares matched based on auction interest only, excluding continuous market orders.
    - `near_price`: The crossing price that will maximize the number of shares matched based on auction and continuous market orders.
    - `[bid/ask]_price`: Price of the most competitive buy/sell level in the non-auction book.
    - `[bid/ask]_size`: The dollar notional amount on the most competitive buy/sell level in the non-auction book.
    - `wap`: The weighted average price in the non-auction book, calculated as:

    (BidPrice * AskSize + AskPrice * BidSize) / (BidSize + AskSize)

    - `seconds_in_bucket`: The number of seconds elapsed since the beginning of the day's closing auction, always starting from 0.
    - `target`: The 60-second future move in the `wap` of the stock, less the 60-second future move of the synthetic index. The synthetic index is a custom weighted index of Nasdaq-listed stocks constructed by Optiver for this competition. The unit of the target is basis points, where a 1 basis point price move is equivalent to a 0.01% price move.

    Where `t` is the time at the current observation, we can define the target as:

    Target = ((StockWAP[t+60] / StockWAP[t]) - (IndexWAP[t+60] / IndexWAP[t])) * 10000

    All size-related columns are in USD terms.
    All price-related columns are converted to a price move relative to the stock `wap` (weighted average price) at the beginning of the auction period.
    ```

    Citation:
    Tom Forbes, John Macgillivray, Matteo Pietrobon, Sohier Dane, and Maggie Demkin. Optiver - Trading at the Close. [https://kaggle.com/competitions/optiver-trading-at-the-close](https://kaggle.com/competitions/optiver-trading-at-the-close), 2023. Kaggle.



    """

    def __init__(self, root: str, force_reload: bool = False):
        super().__init__()

        self.root = root

        logging.info("Load auction data")
        self.setupDirectories()

        if force_reload or not all(
            os.path.exists(path) for path in self.processed_paths()
        ):
            # if we want to force reload, or a processed file is missing. Start the processing
            self.download()  # download auction data
            self.process()

        self.load()

    def __len__(self) -> int:
        return self.train.shape[0]

    def __getitem__(self, idx: int) -> Any:
        return torch.tensor(self.train.drop("ds").row(idx), dtype=torch.float32)

    def processed_paths(self) -> List[str]:
        return [
            os.path.join(self.root, path)
            for path in [
                "raw/train.csv",
                "raw/example_test_files/revealed_targets.csv",
                "raw/example_test_files/sample_submission.csv",
                "raw/example_test_files/test.csv",
            ]
        ]

    def process(self) -> None:
        logging.info("Processing: apply transformations to auction data")

    def download(self) -> None:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore

        logging.info("Downloading dataset from Kaggle")
        api = KaggleApi()
        api.read_config_file()
        api.authenticate()
        competition = "optiver-trading-at-the-close"
        api.competition_download_files(
            competition, path=os.path.join(self.root, "raw"), quiet=False
        )

        path = os.path.join(self.root, "raw", competition + ".zip")
        with ZipFile(path) as zObject:
            zObject.extractall(os.path.join(self.root, "raw"))

        os.remove(path)

    def load(self) -> None:
        path_train = os.path.join(self.root, "raw", "train.csv")
        # TODO: Can we replace NaN with 0?
        self.train = pol.read_csv(path_train)

        # Rename to NeuralForecast convention
        self.train = self.train.rename({"target": "y", "stock_id": "unique_id"})

        # TODO: check exclusion, can we fill these columns?
        self.train = self.train.drop(["row_id", "near_price", "far_price"])

        # TODO: check fill strategy, do we have better strategies?
        self.train = self.train.fill_nan(0)
        self.train = self.train.fill_null(0)

        # filter overlap
        # BUG: the train and test set overlap, split properly for time-series analysis
        self.train = self.train.filter(~pol.col("date_id").is_in([478, 479, 480]))
        self.train = self.train.filter(~pol.col("seconds_in_bucket").is_in([540]))

        # map to proper ds
        self.train = self.map_to_datetime(self.train)

        path_test = os.path.join(self.root, "raw", "example_test_files", "test.csv")
        self.test = pol.read_csv(path_test)

        self.test = self.map_to_datetime(self.test)

        # Rename to NeuralForecast convention
        self.test = self.test.rename({"stock_id": "unique_id"})

        # TODO: check exclusion, can we fill these columns?
        self.test = self.test.drop(["near_price", "far_price"])

        # TODO: check fill strategy, do we have better strategies?
        self.test = self.test.fill_nan(0)
        self.test = self.test.fill_null(0)

    def map_to_datetime(self, df: pol.DataFrame) -> pol.DataFrame:
        start_date = datetime(2023, 1, 1)
        return df.with_columns(
            (
                pol.lit(start_date)
                + (pol.col("date_id") * pol.duration(minutes=9))
                + pol.duration(hours=12)
                + (pol.col("seconds_in_bucket") * pol.duration(seconds=1))
            ).alias("ds")
        )

    def setupDirectories(self) -> None:
        try:
            os.makedirs(self.root, exist_ok=True)
            os.makedirs(os.path.join(self.root, "raw"), exist_ok=True)
            os.makedirs(os.path.join(self.root, "processed"), exist_ok=True)
        except OSError as e:
            logging.error(f"Failed to create directories: {str(e)}")
            raise RuntimeError(f"Failed to setup directories: {str(e)}")
