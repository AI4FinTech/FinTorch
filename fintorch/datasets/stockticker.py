import logging
import os
import shutil
from datetime import date as Date
from typing import Any, Dict, List, Optional

import lightning as L
import pandas as pd  # type: ignore
import polars as pol
import yfinance as yf  # type: ignore
from neuralforecast.tsdataset import TimeSeriesDataset  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from tenacity import retry, stop_after_attempt, wait_exponential  # type: ignore
from torch.utils.data import DataLoader, Dataset


# TODO: directly subclass TimeSeriesDataset from neural forcast
class StockTicker(Dataset):  # type: ignore
    def __init__(
        self,
        root: str,
        tickers: List[str],
        start_date: Date,
        end_date: Date,
        mapping: Dict[str, str],  # TODO: check dict type
        value_name: str = "Close",
        force_reload: bool = False,
        past_length: int = 10,
        future_length: int = 5,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert isinstance(tickers, list), "tickers must be a list"
        assert isinstance(start_date, Date), "start_date must be a Date object"
        assert isinstance(end_date, Date), "end_date must be a Date object"

        # Check if the start_date is before the end_date
        if start_date >= end_date:
            raise ValueError("start_date must be before end_date.")

        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.value_name = value_name
        self.mapping = mapping
        self.root = os.path.expanduser(root)
        self.past_length = past_length
        self.future_length = future_length
        self.start_idx = start_idx

        logging.info("StockTicker dataset initialization")

        if force_reload or not all(
            os.path.exists(path) for path in self.processed_paths()
        ):
            self.remove_directory(self.root)
            self.setupDirectories()

            # if we want to force reload, or a processed file is missing. Start the processing
            self.download(force_reload)  # download uncached tickers
            self.process()

        self.load()

        self.end_idx = (
            end_idx
            if end_idx is not None
            else len(self.df_timeseries_dataset) - past_length - future_length
        )
        self.length = self.end_idx - self.start_idx

    def remove_directory(self, path: str) -> None:
        """Removes a directory or a file.

        Args:
            path: The path to the directory or file to remove.

        Raises:
            FileNotFoundError: If the directory or file does not exist.
            OSError: If there's an error during the removal process (e.g., permission issues).
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory not found: {path}")
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.isfile(path):
                os.remove(path)
            else:
                raise ValueError(f"Path '{path}' is neither a file nor a directory.")

            print(f"Directory '{path}' and its contents removed successfully.")
        except OSError as e:
            raise OSError(f"Error removing directory '{path}': {e}")

    def setupDirectories(self) -> None:
        """
        Sets up the necessary directories for storing raw and processed data.

        This method checks if the root directory exists, and if not, creates it.
        It also checks if the raw and processed directories exist within the root directory,
        and if not, creates them.

        Args:
            None

        Returns:
            None
        """

        try:
            os.makedirs(self.root, exist_ok=True)
            os.makedirs(os.path.join(self.root, "raw"), exist_ok=True)
            os.makedirs(os.path.join(self.root, "processed"), exist_ok=True)
        except OSError as e:
            logging.error(f"Failed to create directories: {str(e)}")
            raise RuntimeError(f"Failed to setup directories: {str(e)}")

    def __len__(self) -> int:
        """Return the length of the dataset."""
        if not hasattr(self, "timeseries_dataset"):
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return self.length

    def __getitem__(self, idx: int) -> Any:
        """
        Get an item from the dataset

        Args:
            idx: Index of the item to retrieve

        Returns:
            TimeseriesDataset: A slice of stock tick data with spatial graph encoding

        Raises:
            RuntimeError: If the dataset hasn't been loaded
        """
        if not hasattr(self, "timeseries_dataset"):
            raise RuntimeError("Dataset not loaded. Call load() first.")

        if isinstance(idx, slice):
            return [
                self[i]
                for i in range(idx.start or 0, idx.stop or len(self), idx.step or 1)
            ]

        # Adjust index to be within the dataset slice
        idx = self.start_idx + idx

        data = self.df_timeseries_dataset.select("y_scaled").cast(pol.Float32)

        past_data = data[idx : idx + self.past_length].to_torch().float()
        target = (
            data[idx + self.past_length : idx + self.past_length + self.future_length]
            .to_torch()
            .float()
            .squeeze()
        )

        # Create a dictionary for past, future, and static data
        past_inputs = {"past_data": past_data}

        return past_inputs, target

    def raw_file_names(self) -> list[str]:
        """
        Generates a list of raw file names based on the tickers, start date, and end date.

        Returns:
            A list of raw file names in the format "{ticker}_date_range_start_{start_date}_end_{end_date}.csv".
        """
        return [
            f"{ticker}_date_range_start_{self.start_date}_end_{self.end_date}.csv"
            for ticker in self.tickers
        ]

    def raw_paths(self) -> list[str]:
        # Return the full paths to the raw files
        # If you don't need to download raw files, return an empty list
        return [os.path.join(self.root, "raw", name) for name in self.raw_file_names()]

    def processed_paths(self) -> list[str]:
        """
        Returns a list of full paths to the processed files.

        Returns:
            list[str]: A list of full paths to the processed files.
        """
        return [
            os.path.join(self.root, "processed", name)
            for name in self.processed_file_names()
        ]

    def processed_file_names(self) -> list[str]:
        """
        Returns a list of processed file names.

        Returns:
            list[str]: A list of processed file names.
        """
        return [
            "timeseries_stocks_v1_partitioned",
        ]

    @retry(  # type: ignore[misc]
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def download_with_retry(
        self, tickers: List[str], start_date: Date, end_date: Date
    ) -> Any:
        return yf.download(tickers, start=start_date, end=end_date)

    def download(self, force_reload: bool = False) -> None:
        """
        Downloads the raw stock data from Yahoo Finance for the specified tickers and time range.

        Args:
            force_reload (bool, optional): If True, forces the download even if the raw data files already exist.
                Defaults to False.

        Raises:
            ValueError: If the specified value_name does not exist as a column in the raw_data dataframe.
        """
        if not all(os.path.exists(path) for path in self.raw_paths()) or force_reload:
            # Download the raw data for tickers that don't have the raw files
            if force_reload:
                missing_tickers = self.tickers
                logging.info("force reloading stock data from yahoo finance")
            else:
                missing_tickers = [
                    ticker
                    for ticker, path in zip(self.tickers, self.raw_paths())
                    if not os.path.exists(path)
                ]
                logging.info(f"Only downloading missing tickers: {missing_tickers}")

            raw_data = self.download_with_retry(
                missing_tickers, self.start_date, self.end_date
            )

            # Check if self.value_name exists as a column in the raw_data dataframe
            if self.value_name not in raw_data.columns:
                raise ValueError(
                    f"{self.value_name} does not exist as a column in the raw_data dataframe"
                )

            # Reshape the data
            df = raw_data[self.value_name]  # No need to unstack here

            # Convert the Series to a DataFrame if it's not already (optional but recommended)
            if isinstance(df, pd.Series):
                df = df.to_frame()

            # Melt the dataframe to long format
            hist = df.melt(
                ignore_index=False, var_name="Ticker", value_name=self.value_name
            )
            hist.reset_index(inplace=True)

            hist.rename(
                columns={"Date": "ds", "Ticker": "unique_id", self.value_name: "y"},
                inplace=True,
            )

            # Split the hist dataframe based on unique_id
            grouped_data = hist.groupby("unique_id")

            # Save a csv file per unique_id
            for ticker, group in grouped_data:
                file_name = f"{ticker}_date_range_start_{self.start_date}_end_{self.end_date}.csv"
                group.to_csv(os.path.join(self.root, "raw", file_name), index=False)
        else:
            logging.info(
                "Using cached data, if you want to force reload set force_reload=True"
            )

    def process(self) -> None:
        """
        Process the stock ticker data.

        Reads the CSV files provided by `raw_files` using polars library.
        Concatenates the files into one data frame.
        Reshapes the data to wide format based on unique_id.
        Constructs a spatial graph based on the wide data frame.
        Stores the time-series data in a parquet file.
        Stores the spatial graph using torch.save.

        Returns:
            None
        """
        # Read all the csv files provided by raw_files with polars
        dfs = []
        for file_path in self.raw_paths():
            try:
                df = pol.read_csv(file_path)
                if df.is_empty():
                    raise ValueError(f"Empty data in file {file_path}")
            except Exception as e:
                logging.error(f"Failed to process {file_path}: {str(e)}")
                raise

            dfs.append(df)

        # Concatenate the files into one data frame
        concatenated_df = pol.concat(dfs)
        concatenated_df = concatenated_df.with_columns(
            pol.col("unique_id").replace(self.mapping, default=None),
            pol.col("ds").str.to_date(),
        )

        # Reshape the data to wide format based on unique_id
        wide_df = concatenated_df.pivot(index="ds", on="unique_id", values="y")
        wide_df = wide_df.drop(["ds"])

        # Store time-series data
        concatenated_df.write_parquet(
            self.processed_paths()[0],
            compression="zstd",
            use_pyarrow=True,
            pyarrow_options={"partition_cols": ["unique_id"]},
        )

    def load(self) -> None:
        """
        Loads the dataset and associated graphs.

        This method loads the dataset from a parquet file and creates the necessary graphs for further processing.
        It also logs the loaded dataset, spatial graph, and temporal encoding.

        Returns:
            None
        """

        try:
            self.df_timeseries_dataset = pol.read_parquet(self.processed_paths()[0])

            self.timeseries_dataset, _, _, _ = TimeSeriesDataset.from_df(
                self.df_timeseries_dataset,
                id_col="unique_id",
                time_col="ds",
                target_col="y",
            )

            scaler = StandardScaler()
            # Convert the polars Series to a NumPy array before reshaping
            y_values = self.df_timeseries_dataset["y"].to_numpy().reshape(-1, 1)
            # Fit and transform the NumPy array
            scaled_y = scaler.fit_transform(y_values)
            # Convert back to a polars Series (optional, but recommended for consistency)
            self.df_timeseries_dataset = self.df_timeseries_dataset.with_columns(
                pol.Series("y_scaled", scaled_y.flatten())
            )
            self.scaler = scaler

            self.length = len(self.df_timeseries_dataset)

        except Exception as e:
            logging.error(f"Failed to load dataset: {str(e)}")
            raise RuntimeError("Failed to load dataset") from e
        logging.info("All datsets loaded sucessfully")

        logging.info(f"loaded dataset:{self.timeseries_dataset}")


class StockTickerDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        tickers: str,
        start_date: Date,
        end_date: Date,
        ticker_index: Dict[str, str],
        batch_size: int,
        workers: int = 1,
        past_length: int = 10,
        future_length: int = 5,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_index = ticker_index
        self.batch_size = batch_size
        self.workers = workers
        self.past_length = past_length
        self.future_length = future_length
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.force_reload = force_reload

        if isinstance(self.tickers, list):
            if len(self.tickers) > 1:
                raise ValueError(
                    "For the moment we only support one ticker at the time. "
                    "For NeuralForecast a list of tickers is supported, but for other purposes we can only retreive"
                    " a single ticker dataset"
                )
            self.tickers = self.tickers[0]

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = StockTicker(
            self.data_path,
            tickers=[self.tickers],
            start_date=self.start_date,
            end_date=self.end_date,
            mapping=self.ticker_index,
            force_reload=self.force_reload,
            past_length=self.past_length,
            future_length=self.future_length,
            start_idx=self.start_idx,
            end_idx=self.end_idx,
        )

        train_size = int(0.8 * len(self.dataset))
        val_size = int(0.1 * len(self.dataset))

        # TODO: create train_dataset, test_dataset, val_dataset
        self.train_dataset = self.dataset
        self.test_dataset = self.dataset
        self.val_dataset = self.dataset

        # Create separate dataset instances to preserve time-series order
        self.train_dataset = StockTicker(
            self.data_path,
            tickers=[self.tickers],
            start_date=self.start_date,
            end_date=self.end_date,
            mapping=self.ticker_index,
            force_reload=self.force_reload,
            past_length=self.past_length,
            future_length=self.future_length,
            start_idx=0,
            end_idx=train_size,
        )

        self.val_dataset = StockTicker(
            self.data_path,
            tickers=[self.tickers],
            start_date=self.start_date,
            end_date=self.end_date,
            mapping=self.ticker_index,
            force_reload=self.force_reload,
            past_length=self.past_length,
            future_length=self.future_length,
            start_idx=train_size,
            end_idx=train_size + val_size,
        )

        self.test_dataset = StockTicker(
            self.data_path,
            tickers=[self.tickers],
            start_date=self.start_date,
            end_date=self.end_date,
            mapping=self.ticker_index,
            force_reload=self.force_reload,
            past_length=self.past_length,
            future_length=self.future_length,
            start_idx=train_size + val_size,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
        )
