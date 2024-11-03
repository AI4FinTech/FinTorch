import logging
import os
from datetime import date as Date
from typing import List

import lightning.pytorch as pl
import pandas as pd
import polars as pol
import torch
import torch_geometric
import yfinance as yf
from neuralforecast.tsdataset import TimeSeriesDataset, TimeSeriesLoader
from torch.utils.data import Dataset


class StockTicker(Dataset):
    def __init__(
        self,
        root: str,
        tickers: List[str],
        start_date: Date,
        end_date: Date,
        mapping: dict,
        value_name: str = "Adj Close",
        force_reload: bool = False,
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

        logging.info("StockTicker dataset initialization")
        self.setupDirectories()

        if force_reload or not all(
            os.path.exists(path) for path in self.processed_paths()
        ):
            # if we want to force reload, or a processed file is missing. Start the processing
            self.download(force_reload)  # download uncached tickers
            self.process()

        self.load()

    def setupDirectories(self):
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
        # Check if the root directory exists, if not create it
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        # Check if the raw directory exists, if not create it
        raw_dir = os.path.join(self.root, "raw")
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)

        # Check if the processed directory exists, if not create it
        processed_dir = os.path.join(self.root, "processed")
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)

    def __len__(self):
        # The length of the dataset is the sequence length of all stocks
        return len(self.timeseries_dataset)

    def __getitem__(self, idx):
        # We return
        # - stocktick data (TimeseriesDataset), possibly a slice
        # - spatial_graph encoding
        return self.timeseries_dataset.__getitem__(idx)

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
            "spatial_graph_v1.pt",
        ]

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

            raw_data = yf.download(
                missing_tickers, start=self.start_date, end=self.end_date
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
                columns={"Date": "ds", "Ticker": "unique_ticker", self.value_name: "y"},
                inplace=True,
            )

            # Split the hist dataframe based on unique_id
            grouped_data = hist.groupby("unique_ticker")

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
            df = pol.read_csv(file_path)
            dfs.append(df)

        # Concatenate the files into one data frame
        concatenated_df = pol.concat(dfs)
        concatenated_df = concatenated_df.with_columns(
            pol.col("unique_ticker").replace(self.mapping, default=None),
            pol.col("ds").str.to_date(),
        )

        # Reshape the data to wide format based on unique_id
        wide_df = concatenated_df.pivot(index="ds", columns="unique_ticker", values="y")
        wide_df = wide_df.drop(["ds"])

        spatial_graph_timeseries = self.spatial_graph_construction(wide_df)

        # Store time-series data
        concatenated_df.write_parquet(
            self.processed_paths()[0],
            compression="zstd",
            use_pyarrow=True,
            pyarrow_options={"partition_cols": ["unique_ticker"]},
        )

        # Store spatial graph
        torch.save(spatial_graph_timeseries, self.processed_paths()[1])

    def load(self) -> None:
        """
        Loads the dataset and associated graphs.

        This method loads the dataset from a parquet file and creates the necessary graphs for further processing.
        It also logs the loaded dataset, spatial graph, and temporal encoding.

        Returns:
            None
        """

        self.spatial_graph = torch.load(self.processed_paths()[1])

        concatenated_df = pol.read_parquet(self.processed_paths()[0])

        temporal_embedding_graph_embedding = self.temporal_signal_encoding(
            concatenated_df["ds"]
        )

        self.df_timeseries_dataset = concatenated_df

        self.timeseries_dataset, _, _, _ = TimeSeriesDataset.from_df(
            concatenated_df,
            id_col="unique_ticker",
            time_col="ds",
            target_col="y",
            static_df=temporal_embedding_graph_embedding,
        )

        logging.info("All datsets loaded sucessfully")

        logging.info(f"loaded dataset:{self.timeseries_dataset}")

    def spatial_graph_construction(self, df):
        """
        Constructs a spatial graph from a given DataFrame.

        Args:
            df (pandas.DataFrame): The input DataFrame containing the data.

        Returns:
            torch_geometric.data.Data: The constructed spatial graph.

        """
        # Calculate the Spearman correlation matrix
        corr_matrix = df.corr()

        # Get unique column names
        columns = corr_matrix.columns

        # Create an empty list to store results
        results = []

        # Iterate through the lower triangle of the correlation matrix
        for i, row in enumerate(columns):
            for j, col in enumerate(columns):
                if i < j:
                    results.append((row, col, corr_matrix[i, j]))

        result_df = pol.DataFrame(
            {
                "src": [row[0] for row in results],
                "dst": [row[1] for row in results],
                "weight": [row[2] for row in results],
            }
        )

        result_df = result_df.with_columns(
            [
                pol.col("src").cast(pol.Int64),  # or pl.Int32 if IDs are smaller
                pol.col("dst").cast(pol.Int64),
            ]
        )

        # Create a PyG data object from the edge list
        edge_index = torch.tensor(
            result_df.select(["src", "dst"]).to_numpy().T, dtype=torch.long
        )
        edge_attr = torch.tensor(
            result_df["weight"].to_numpy(), dtype=torch.float32
        )  # Adjust dtype if needed

        graph = torch_geometric.data.Data(
            x=df.transpose(include_header=False).to_torch(),
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        return graph

    def temporal_signal_encoding(self, trading_dates: pol.Series):
        """
        Encodes the trading dates into a temporal representation using a one-hot encoding.

        The one-hot encoding is used to represent the temporal information in the model. This method is inspired by the
        work in the following references:

        - Li, Y., Fu, K., Wang, Z., Shahabi, C., Ye, J., & Liu, Y. (2018). Multi-task representation learning for travel
          time estimation. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data
          mining (pp. 1695–1704). doi:10.1145/3219819.3220033.

        - Yuan, H., Li, G., Bao, Z., & Feng, L. (2020). Effective travel time estimation: When historical trajectories
          over road networks matter. In Proceedings of the 2020 ACM SIGMOD international conference on management of data
          (pp. 2135–2149) doi:10.1145/3318464.3389771

        Args:
            trading_dates (pol.Series): A series of trading dates.

        Returns:
            torch.Tensor: A tensor containing the encoded dates, where each date is represented as a one-hot encoding.
        """
        assert isinstance(
            trading_dates, pol.Series
        ), "trading_dates must be a pol.Series"
        assert isinstance(
            trading_dates.dtype, pol.Date
        ), "trading_dates must be a pol.Series with dtype polars.Date"

        encoded_dates_list = []
        for trading_date in trading_dates:
            # Construct graph as a 12 x 21 grid
            # decompose to grid location
            day_of_week_num = trading_date.weekday()  # Monday is 0 and Sunday is 6
            month_num = trading_date.month - 1  # 0 for January, 1 for February, etc.

            # One-hot encode the row and column
            # we have 12-months
            month_encoding = torch.nn.functional.one_hot(
                torch.tensor(month_num), num_classes=12
            )
            # and 21 trading days
            day_encoding = torch.nn.functional.one_hot(
                torch.tensor(day_of_week_num), num_classes=6
            )

            # Concatenate the encodings
            encoding = torch.cat((month_encoding, day_encoding), dim=-1)
            encoded_dates_list.append(encoding)

        encoded_dates_tensor = torch.stack(encoded_dates_list, dim=0)
        encoded_dates = pol.DataFrame(encoded_dates_tensor.numpy())

        return encoded_dates


class StockTickerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tickers: list[str],
        start_date: Date,
        end_date: Date,
        mapping: dict,
        force_reload: bool = False,
    ):
        super().__init__()

        assert start_date < end_date, "start_date must be larger than end_date"

        self.start_date = start_date
        self.end_date = end_date
        self.tickers = tickers
        self.mapping = mapping
        self.force_reload = force_reload

    def setup(self, stage=None):
        self.stocktickerdata = StockTicker(
            "~/.fintorch_data/stocktickers/",
            tickers=self.tickers,
            start_date=self.start_date,
            end_date=self.end_date,
            mapping=self.mapping,
            force_reload=self.force_reload,
        )

    def train_dataloader(self):
        return TimeSeriesLoader(self.stocktickerdata)

    def val_dataloader(self):
        return super().val_dataloader()

    def test_dataloader(self):
        return super().test_dataloader()
