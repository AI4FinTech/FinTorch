import logging
import os
import shutil
from datetime import datetime
from typing import Any, Generator, List
from zipfile import ZipFile

import polars as pol
from torch.utils.data import IterableDataset


class MarketDataset(IterableDataset):  # type: ignore
    """PyTorch IterableDataset for Jane Street Market Data.

    Args:
        root (str): Root directory for dataset storage
        split (str, optional): Data split ('train' or 'test'). Defaults to "train"
        force_reload (bool, optional): Force data reprocessing. Defaults to False
        batch_size (int, optional): Number of samples per batch. Defaults to 10000000
    """

    REQUIRED_COLUMNS = {
        "train": {"date_id", "time_id", "symbol_id", "responder_6"},
        "test": {"date_id", "time_id", "symbol_id"},
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        force_reload: bool = False,
        batch_size: int = 100000,
    ):
        super().__init__()
        if split not in ["train", "test"]:
            raise ValueError("split must be either 'train' or 'test'")

        self.root = os.path.expanduser(root)
        self.batch_size = batch_size
        self.offset = 0
        self.split = split  # 'train' or 'test'

        logging.info(f"Load {self.split} market data")
        self.setup_directories()

        if force_reload or not all(os.path.exists(path) for path in self.raw_paths()):
            # If we want to force reload, or a processed file is missing, start the processing
            self.download()  # Download auction data
            self.process()

        if (
            force_reload
            or not os.path.exists(os.path.join(self.root, "processed", self.split))
            or len(os.listdir(os.path.join(self.root, "processed", self.split))) == 0
        ):
            self.process()

        self.load()

    def __iter__(self) -> Generator[tuple[int, pol.DataFrame], Any, None]:
        self.offset = 0
        idx = 0
        while True:
            # Slice the LazyFrame to get the next batch
            batch_df = self.data.slice(self.offset, self.batch_size).collect()

            # If the batch is empty, we've reached the end of the dataset
            if batch_df.is_empty():
                break

            # Yield the batch data
            yield idx, batch_df

            # Update the offset for the next batch
            self.offset += self.batch_size
            idx += 1

    def raw_paths(self) -> List[str]:
        return [
            os.path.join(self.root, path)
            for path in [
                "raw/features.csv",
                "raw/responders.csv",
                "raw/sample_submission.csv",
                "raw/train.parquet",
                "raw/test.parquet",
                "raw/lags.parquet",
            ]
        ]

    def batch_raw(
        self, raw_data: pol.LazyFrame
    ) -> Generator[tuple[int, pol.DataFrame], Any, None]:
        idx = 0
        while True:
            # Slice the LazyFrame to get the next batch
            batch_df = raw_data.slice(self.offset, self.batch_size).collect()

            # If the batch is empty, we've reached the end of the dataset
            if batch_df.is_empty():
                break

            # Yield the batch data
            yield idx, batch_df

            # Update the offset for the next batch
            self.offset += self.batch_size
            idx += 1

    def process(self) -> None:
        logging.info(f"Processing: apply transformations to {self.split} market data")

        try:
            path = os.path.join(self.root, "raw", f"{self.split}.parquet")
            raw = pol.scan_parquet(path)
        except Exception as e:
            raise RuntimeError(f"Failed to read parquet file {path}: {str(e)}") from e

        processed_dir = os.path.join(self.root, "processed", self.split)
        if os.path.exists(processed_dir) and len(os.listdir(processed_dir)) > 0:
            # Clean up old files
            try:
                logging.info("Cleaning up old processed files")
                shutil.rmtree(processed_dir)
                os.mkdir(processed_dir)
            except Exception as e:
                logging.error(f"An error occurred cleaning up the files: {e}")
                raise RuntimeError(
                    f"Failed to clean up processed directory: {str(e)}"
                ) from e

        for idx, batch in self.batch_raw(raw):
            try:
                batch_prep = self.preprocess_batch(batch)
                batch_prep = batch_prep.with_columns(pol.lit(idx).alias("partition_id"))
                batch_prep.write_parquet(
                    processed_dir,
                    partition_by=["partition_id", "unique_id"],
                )
                del batch_prep
            except Exception as e:
                raise RuntimeError(f"Failed to process batch {idx}: {str(e)}") from e

    def download(self) -> None:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore

        logging.info("Downloading dataset from Kaggle")
        try:
            api = KaggleApi()
            api.read_config_file()
            api.authenticate()
        except Exception as e:
            raise RuntimeError(
                "Failed to authenticate with Kaggle API. Ensure you have a valid kaggle.json file. See https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md for more info."
            ) from e

        try:
            competition = "jane-street-real-time-market-data-forecasting"
            api.competition_download_files(
                competition, path=os.path.join(self.root, "raw"), quiet=False
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download compitition files:{str(e)}") from e

        path = os.path.join(self.root, "raw", competition + ".zip")

        try:
            with ZipFile(path) as zObject:
                zObject.extractall(os.path.join(self.root, "raw"))

            os.remove(path)
        except Exception as e:
            raise RuntimeError(f"Failed to extract dataset: {str(e)}") from e

        # Verify all required files are present
        missing_files = [p for p in self.raw_paths() if not os.path.exists(p)]
        if missing_files:
            raise RuntimeError(
                f"Missing required files after download: {missing_files}"
            )

    def load(self) -> None:
        path = os.path.join(self.root, "processed", self.split)
        self.data = pol.scan_parquet(path)

    def preprocess_batch(self, batch: pol.DataFrame) -> pol.DataFrame:
        # Validate required columns
        missing_cols = self.REQUIRED_COLUMNS[self.split] - set(batch.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        df = (
            self.map_to_datetime(batch)
            .fill_nan(0)
            .fill_null(0)
            .rename({"symbol_id": "unique_id"})
        )
        if self.split == "train":
            df = df.rename({"responder_6": "y"})
        return df

    @staticmethod
    def map_to_datetime(df: pol.DataFrame) -> pol.DataFrame:
        start_date = datetime(2023, 1, 1)
        return df.with_columns(
            (
                pol.lit(start_date)
                + (pol.col("date_id") * pol.duration(minutes=9))
                + pol.duration(hours=12)
                + (pol.col("time_id") * pol.duration(seconds=1))
            ).alias("ds")
        )

    def setup_directories(self) -> None:
        try:
            os.makedirs(self.root, exist_ok=True)
            os.makedirs(os.path.join(self.root, "raw"), exist_ok=True)
            os.makedirs(os.path.join(self.root, "processed"), exist_ok=True)
        except OSError as e:
            logging.error(f"Failed to create directories: {str(e)}")
            raise RuntimeError(f"Failed to set up directories: {str(e)}") from e
