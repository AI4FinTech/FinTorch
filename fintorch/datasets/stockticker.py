import os
from datetime import date as Date
from typing import Callable, List, Optional

import lightning.pytorch as pl
import pandas as pd
import polars as pol
import torch
import yfinance as yf
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import Node2Vec


class StockTicker(InMemoryDataset):
    def __init__(
        self,
        root: str,
        tickers: List[str],
        start_date: Date,
        end_date: Date,
        value_name: str = "Adj Close",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ):
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

        super().__init__(
            root, transform, pre_transform, pre_filter, force_reload=force_reload
        )

    @property
    def raw_file_names(self):
        return [
            f"{ticker}_date_range_start_{self.start_date}_end_{self.end_date}.csv"
            for ticker in self.tickers
        ]

    @property
    def raw_paths(self):
        # Return the full paths to the raw files
        # If you don't need to download raw files, return an empty list
        return [os.path.join(self.root, name) for name in self.raw_file_names]

    @property
    def processed_file_names(self):
        return [
            "timeseries_stocks_v1.pt",
            "spatial_graph_v1.pt",
            "temporal_graph_v1.pt",
        ]

    def download(self):
        # TODO: improve by only downloading the data for tickers where we do not have the cached files.
        raw_data = yf.download(self.tickers, start=self.start_date, end=self.end_date)

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
            file_name = (
                f"{ticker}_date_range_start_{self.start_date}_end_{self.end_date}.csv"
            )
            group.to_csv(os.path.join(self.root, file_name), index=False)

    def process(self) -> None:
        # Read all the csv files provided by raw_files with polars
        dfs = []
        for file_path in self.raw_paths:
            df = pol.read_csv(file_path)
            dfs.append(df)

        # Concatenate the files into one data frame
        concatenated_df = pol.concat(dfs)

        # Reshape the data to wide format based on unique_id
        wide_df = concatenated_df.pivot(index="ds", columns="unique_id", values="y")
        wide_df = wide_df.drop(columns=["ds"])

        self.timeseries()
        self.spatial_graph_construction(wide_df)
        self.temporal_graph_construction()

        return super().process()

    def timeseries(self):
        pass

    def spatial_graph_construction(self, df):
        """ """
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

        ticker_to_idx = {src: idx for idx, src in enumerate(self.tickers)}
        print(ticker_to_idx)
        result_df = result_df.with_columns(
            pol.col("dst").map_dict(ticker_to_idx, default=pol.col("dst")),
            pol.col("src").map_dict(ticker_to_idx, default=pol.col("src")),
        )
        result_df = result_df.with_columns(
            [
                pol.col("src").cast(pol.Int64),  # or pl.Int32 if IDs are smaller
                pol.col("dst").cast(pol.Int64),
            ]
        )

        print(result_df.select(["src", "dst"]).dtypes)

        # Create a PyG data object from the edge list
        edge_index = torch.tensor(
            result_df.select(["src", "dst"]).to_numpy().T, dtype=torch.long
        )
        edge_attr = torch.tensor(
            result_df["weight"].to_numpy(), dtype=torch.float32
        )  # Adjust dtype if needed
        data = Data(edge_index=edge_index, edge_attr=edge_attr)

        # TODO: Write as HeteroData
        # HeteroData.timeseries = [tickes x series]
        # HeteroData (src, spatial, dst).edge_index = Spatial graph
        # HeteroData (src, temporal, dst).edge_index = Temporal graph

        data.x = []  # TODO: time-series of each

        device = "cuda" if torch.cuda.is_available() else "cpu"
        data.to(device)

        # Run a simple graph neural network on the data to encode the data
        model = Node2Vec(
            data.edge_index,
            embedding_dim=128,
            walk_length=20,
            context_size=10,
            walks_per_node=10,
            num_negative_samples=1,
            num_nodes=3,
            p=1.0,
            q=1.0,
            sparse=True,
        ).to(device)

        loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

        def train():
            model.train()
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        for epoch in range(1, 101):
            loss = train()
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

    def temporal_graph_construction(self):
        """

        References:
        - Li, Y., Fu, K., Wang, Z., Shahabi, C., Ye, J., & Liu, Y. (2018).  Multi-task representation learning for
        travel time estimation.
        In Proceedings  of  the  24th  ACM  SIGKDD  international  conference  on  knowledge  discovery  &
        data mining (pp. 1695–1704). doi:10.1145/3219819.3220033.

        - Yuan, H., Li, G., Bao, Z., & Feng, L. (2020). Effective travel time estimation:
          When historical trajectories over road networks matter.
        In Proceedings  of  the  2020  acm  sigmod  international  conference  on  management  of  data (pp. 2135–2149)
        doi:10.1145/3318464.3389771

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """


class StockTickerDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def setup(self, stage=None):
        raise NotImplementedError

    def train_dataloader(self):
        return super().train_dataloader()

    def val_dataloader(self):
        return super().val_dataloader()

    def test_dataloader(self):
        return super().test_dataloader()
