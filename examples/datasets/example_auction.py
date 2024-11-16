import logging
from pathlib import Path

import polars as pl
import torch
from neuralforecast import NeuralForecast
from neuralforecast.losses.numpy import mae, mse
from neuralforecast.models import NBEATS, NHITS, BiTCN, NBEATSx

from fintorch.datasets.auctiondata import AuctionDataset

logging.basicConfig(level=logging.INFO)
torch.set_float32_matmul_precision("medium")

data_path = Path("~/.fintorch_data/auctiondata-optiver/").expanduser()
auction_data = AuctionDataset(data_path, False)


input_size = 30
days = 3
steps_per_day = 55
horizon = days * steps_per_day
max_steps = 10

# Initialize the model
models = [
    NHITS(
        input_size=input_size,
        h=horizon,
        futr_exog_list=[
            "wap",
            "bid_price",
            "ask_price",
        ],
        scaler_type="robust",
        max_steps=max_steps,
    ),
    BiTCN(
        input_size=input_size,
        h=horizon,
        futr_exog_list=[
            "wap",
            "bid_price",
            "ask_price",
        ],
        scaler_type="robust",
        max_steps=max_steps,
    ),
    NBEATS(input_size=input_size, h=horizon, max_steps=max_steps),
    NBEATSx(
        input_size=input_size,
        h=horizon,
        futr_exog_list=[
            "wap",
            "bid_price",
            "ask_price",
        ],
        scaler_type="robust",
        max_steps=max_steps,
    ),
]


# Create a NeuralForecast object
nf = NeuralForecast(models=models, freq="10s")

# Define validation and test sizes
val_size = horizon  # Number of days for validation
test_size = horizon  # Number of days for testing

train_df = auction_data.train.select(
    [
        "y",
        "ds",
        "unique_id",
        "wap",
        "imbalance_size",
        "imbalance_buy_sell_flag",
        "reference_price",
        "matched_size",
        "bid_price",
        "ask_price",
        "ask_size",
    ]
)

# Perform cross-validation
Y_hat_df = nf.cross_validation(
    df=train_df,
    val_size=val_size,
    test_size=test_size,
    n_windows=None,  # Uses expanding window if None
).to_pandas()


# List of model names
model_names = ["NHITS", "BiTCN", "NBEATS", "NBEATSx"]

# Number of unique series
n_series = len(auction_data.train["unique_id"].unique())

days_train = 479

# Iterate over each model to compute and print MSE and MAE
for model_name in model_names:
    # Extract true values and predictions
    y_true = Y_hat_df.y.values
    y_hat = Y_hat_df[model_name].values

    # Reshape to (n_series, n_windows, horizon)
    y_true = y_true.reshape(n_series, -1, horizon)
    y_hat = y_hat.reshape(n_series, -1, horizon)

    # Compute MSE and MAE
    mse_score = mse(y_true, y_hat)
    mae_score = mae(y_true, y_hat)

    # Print results
    print(f"\nModel: {model_name}")
    print(f"MSE: {mse_score}")
    print(f"MAE: {mae_score}")


# Create the future dataframe
# TODO: Make custom, because there is overlap between the train and test set which causes errors
fcsts_df = nf.make_future_dataframe()

selected_data = auction_data.test.select(
    [
        "unique_id",
        "ds",
        "wap",
        "imbalance_size",
        "imbalance_buy_sell_flag",
        "reference_price",
        "matched_size",
        "bid_price",
        "ask_price",
        "ask_size",
        "row_id",
        "time_id",
    ]
)

# BUG: without y-values the code doesn't run, NeuralForecast requires a y column eventhough the documentation
#     states that it is not required for the futr_df column
selected_data = selected_data.with_columns(pl.lit(0).alias("y"))
fcsts_df = nf.predict(futr_df=selected_data)

print("Predictions on the test dataset by all models:")
fcsts_df = fcsts_df.join(selected_data, on=["unique_id", "ds"], how="right")
print(fcsts_df)
