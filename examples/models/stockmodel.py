from datetime import date
from pathlib import Path

import torch
from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoNBEATS, AutoNHITS, AutoTFT
from neuralforecast.losses.numpy import mae, mse
from neuralforecast.losses.pytorch import MAE, HuberLoss
from ray import tune

from fintorch.datasets import stockticker

torch.set_float32_matmul_precision("medium")

tickers = ["AAPL", "MSFT", "GOOG"]
data_path = Path("~/.fintorch_data/stocktickers/").expanduser()
start_date = date(2015, 1, 1)
end_date = date(2023, 6, 30)

# Create a dictionary mapping from tickers to index
ticker_index = {ticker: index for index, ticker in enumerate(tickers)}

# Auto model settings
horizon = 96
num_samples = 10
freq = "1d"


def main():
    max_steps = tune.choice([500, 1500, 2000])

    tft_config = {
        "input_size": tune.choice([10, 20, 30]),
        "hidden_size": tune.choice([64, 128, 256]),
        "n_head": tune.choice([4, 8]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice([None, "robust", "standard"]),
        "max_steps": max_steps,
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "random_seed": tune.randint(1, 20),
    }

    nbeats_config = {
        "input_size": tune.choice([10, 20, 30]),
        "n_harmonics": tune.choice([2, 3, 4]),
        "n_polynomials": tune.choice([1, 2, 3]),
        "stack_types": tune.choice([["trend", "seasonality"]]),
        "n_blocks": tune.choice([[3, 3], [4, 4], [5, 5]]),
        "mlp_units": tune.choice(
            [
                [[1024, 2048], [2048, 2048]],
                [[2048, 4096], [4096, 4096]],
                [[2048, 4096], [4096, 8192]],
            ]
        ),
        "dropout_prob_theta": 0.0,
        "activation": tune.choice(["ReLU", "LeakyReLU"]),
        "shared_weights": tune.choice([True, False]),
        "valid_loss": None,
        "max_steps": max_steps,
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "num_lr_decays": tune.choice([2, 3, 4]),
        "early_stop_patience_steps": -1,
        "val_check_steps": 100,
        "batch_size": 1024,
        "valid_batch_size": None,
        "windows_batch_size": 1024,
        "inference_windows_batch_size": -1,
        "start_padding_enabled": False,
        "step_size": 1,
        "scaler_type": "identity",
        "random_seed": 1,
        "num_workers_loader": 0,
    }

    nhits_config = {
        # "learning_rate": tune.choice([1e-3]),                                     # Initial Learning rate
        "learning_rate": tune.loguniform(1e-5, 5e-3),
        "max_steps": max_steps,  # Number of SGD steps
        "input_size": tune.choice([7 * horizon]),  # input_size = multiplier * horizon
        "batch_size": tune.choice([7]),  # Number of series in windows
        "windows_batch_size": tune.choice([256]),  # Number of windows in batch
        "n_pool_kernel_size": tune.choice(
            [[2, 2, 2], [16, 8, 1]]
        ),  # MaxPool's Kernelsize
        "n_freq_downsample": tune.choice(
            [[(96 * 7) // 2, 96 // 2, 1], [(24 * 7) // 2, 24 // 2, 1], [1, 1, 1]]
        ),  # Interpolation expressivity ratios
        "dropout_prob_theta": tune.choice([0.5]),  # Dropout regularization
        "activation": tune.choice(["ReLU"]),  # Type of non-linear activation
        "n_blocks": tune.choice([[1, 1, 1]]),  # Blocks per each 3 stacks
        "mlp_units": tune.choice(
            [[[512, 512], [512, 512], [512, 512]]]
        ),  # 2 512-Layers per block for each stack
        "interpolation_mode": tune.choice(
            ["linear"]
        ),  # Type of multi-step interpolation
        "val_check_steps": tune.choice([100]),  # Compute validation every 100 epochs
        "random_seed": tune.randint(1, 10),
    }

    models = [
        AutoTFT(
            h=horizon,
            loss=HuberLoss(delta=0.5),
            valid_loss=MAE(),
            config=tft_config,
            num_samples=num_samples,
            refit_with_val=True,
        ),
        AutoNBEATS(
            h=horizon,
            loss=HuberLoss(delta=0.5),
            valid_loss=MAE(),
            config=nbeats_config,
            num_samples=num_samples,
            refit_with_val=True,
        ),
        AutoNHITS(
            h=horizon,
            loss=HuberLoss(delta=0.5),
            valid_loss=MAE(),
            config=nhits_config,
            num_samples=num_samples,
            refit_with_val=True,
        ),
    ]

    stockdata = stockticker.StockTicker(
        data_path,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        mapping=ticker_index,
        force_reload=True,
    )

    n_time = len(stockdata.df_timeseries_dataset["ds"].unique())
    val_size = int(n_time * 0.1)
    test_size = int(n_time * 0.1)

    nf = NeuralForecast(models=models, freq=freq)
    Y_hat_df = nf.cross_validation(
        df=stockdata.df_timeseries_dataset,
        val_size=val_size,
        test_size=test_size,
        n_windows=None,
    ).to_pandas()

    # List of model names
    model_names = ["AutoTFT", "AutoNHITS", "AutoNBEATS"]

    # Number of unique series
    n_series = len(stockdata.df_timeseries_dataset["unique_id"].unique())

    # Iterate over each model to compute and print MSE and MAE
    for model_name in model_names:
        # Extract true values and predictions
        y_true = Y_hat_df.y.values
        y_hat = Y_hat_df[model_name].values

        # Reshape to (n_series, n_windows, horizon)
        y_true = y_true.reshape(n_series, -1, horizon)
        y_hat = y_hat.reshape(n_series, -1, horizon)

        # Compute MSE and MAE
        mse_score = mse(y_hat, y_true)
        mae_score = mae(y_hat, y_true)

        # Print results
        print(f"\nModel: {model_name}")
        print(f"MSE: {mse_score}")
        print(f"MAE: {mae_score}")


if __name__ == "__main__":
    main()
