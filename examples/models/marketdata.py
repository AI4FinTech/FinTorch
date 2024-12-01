import csv
import logging

import torch
from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoNBEATS, AutoNHITS, AutoTFT
from neuralforecast.losses.numpy import mae, mse
from neuralforecast.losses.pytorch import MAE, HuberLoss
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback

from fintorch.datasets.marketdata import MarketDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
torch.set_float32_matmul_precision("medium")

input_size = 2  # Number of past time steps used for prediction
days = 1  # Number of days to forecast
steps_per_day = 5  # Number of steps per day
horizon = days * steps_per_day  # Forecast horizon
max_steps = 1000  # Max training steps

# Define validation and test sizes
val_size = horizon  # Validation set size
test_size = horizon  # Test set size

num_samples = 10  # Number of samples for hyperparameter tuning

windows_batch_size = 16
valid_batch_size = 16
batch_size = 1

# Define configurations for Auto models
tft_config = {
    "input_size": tune.choice([2, 4, 6]),
    "hidden_size": tune.choice([64, 128, 256]),
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "scaler_type": tune.choice(["robust", "standard"]),
    "max_steps": max_steps,
    "batch_size": batch_size,
    "windows_batch_size": windows_batch_size,
    "valid_batch_size": valid_batch_size,
    "random_seed": tune.randint(1, 20),
    "futr_exog_list": ["feature_00", "feature_01", "feature_02"],
}

nbeats_config = {
    "input_size": tune.choice([2, 4, 6]),
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "max_steps": max_steps,
    "batch_size": batch_size,
    "valid_batch_size": valid_batch_size,
    "windows_batch_size": windows_batch_size,
    "random_seed": tune.randint(1, 20),
}

nhits_config = {
    "input_size": tune.choice([2, 4, 6]),
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "max_steps": max_steps,
    "batch_size": batch_size,
    "valid_batch_size": valid_batch_size,
    "windows_batch_size": windows_batch_size,
    "random_seed": tune.randint(1, 20),
    "futr_exog_list": ["feature_00", "feature_01", "feature_02"],
}

# Set up WandbLoggerCallback
wandb_callback_tft = WandbLoggerCallback(
    project="jane-tft",  # Replace with your W&B project name
    entity="ai4fintech",  # Replace with your W&B username or team name
    log_config=True,
)

wandb_callback_nbeats = WandbLoggerCallback(
    project="jane-nbeats",  # Replace with your W&B project name
    entity="ai4fintech",  # Replace with your W&B username or team name
    log_config=True,
)

wandb_callback_nhits = WandbLoggerCallback(
    project="jane-nhits",  # Replace with your W&B project name
    entity="ai4fintech",  # Replace with your W&B username or team name
    log_config=True,
)

# Define ray_tune_kwargs with the Wandb callback
ray_tune_kwargs_tft = [wandb_callback_tft]
ray_tune_kwargs_nbeats = [wandb_callback_nbeats]
ray_tune_kwargs_nhits = [wandb_callback_nhits]

# Initialize the models
models = [
    AutoTFT(
        h=horizon,
        loss=HuberLoss(delta=0.5),
        valid_loss=MAE(),
        config=tft_config,
        num_samples=num_samples,
        refit_with_val=True,
        callbacks=ray_tune_kwargs_tft,
    ),
    AutoNBEATS(
        h=horizon,
        loss=HuberLoss(delta=0.5),
        valid_loss=MAE(),
        config=nbeats_config,
        num_samples=num_samples,
        refit_with_val=True,
        callbacks=ray_tune_kwargs_nbeats,
    ),
    AutoNHITS(
        h=horizon,
        loss=HuberLoss(delta=0.5),
        valid_loss=MAE(),
        config=nhits_config,
        num_samples=num_samples,
        refit_with_val=True,
        callbacks=ray_tune_kwargs_nhits,
    ),
]

# List of model names
model_names = ["AutoTFT", "AutoNBEATS", "AutoNHITS"]

# Create NeuralForecast object
nf = NeuralForecast(models=models, freq="10s")

# Load data
df = MarketDataset("~/.fintorch_data/marketdata-janestreet/")
df = df.data.collect()
# Number of unique series
n_series = len(df["unique_id"].unique())

# Perform cross-validation
Y_hat_df = nf.cross_validation(
    df=df,
    val_size=val_size,
    test_size=test_size,
    n_windows=None,  # Uses expanding window if None
).to_pandas()

# Iterate over models to compute metrics
results = []
for model_name in model_names:
    # Extract true values and predictions
    y_true = Y_hat_df.y.values
    y_hat = Y_hat_df[model_name].values

    # Reshape arrays
    y_true = y_true.reshape(n_series, -1, horizon)
    y_hat = y_hat.reshape(n_series, -1, horizon)

    # Compute metrics
    mse_score = mse(y_true, y_hat)
    mae_score = mae(y_true, y_hat)

    # Print results
    print(f"\nModel: {model_name}")
    print(f"MSE: {mse_score}")
    print(f"MAE: {mae_score}")

    # Append results to the list
    results.append({"Model": model_name, "MSE": mse_score, "MAE": mae_score})

# Save results to a CSV file
file_name = "model_scores.csv"
with open(file_name, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["Model", "MSE", "MAE"])
    writer.writeheader()
    writer.writerows(results)
