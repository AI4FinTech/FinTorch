import os
from random import randint

import lightning as L
import ray
import torch
from fintorch.datasets.synthetic import SimpleSyntheticDataModule
from fintorch.models.timeseries.causalformer.causalformer_module import (
    CausalFormerModule,
)
from lightning.pytorch.callbacks import EarlyStopping
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler

# --- Configuration (Constants) ---
# Data Parameters (Keep these fixed for the search)
TRAIN_LENGTH = 1000
VAL_LENGTH = 100
TEST_LENGTH = 100
# BATCH_SIZE = 32 # Batch size can also be tuned, but let's fix it for now
NOISE_LEVEL = 5
TREND_SLOPE = 0.1
SEASONALITY_AMPLITUDE = 10
SEASONALITY_PERIOD = 100
STATIC_LENGTH = 0
WORKERS = (
    os.cpu_count() // 2 if os.cpu_count() > 1 else 1
)  # Avoid using all cores for stability

# Fixed Model/Training Params
NUMBER_OF_SERIES = 1
FEATURE_DIMENSIONALITY = 1
OUTPUT_DIMENSIONALITY = 1
LENGTH_INPUT_WINDOW = 24
LENGTH_OUTPUT_WINDOW = 12
TAU = 1.0
LR_STEP_SIZE = 15  # Can be tuned if desired
LR_GAMMA = 0.1  # Can be tuned if desired
WEIGHT_DECAY = 1e-5  # Can be tuned if desired
MAX_EPOCHS = 50  # Max epochs per trial
PATIENCE = 10  # Early stopping patience per trial

# Set precision
torch.set_float32_matmul_precision("medium")

# --- Ray Tune Search Space ---
# Define hyperparameters to search over
config = {
    "lr": tune.loguniform(1e-4, 1e-1),  # Learning Rate
    "batch_size": tune.choice([32, 64, 128]),
    "num_layers": tune.choice([1, 2, 3, 4, 5]),
    "num_heads": tune.choice([1, 2, 4, 8, 16]),
    "embedding_size": tune.choice([32, 64, 128, 256, 512]),
    "ffn_hidden_dim": tune.choice([64, 128, 256, 512, 1024]),
    "dropout": tune.uniform(0.0, 0.5),
}


# --- Training Function for Ray Tune ---
def train_causalformer_tune(config, data_module=None, max_epochs=MAX_EPOCHS):
    """
    Training function callable by Ray Tune.
    Args:
        config (dict): Dictionary of hyperparameters for the trial.
        data_module (LightningDataModule): Pre-initialized data module.
        max_epochs (int): Maximum epochs for this trial.
    """
    # --- 1. (Re)Setup DataModule with tuned batch size ---
    current_batch_size = config["batch_size"]
    data_module = SimpleSyntheticDataModule(
        train_length=TRAIN_LENGTH,
        val_length=VAL_LENGTH,
        test_length=TEST_LENGTH,
        batch_size=current_batch_size,  # Use tuned batch size
        noise_level=NOISE_LEVEL,
        past_length=LENGTH_INPUT_WINDOW,
        future_length=LENGTH_OUTPUT_WINDOW,
        static_length=STATIC_LENGTH,
        trend_slope=TREND_SLOPE,
        seasonality_amplitude=SEASONALITY_AMPLITUDE,
        seasonality_period=SEASONALITY_PERIOD,
        workers=WORKERS,
    )

    # --- 2. Initialize Model with Tuned Hyperparameters ---
    model = CausalFormerModule(
        number_of_layers=config["num_layers"],
        number_of_heads=config["num_heads"],
        number_of_series=NUMBER_OF_SERIES,
        length_input_window=LENGTH_INPUT_WINDOW,
        length_output_window=LENGTH_OUTPUT_WINDOW,
        embedding_size=config["embedding_size"],
        feature_dimensionality=FEATURE_DIMENSIONALITY,
        ffn_hidden_dimensionality=config["ffn_hidden_dim"],
        output_dimensionality=OUTPUT_DIMENSIONALITY,
        tau=TAU,
        dropout=config["dropout"],
        learning_rate=config["lr"],
        lr_step_size=LR_STEP_SIZE,
        lr_gamma=LR_GAMMA,
        weight_decay=WEIGHT_DECAY,
    )

    # --- 3. Configure Trainer Callbacks for Ray Tune ---
    tune_checkpoint_callback = TuneReportCheckpointCallback(
        metrics={
            "loss": "val_loss",  # Metric to report (must match key in validation_step log)
        },
        filename="tune_ckpt",  # Checkpoint filename within the trial directory
        on="validation_end",  # Report and potentially checkpoint at validation end
    )

    # Early stopping for the trial
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        verbose=False,
        mode="min",
    )

    # --- 4. Configure Trainer ---
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        callbacks=[tune_checkpoint_callback, early_stopping],
        enable_checkpointing=True,
    )

    # --- 5. Train the Model ---
    trainer.fit(model, datamodule=data_module)


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Configure Ray Tune Analysis ---
    scheduler = ASHAScheduler(
        max_t=MAX_EPOCHS,  # Max 'time' units (epochs in this case) per trial
        grace_period=min(10, MAX_EPOCHS),  # Min epochs before a trial can be stopped
        reduction_factor=2,  # Reduce trials by half each round
    )

    # Define resources per trial
    resources_per_trial = {"cpu": WORKERS, "gpu": 1 if torch.cuda.is_available() else 0}

    print("Starting Ray Tune hyperparameter search...")
    analysis = tune.run(
        tune.with_resources(train_causalformer_tune, resources=resources_per_trial),
        config=config,
        num_samples=50,  # Number of different hyperparameter combinations to try
        scheduler=scheduler,
        metric="loss",  # Metric to optimize (minimize)
        mode="min",
        # Optional: Specify search algorithm (requires imports like `from ray.tune.search.hyperopt import HyperOptSearch`)
        # search_alg=HyperOptSearch(metric="loss", mode="min"),
        name="causalformer_tune",  # Name of the experiment
        verbose=3,  # Set verbosity level (0, 1, 2, 3)
    )

    print("Hyperparameter search finished.")
    best_config = analysis.get_best_config(metric="loss", mode="min")
    print("Best hyperparameters found: ", best_config)

    best_trial = analysis.get_best_trial(metric="loss", mode="min", scope="last")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")

    # --- (Optional) Retrain the best model on full data ---
    print("\nRetraining best model...")
    best_model = CausalFormerModule(
        number_of_layers=best_config["num_layers"],
        number_of_heads=best_config["num_heads"],
        number_of_series=NUMBER_OF_SERIES,
        length_input_window=LENGTH_INPUT_WINDOW,
        length_output_window=LENGTH_OUTPUT_WINDOW,
        embedding_size=best_config["embedding_size"],
        feature_dimensionality=FEATURE_DIMENSIONALITY,
        ffn_hidden_dimensionality=best_config["ffn_hidden_dim"],
        output_dimensionality=OUTPUT_DIMENSIONALITY,
        tau=TAU,
        dropout=best_config["dropout"],
        learning_rate=best_config["lr"],
        lr_step_size=LR_STEP_SIZE,
        lr_gamma=LR_GAMMA,
        weight_decay=WEIGHT_DECAY,
    )

    # Re-initialize DataModule with best batch size
    final_data_module = SimpleSyntheticDataModule(
        train_length=TRAIN_LENGTH,
        val_length=VAL_LENGTH,
        test_length=TEST_LENGTH,
        batch_size=best_config["batch_size"],  # Use best batch size
        noise_level=NOISE_LEVEL,
        past_length=LENGTH_INPUT_WINDOW,
        future_length=LENGTH_OUTPUT_WINDOW,
        static_length=STATIC_LENGTH,
        trend_slope=TREND_SLOPE,
        seasonality_amplitude=SEASONALITY_AMPLITUDE,
        seasonality_period=SEASONALITY_PERIOD,
        workers=WORKERS,
    )

    # Final Trainer (can use more epochs, loggers, etc.)
    final_trainer = L.Trainer(
        max_epochs=MAX_EPOCHS * 2,  # Train longer potentially
        accelerator="auto",
        devices=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=PATIENCE * 2, mode="min")
        ],  # Longer patience maybe
    )

    final_trainer.fit(best_model, datamodule=final_data_module)

    print("\nTesting best model...")
    test_results = final_trainer.test(best_model, datamodule=final_data_module)
    print("Test Results (Best Model):", test_results)

    # --- 6. Make Predictions and Plot ---
    print("Generating Predictions and Plotting...")

    model = best_model
    model.eval()

    # Get the test dataloader
    test_loader = final_data_module.test_dataloader()

    # Get a batch from the test dataloader
    batch = next(iter(test_loader))
    x_test, y_test = model._prepare_data(batch)  # Prepare the data

    # Move data to the correct device
    device = next(model.parameters()).device
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    # Generate predictions with torch.no_grad()
    with torch.no_grad():
        predictions = model(x_test)

    # Concatenate predictions
    all_predictions = predictions

    # Number of batches to plot
    num_batches_to_plot = 5

    # Plotting
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 5 * num_batches_to_plot))  # Adjust figure size

    for idx in range(0, num_batches_to_plot):
        batch_idx = randint(0, len(all_predictions) - 1)

        selected_batch_predictions = all_predictions[batch_idx]
        _, _, _, selected_batch_target = final_data_module.test_dataset[batch_idx]

        plt.subplot(num_batches_to_plot, 1, idx + 1)  # Create subplots

        plt.plot(
            selected_batch_target.squeeze(), label="Target", marker="o", linestyle="-"
        )
        plt.plot(
            selected_batch_predictions.squeeze(),
            label="Predicted",
            marker="x",
            linestyle="--",
        )

        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.title(f"Target vs Predicted for Batch {batch_idx + 1}")
        plt.legend()

    plt.tight_layout()  # Adjust subplot parameters for a tight layout
    plt.show()
    # --- Shutdown Ray ---
    ray.shutdown()  # Optional: explicitly shut down Ray
