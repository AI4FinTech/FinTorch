import os
from random import randint

import lightning as L
import matplotlib.pyplot as plt
import torch

# Updated imports to use CausalDataModule instead of DiamondDataModule
from fintorch.datasets.causal_data import create_causal_datamodule, list_available_datasets
from fintorch.models.timeseries.causalformer.causalformer_module import (
    CausalFormerModule,
)
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# --- Configuration ---
# Data Parameters (Updated for CausalDataModule)
DATASET_TYPE = "diamond"  # Use diamond dataset from causal_data (note: singular form)
TIME_STEP = 3  # Input window length (past)
OUTPUT_WINDOW = 1  # Output window length (future/prediction)
STATIC_LENGTH = 0  # CausalFormer doesn't use static features in this setup
BATCH_SIZE = 32
NUM_WORKERS = max(1, (os.cpu_count() or 1) // 2)  # Use half cores or minimum 1
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
# TEST_SPLIT is inferred

# Model Hyperparameters (Adjust based on dataset complexity and performance)
LENGTH_INPUT_WINDOW = TIME_STEP
LENGTH_OUTPUT_WINDOW = OUTPUT_WINDOW

# FEATURE_DIMENSIONALITY and OUTPUT_DIMENSIONALITY will be determined from data
FEATURE_DIMENSIONALITY = 1
OUTPUT_DIMENSIONALITY = 1

# CausalFormer Architecture Parameters
NUMBER_OF_LAYERS = 2
NUMBER_OF_HEADS = 8
EMBEDDING_SIZE = 128
FFN_HIDDEN_DIMENSIONALITY = 128
TAU = 1.0
DROPOUT = 0.2
NUMBER_OF_SERIES = 4  # Default value, will be updated from data

# Training Hyperparameters
LEARNING_RATE = 0.01
LR_STEP_SIZE = 10
LR_GAMMA = 0.1
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 30
PATIENCE = 10


# --- Main Script Logic ---
if __name__ == "__main__":
    # Set precision for better performance on compatible GPUs
    torch.set_float32_matmul_precision("medium")

    data_module = None  # Initialize to None for finally block

    # --- 1. List Available Datasets ---
    print("Available datasets:")
    available_datasets = list_available_datasets()
    for dataset in available_datasets:
        print(f"  - {dataset}")

    if DATASET_TYPE not in available_datasets:
        print(f"Warning: {DATASET_TYPE} not in available datasets. Using first available dataset.")
        DATASET_TYPE = available_datasets[0] if available_datasets else "diamond"

    print(f"\nUsing dataset: {DATASET_TYPE}")

    # --- 2. Setup Data ---
    print("Setting up CausalDataModule...")
    # Use create_causal_datamodule function
    data_module = create_causal_datamodule(
        dataset_type=DATASET_TYPE,
        time_step=TIME_STEP,
        output_window=OUTPUT_WINDOW,
        static_length=STATIC_LENGTH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        train_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
    )

    # Setup (loads data into memory, creates splits)
    print("Setting up data splits...")
    data_module.setup()
    print("DataModule setup complete.")

    # --- Update model parameters based on loaded data ---
    # Get dimensions from the dataset (accessible after setup)
    if data_module.dataset:
        # Update NUMBER_OF_SERIES to match actual data dimensions
        NUMBER_OF_SERIES = data_module.dataset.series_dim
        print(f"Updated NUMBER_OF_SERIES based on loaded data: {NUMBER_OF_SERIES}")

        FEATURE_DIMENSIONALITY = data_module.dataset.features_dim
        print(f"Updated FEATURE_DIMENSIONALITY based on loaded data: {FEATURE_DIMENSIONALITY}")

        # Fix: OUTPUT_DIMENSIONALITY should be features per series, not total series count
        OUTPUT_DIMENSIONALITY = 1  # Each series has 1 target feature
        print(f"Updated OUTPUT_DIMENSIONALITY based on loaded data: {OUTPUT_DIMENSIONALITY}")

        # Print dataset information
        print("\nDataset Information:")
        print(f"  Time steps (past): {data_module.dataset.time_steps}")
        print(f"  Future steps: {data_module.dataset.future_steps}")
        print(f"  Series dimension: {data_module.dataset.series_dim}")
        print(f"  Features dimension: {data_module.dataset.features_dim}")
        print(f"  Static length: {data_module.dataset.static_length}")
    else:
        print("Warning: Dataset not loaded, using default parameters")

    # Print dataset sizes
    print("\nDataset Sizes:")
    if data_module.dataset:
        print(f"  Total samples: {len(data_module.dataset)}")
    if data_module.train_dataset:
        print(f"  Train samples: {len(data_module.train_dataset)}")
    if data_module.val_dataset:
        print(f"  Val samples: {len(data_module.val_dataset)}")
    if data_module.test_dataset:
        print(f"  Test samples: {len(data_module.test_dataset)}")

    # --- Optional: Print Batch Shapes ---
    try:
        print("\n--- Sample Batch Shapes ---")
        train_loader = data_module.train_dataloader()
        if len(train_loader) > 0:
            # Get a sample batch - CausalDataModule returns a dictionary format
            batch = next(iter(train_loader))
            print("Train Batch Structure:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {type(value)}")
        else:
            print("Training DataLoader is empty.")
    except Exception as e:
        print(f"Could not retrieve or print batch shapes: {e}")

    # --- 3. Initialize Model ---
    print("\nInitializing CausalFormerModule...")
    causalformer_module = CausalFormerModule(
        number_of_layers=NUMBER_OF_LAYERS,
        number_of_heads=NUMBER_OF_HEADS,
        number_of_series=NUMBER_OF_SERIES,
        length_input_window=LENGTH_INPUT_WINDOW,
        length_output_window=LENGTH_OUTPUT_WINDOW,
        embedding_size=EMBEDDING_SIZE,
        feature_dimensionality=FEATURE_DIMENSIONALITY,
        ffn_hidden_dimensionality=FFN_HIDDEN_DIMENSIONALITY,
        output_dimensionality=OUTPUT_DIMENSIONALITY,
        tau=TAU,
        dropout=DROPOUT,
        learning_rate=LEARNING_RATE,
        lr_step_size=LR_STEP_SIZE,
        lr_gamma=LR_GAMMA,
        weight_decay=WEIGHT_DECAY,
    )

    # --- 4. Configure Trainer ---
    print("Configuring Trainer...")
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=PATIENCE, verbose=True, mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="causalformer_causal_checkpoints/",
        filename="causalformer-causal-best-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min",
    )
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[early_stopping, checkpoint_callback],
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,
    )

    # --- 5. Train the Model ---
    print("Starting Training...")
    trainer.fit(causalformer_module, datamodule=data_module)

    # --- 6. Test the Model ---
    print("\nStarting Testing...")
    test_results = trainer.test(datamodule=data_module, ckpt_path="best")
    print("Test Results:", test_results)

    # --- 7. Make Predictions and Plot ---
    print("\nGenerating Predictions and Plotting...")
    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path:
        print("Could not find best model path. Using last model state for prediction.")
        model = causalformer_module
    else:
        print(f"Loading best model from: {best_model_path}")
        model = CausalFormerModule.load_from_checkpoint(
            best_model_path,
            number_of_layers=NUMBER_OF_LAYERS,
            number_of_heads=NUMBER_OF_HEADS,
            number_of_series=NUMBER_OF_SERIES,
            length_input_window=LENGTH_INPUT_WINDOW,
            length_output_window=LENGTH_OUTPUT_WINDOW,
            embedding_size=EMBEDDING_SIZE,
            feature_dimensionality=FEATURE_DIMENSIONALITY,
            ffn_hidden_dimensionality=FFN_HIDDEN_DIMENSIONALITY,
            output_dimensionality=OUTPUT_DIMENSIONALITY,
            tau=TAU,
            dropout=DROPOUT,
            learning_rate=LEARNING_RATE,
            lr_step_size=LR_STEP_SIZE,
            lr_gamma=LR_GAMMA,
            weight_decay=WEIGHT_DECAY,
        )

    model.eval()
    device = next(model.parameters()).device
    print(f"Model running on device: {device}")

    # --- 8. Generate Plots ---
    num_plots = 5
    if not data_module.test_dataset or len(data_module.test_dataset) == 0:
        print("Test dataset is empty. Cannot generate plots.")
    else:
        num_test_samples = len(data_module.test_dataset)
        plot_indices = [
            randint(0, num_test_samples - 1)
            for _ in range(min(num_plots, num_test_samples))
        ]

        plt.figure(figsize=(15, 5 * min(num_plots, num_test_samples)))

        for i, idx in enumerate(plot_indices):
            try:
                # Get a single sample from test dataset
                if data_module.test_dataset is None:
                    print(f"Test dataset is None, skipping sample {idx}")
                    continue
                sample = data_module.test_dataset[idx]

                # Create a batch with single sample for prediction
                batch = {}
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.unsqueeze(0)  # Add batch dimension
                    else:
                        batch[key] = value

                # Move batch to device
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(device)

                # Generate predictions
                with torch.no_grad():
                    prediction = model.predict_step(batch, 0)

                # Extract target for plotting
                target_data = batch["output_target"].cpu().squeeze(0)  # Remove batch dim
                prediction_data = prediction.cpu().squeeze(0)  # Remove batch dim

                # Plot first series if multiple series exist
                if target_data.dim() > 2:  # [series, time, features]
                    target_plot = target_data[0, :, 0].numpy()  # First series, first feature
                    prediction_plot = prediction_data[0, :, 0].numpy()
                elif target_data.dim() == 2:  # [time, features] or [series, time]
                    if target_data.shape[1] == OUTPUT_WINDOW:  # [series, time]
                        target_plot = target_data[0, :].numpy()  # First series
                        prediction_plot = prediction_data[0, :].numpy()
                    else:  # [time, features]
                        target_plot = target_data[:, 0].numpy()  # First feature
                        prediction_plot = prediction_data[:, 0].numpy()
                else:  # [time]
                    target_plot = target_data.numpy()
                    prediction_plot = prediction_data.numpy()

                # Plotting
                plt.subplot(min(num_plots, num_test_samples), 1, i + 1)
                plt.plot(target_plot, label="Actual Target", marker=".", linestyle="-")
                plt.plot(
                    prediction_plot,
                    label="CausalFormer Prediction",
                    marker="x",
                    linestyle="--",
                )
                plt.title(f"Test Sample {idx} - Target vs. Prediction")
                plt.xlabel(f"Future Time Step (Window Size = {OUTPUT_WINDOW})")
                plt.ylabel("Value")
                plt.legend()

            except Exception as e:
                print(f"Error plotting sample {idx}: {e}")
                continue

        plt.tight_layout()
        plt.savefig("causalformer_causal_predictions.png")
        plt.show()
        print("Plots saved as 'causalformer_causal_predictions.png'")

    print("\nExample script finished successfully!")
