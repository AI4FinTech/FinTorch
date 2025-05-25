import os
from random import randint

import lightning as L
import matplotlib.pyplot as plt
import torch

# Assuming fintorch is installed or accessible in the Python path
# Make sure this imported module matches the structure from the immersive artifact
# (accepts download_url, local_path, static_length and returns dicts + target)
from fintorch.datasets.diamondata import DiamondDataModule
from fintorch.models.timeseries.causalformer.causalformer_module import (
    CausalFormerModule,
)
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# --- Configuration ---
# Data Parameters (Updated for DiamondDataModule with download)
LOCAL_DATA_PATH = "data_0.csv"  # Local filename for downloaded data
TIME_STEP = 24  # Input window length (past)
OUTPUT_WINDOW = 12  # Output window length (future/prediction)
STATIC_LENGTH = 0  # CausalFormer doesn't use static features in this setup
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() else 1  # Use half cores or 1
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
# TEST_SPLIT is inferred

# Model Hyperparameters (Adjust based on dataset complexity and performance)
LENGTH_INPUT_WINDOW = TIME_STEP
LENGTH_OUTPUT_WINDOW = OUTPUT_WINDOW

# FEATURE_DIMENSIONALITY and OUTPUT_DIMENSIONALITY assume 1 based on DataModule reshape
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

    # --- 1. Setup Data ---
    print("Setting up DiamondDataModule...")
    # Use updated parameters for DiamondDataModule
    data_module = DiamondDataModule(
        local_path=LOCAL_DATA_PATH,
        time_step=TIME_STEP,
        output_window=OUTPUT_WINDOW,
        static_length=STATIC_LENGTH,  # Pass static length
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        train_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
    )

    # Setup (loads data into memory, creates splits)
    print("Setting up data splits...")
    data_module.setup()
    print("DataModule setup complete.")

    # --- Update NUMBER_OF_SERIES based on loaded data ---
    if hasattr(data_module, "series_num") and data_module.series_num is not None:
        NUMBER_OF_SERIES = data_module.series_num
        print(f"Updated NUMBER_OF_SERIES based on loaded data: {NUMBER_OF_SERIES}")
    else:
        print(
            f"Warning: Could not determine number of series from data_module. Using default: {NUMBER_OF_SERIES}"
        )

    # Print dataset sizes
    if data_module.dataset:
        print(f"Total samples processed: {len(data_module.dataset)}")
        print(
            f"Train samples: {len(data_module.train_dataset) if data_module.train_dataset else 'N/A'}"
        )  # type: ignore
        print(
            f"Val samples:   {len(data_module.val_dataset) if data_module.val_dataset else 'N/A'}"
        )  # type: ignore
        print(
            f"Test samples:  {len(data_module.test_dataset) if data_module.test_dataset else 'N/A'}"
        )  # type: ignore
    else:
        print("Warning: DataModule dataset object not found after setup.")

    # --- Optional: Print Batch Shapes (Updated Unpacking) ---
    try:
        print("\n--- Sample Batch Shapes ---")
        train_loader = data_module.train_dataloader()
        if len(train_loader) > 0:
            # Unpack the batch according to the new structure
            past_batch, future_batch, static_batch, target_batch = next(
                iter(train_loader)
            )
            print("Train Batch Shapes:")
            print(f"  past_inputs['past_data']:   {past_batch['past_data'].shape}")
            print(f"  future_inputs['future_data']:{future_batch['future_data'].shape}")
            print(f"  static_inputs['static_data']:{static_batch['static_data'].shape}")
            print(f"  target:                     {target_batch.shape}")
        else:
            print("Training DataLoader is empty.")
    except Exception as e:
        print(f"Could not retrieve or print batch shapes: {e}")

    # --- 2. Initialize Model ---
    print("\nInitializing CausalFormerModule...")
    # Pass the potentially updated NUMBER_OF_SERIES
    causalformer_module = CausalFormerModule(
        number_of_layers=NUMBER_OF_LAYERS,
        number_of_heads=NUMBER_OF_HEADS,
        number_of_series=NUMBER_OF_SERIES,  # Use value derived from data
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
        # Add static_length if your CausalFormerModule accepts it
        # static_length=STATIC_LENGTH,
    )

    # --- 3. Configure Trainer ---
    print("Configuring Trainer...")
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=PATIENCE, verbose=True, mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="causalformer_diamond_checkpoints/",
        filename="causalformer-diamond-best-{epoch:02d}-{val_loss_epoch:.4f}",
        save_top_k=1,
        mode="min",
    )
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[early_stopping, checkpoint_callback],
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,  # Adjusted logging frequency
    )

    # --- 4. Train the Model ---
    print("Starting Training...")
    # The CausalFormerModule's training_step needs to handle the new batch structure
    # (past_inputs, future_inputs, static_inputs, target)
    trainer.fit(causalformer_module, datamodule=data_module)

    # --- 5. Test the Model ---
    print("\nStarting Testing...")
    # The CausalFormerModule's test_step needs to handle the new batch structure
    test_results = trainer.test(datamodule=data_module, ckpt_path="best")
    print("Test Results:", test_results)

    # --- 6. Make Predictions and Plot (Updated Unpacking) ---
    print("\nGenerating Predictions and Plotting...")
    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path:
        print("Could not find best model path. Using last model state for prediction.")
        model = causalformer_module
    else:
        print(f"Loading best model from: {best_model_path}")
        # Ensure all necessary hyperparameters are passed for loading
        model = CausalFormerModule.load_from_checkpoint(
            best_model_path,
            number_of_layers=NUMBER_OF_LAYERS,
            number_of_heads=NUMBER_OF_HEADS,
            number_of_series=NUMBER_OF_SERIES,  # Use updated value
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
            # Add static_length if needed by your model's __init__
            # static_length=STATIC_LENGTH,
        )

    model.eval()
    device = next(model.parameters()).device
    print(f"Model running on device: {device}")

    num_plots = 5
    # Ensure test_dataset exists before proceeding
    if not data_module or not data_module.test_dataset:
        print("Test dataset not available. Cannot generate plots.")
    else:
        num_test_samples = len(data_module.test_dataset)  # type: ignore
        if num_test_samples == 0:
            print("Test dataset is empty. Cannot generate plots.")
        else:
            plot_indices = [
                randint(0, num_test_samples - 1)
                for _ in range(min(num_plots, num_test_samples))
            ]

            plt.figure(figsize=(15, 5 * min(num_plots, num_test_samples)))

            for i, idx in enumerate(plot_indices):
                # Get a single sample using the new structure
                # Note: static_inputs and future_inputs are ignored for prediction/plotting here
                past_inputs, _future_inputs, _static_inputs, target_data = (
                    data_module.test_dataset[idx]
                )  # type: ignore
                past_data = past_inputs["past_data"]  # Extract the tensor

                # Prepare input tensor (adjust shape for model)
                # Assuming past_data is (time_step, series_num, 1)
                if past_data.dim() == 2:
                    # Permute to (series_num, time_step, 1) -> add batch dim
                    input_tensor = (
                        past_data.unsqueeze(-1).permute(1, 0, 2).unsqueeze(0)
                    )  # (batch=1, series=N, time=T, feat=1)
                else:
                    print(
                        f"Warning: Unexpected input data dimension {past_data.dim()}. Check data loading."
                    )
                    # Attempt a reasonable reshape if possible, otherwise skip sample
                    continue  # Or handle differently

                input_tensor = input_tensor.to(device)

                # Generate predictions
                with torch.no_grad():
                    # The CausalFormerModule's forward method needs to handle the input shape
                    # It might also need static features if designed for it
                    prediction = model(input_tensor)  # Pass only past_data tensor

                # Prepare target and prediction for plotting
                # Assuming target_data is (output_window, series_num)
                # Assuming prediction is (batch=1, series_num, output_window, 1)
                target_plot = (
                    target_data.squeeze()[3, :].cpu().numpy()
                )  # Plot first series
                prediction_plot = (
                    prediction.squeeze()[3, :].cpu().numpy()
                )  # Plot first series

                # Plotting
                plt.subplot(min(num_plots, num_test_samples), 1, i + 1)
                plt.plot(target_plot, label="Actual Target", marker=".", linestyle="-")
                plt.plot(
                    prediction_plot,
                    label="CausalFormer Prediction",
                    marker="x",
                    linestyle="--",
                )
                plt.title(
                    f"Test Sample Index: {idx} - Target vs. Prediction (Series 3)"
                )
                plt.xlabel(f"Future Time Step (Window Size = {OUTPUT_WINDOW})")
                plt.ylabel("Value")
                plt.legend()

            plt.tight_layout()
            plt.savefig("causalformer_predictions.png")
            plt.show()

    print("\nExample script finished.")
