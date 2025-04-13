import os
from random import randint

import lightning as L
import matplotlib.pyplot as plt
import torch

# Assuming fintorch is installed or accessible in the Python path
from fintorch.datasets.synthetic import SimpleSyntheticDataModule
from fintorch.models.timeseries.causalformer.causalformer_module import (
    CausalFormerModule,
)
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# --- Configuration ---
# Data Parameters
TRAIN_LENGTH = 1000
VAL_LENGTH = 100
TEST_LENGTH = 100
BATCH_SIZE = 32
NOISE_LEVEL = 5
TREND_SLOPE = 0.1
SEASONALITY_AMPLITUDE = 10
SEASONALITY_PERIOD = 100
STATIC_LENGTH = 0  # CausalFormer doesn't use static features in this setup
WORKERS = os.cpu_count()

# Model Hyperparameters
# Note: Set number_of_series=1 and feature_dimensionality=1 for univariate synthetic data
NUMBER_OF_SERIES = 1
FEATURE_DIMENSIONALITY = 1
OUTPUT_DIMENSIONALITY = 1  # Predicting the single feature
LENGTH_INPUT_WINDOW = 24  # Corresponds to past_length
LENGTH_OUTPUT_WINDOW = 12  # Corresponds to future_length

NUMBER_OF_LAYERS = 1
NUMBER_OF_HEADS = 1
EMBEDDING_SIZE = 32
FFN_HIDDEN_DIMENSIONALITY = 32
TAU = 1.0  # Scaling factor for attention (adjust as needed)
DROPOUT = 0.1
LEARNING_RATE = 0.001
LR_STEP_SIZE = 15
LR_GAMMA = 0.1
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 50
PATIENCE = 10  # For EarlyStopping

# Set precision for better performance on compatible GPUs
torch.set_float32_matmul_precision("medium")

# --- 1. Setup Data ---
print("Setting up DataModule...")
data_module = SimpleSyntheticDataModule(
    train_length=TRAIN_LENGTH,
    val_length=VAL_LENGTH,
    test_length=TEST_LENGTH,
    batch_size=BATCH_SIZE,
    noise_level=NOISE_LEVEL,
    past_length=LENGTH_INPUT_WINDOW,
    future_length=LENGTH_OUTPUT_WINDOW,
    static_length=STATIC_LENGTH,  # Ignored by model but needed by datamodule
    trend_slope=TREND_SLOPE,
    seasonality_amplitude=SEASONALITY_AMPLITUDE,
    seasonality_period=SEASONALITY_PERIOD,
    workers=WORKERS,
)

data_module.setup()

# --- (Optional) Plot Data ---
print("Plotting sample data...")
try:
    plot_data = data_module.train_dataset.data[:500]  # Plot first 500 points
    plt.figure(figsize=(15, 5))
    plt.plot(plot_data, label="Synthetic Training Data Sample")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title("Synthetic Data Sample")
    plt.legend()
    plt.show()
except Exception as e:
    print(f"Could not plot data: {e}")


# --- 2. Initialize Model ---
print("Initializing CausalFormerModule...")
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

# --- 3. Configure Trainer ---
print("Configuring Trainer...")
# Stop training if validation loss doesn't improve for 'patience' epochs
early_stopping = EarlyStopping(
    monitor="val_loss",  # Monitor the epoch-level validation loss
    patience=PATIENCE,
    verbose=True,
    mode="min",  # Stop when the monitored quantity stops decreasing
)

# Save the model checkpoint based on validation loss
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="causalformer_checkpoints/",
    filename="causalformer-best-{epoch:02d}-{val_loss_epoch:.2f}",
    save_top_k=1,
    mode="min",
)

trainer = L.Trainer(
    max_epochs=MAX_EPOCHS,
    callbacks=[early_stopping, checkpoint_callback],
    accelerator="auto",  # Automatically selects GPU/CPU/TPU
    devices="auto",
    log_every_n_steps=10,  # Log less frequently for cleaner output
    # precision="16-mixed" # Optional: Use mixed precision if desired and supported
)

# --- 4. Train the Model ---
print("Starting Training...")
trainer.fit(causalformer_module, datamodule=data_module)

# --- 5. Test the Model ---
print("Starting Testing...")
# Loads the best checkpoint automatically
test_results = trainer.test(datamodule=data_module, ckpt_path="best")
print("Test Results:", test_results)

# --- 6. Make Predictions and Plot ---
print("Generating Predictions...")
# Load the best model checkpoint for prediction
best_model_path = checkpoint_callback.best_model_path
print(f"Loading best model from: {best_model_path}")
# If trainer.test loaded the best model, we can use the current module instance
# Otherwise, load explicitly:
# model = CausalFormerModule.load_from_checkpoint(best_model_path)
model = causalformer_module
model.eval()  # Set model to evaluation mode


# Get a batch from the test dataloader and prepare the data
test_loader = data_module.test_dataloader()
batch = next(iter(test_loader))
x_test, y_test = model._prepare_data(batch)


# Move data to the correct device (if model is on GPU)
device = next(model.parameters()).device
x_test = x_test.to(device)
y_test = y_test.to(device)  # Keep y_test for comparison

# Generate predictions
with torch.no_grad():
    predictions = model(x_test)

# Reshape predictions and actuals for plotting if needed
# Concatenate predictions
all_predictions = predictions

# Number of batches to plot
num_batches_to_plot = 5

# Plotting
plt.figure(figsize=(15, 5 * num_batches_to_plot))  # Adjust figure size

for idx in range(0, num_batches_to_plot):
    batch_idx = randint(0, len(all_predictions) - 1)

    selected_batch_predictions = all_predictions[batch_idx]
    _, _, _, selected_batch_target = data_module.test_dataset[batch_idx]

    plt.subplot(num_batches_to_plot, 1, idx + 1)  # Create subplots

    plt.plot(selected_batch_target.squeeze(), label="Target", marker="o", linestyle="-")
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

print("Example script finished.")
