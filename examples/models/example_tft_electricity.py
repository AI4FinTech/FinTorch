import os

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping

from fintorch.datasets.electricity_simple import ElectricityDataModule
from fintorch.models.timeseries.tft import TemporalFusionTransformerModule

static_length = 2
past_inputs = 1
future_inputs = 1

# Define hyperparameters
number_of_past_inputs = 168
number_of_future_inputs = 24
embedding_size_inputs = hidden_dimension = 160
dropout = 0.1
number_of_heads = 4
batch_size = 1024


past_inputs = {"past_data": past_inputs}
future_inputs = None
static_inputs = None

# Create an instance of TemporalFusionTransformerModule
tft_module = TemporalFusionTransformerModule(
    number_of_past_inputs,
    number_of_future_inputs,
    embedding_size_inputs,
    hidden_dimension,
    dropout,
    number_of_heads,
    past_inputs,
    future_inputs,
    static_inputs,
    batch_size=batch_size,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

data_module = ElectricityDataModule(
    batch_size=batch_size,
    past_length=number_of_past_inputs,
    horizon=number_of_future_inputs,
    workers=os.cpu_count(),
)

# Set the precision
torch.set_float32_matmul_precision("medium")

# Prepare the data
data_module.setup()


plot_all_data = data_module.train_dataset.data

# Plot all data
plt.figure(figsize=(15, 5))
plt.plot(plot_all_data, label="All Data")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.title("All Data")
plt.legend()
plt.show()


# Create a trainer with TensorBoard for better monitoring
early_stopping = EarlyStopping("val_loss_epoch", patience=50)
trainer = L.Trainer(
    max_epochs=2000,
    callbacks=[early_stopping],
    gradient_clip_val=0.01,
)

# Train the model
trainer.fit(tft_module, data_module)

# Get the first 3 batches from the training dataloader (for demonstration)
train_dataloader = data_module.train_dataloader()
train_iter = iter(train_dataloader)
batches = [next(train_iter) for _ in range(3)]

# Make predictions directly using the trained model
predictions = []
for batch in batches:
    past_inputs, future_inputs, static_inputs, target = batch
    with torch.no_grad():  # Important: disable gradient calculation during prediction
        prediction, _ = tft_module(past_inputs, future_inputs, static_inputs)
    predictions.append(prediction)


num_batches_to_plot = 3
plt.figure(figsize=(15, 5 * num_batches_to_plot))

for batch_index in range(num_batches_to_plot):
    # Extract data
    target_timeseries = batches[batch_index][3][batch_index].detach().cpu().numpy()
    past_timeseries = (
        batches[batch_index][0]["past_data"][batch_index].detach().cpu().numpy()
    )
    future_timeseries = (
        batches[batch_index][1]["future_data"][batch_index].detach().cpu().numpy()
    )
    prediction = predictions[batch_index][batch_index].detach().cpu().numpy()

    print(
        f"Batch {batch_index + 1} with Predictions: {prediction} and future: {future_timeseries}"
    )

    # Plotting
    plt.subplot(num_batches_to_plot, 1, batch_index + 1)
    plt.plot(past_timeseries, label="Past")
    plt.plot(
        np.arange(len(past_timeseries), len(past_timeseries) + len(future_timeseries)),
        future_timeseries,
        label="Future",
    )
    plt.plot(
        np.arange(len(past_timeseries), len(past_timeseries) + len(target_timeseries)),
        target_timeseries,
        label="Target",
        linestyle="--",
    )
    plt.plot(
        np.arange(len(past_timeseries), len(past_timeseries) + len(prediction)),
        prediction,
        label="Prediction",
        linestyle=":",
    )
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title(f"Batch {batch_index + 1}: Past, Future, Target")
    plt.legend()

plt.tight_layout()  # Adjusts subplot params for a tight layout
plt.show()
