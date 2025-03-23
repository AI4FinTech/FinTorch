import os

import lightning as L
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import EarlyStopping

from fintorch.datasets.airpassenger import AirPassengerDataModule
from fintorch.models.timeseries.tft import TemporalFusionTransformerModule

# Define hyperparameters
number_of_past_inputs = 24
number_of_future_inputs = 12
embedding_size_inputs = hidden_dimension = 64
dropout = 0.1
number_of_heads = 4
batch_size = 16

past_inputs = {"past_data": 1}
future_inputs = None
static_inputs = None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

quantiles = [0.05, 0.5, 0.95]

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
    device=device,
    quantiles=quantiles,
).to(device)

data_module = AirPassengerDataModule(
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
plt.savefig("/home/marcel/Documents/research/FinTorch/plots/all_data_plot.png")

# Create a trainer with TensorBoard for better monitoring
early_stopping = EarlyStopping("val_loss_epoch", patience=50)
trainer = L.Trainer(max_epochs=200, callbacks=[early_stopping], log_every_n_steps=10)

# Train the model
trainer.fit(tft_module, data_module)

trainer.test(tft_module, data_module)
