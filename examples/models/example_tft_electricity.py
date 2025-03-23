import os

import lightning as L
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import EarlyStopping

from fintorch.datasets.electricity_simple import ElectricityDataModule
from fintorch.models.timeseries.tft import TemporalFusionTransformerModule

# Define hyperparameters
number_of_past_inputs = 168
number_of_future_inputs = 24
embedding_size_inputs = hidden_dimension = 160
dropout = 0.1
number_of_heads = 4
batch_size = 1024


past_inputs = {"past_data": 1}
future_inputs = None
static_inputs = None
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
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    quantiles=quantiles,
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
plt.savefig("/home/marcel/Documents/research/FinTorch/plots/all_data_plot.png")

# Create a trainer with TensorBoard for better monitoring
early_stopping = EarlyStopping("val_loss_epoch", patience=50)
trainer = L.Trainer(
    max_epochs=2000,
    callbacks=[early_stopping],
    gradient_clip_val=0.01,
)

# Train the model
trainer.fit(tft_module, data_module)

trainer.test(tft_module, data_module)
