import os

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping

from fintorch.datasets.electricity_simple import ElectricityDataModule
from fintorch.models.timeseries.tft import TemporalFusionTransformerModule

static_length = 2
past_inputs = 1
future_inputs = 1

# Define hyperparameters
number_of_past_inputs = 24
number_of_future_inputs = 12
embedding_size_inputs = hidden_dimension = 32
dropout = 0.5
number_of_heads = 1
past_inputs = {"past_data": past_inputs}
future_inputs = {"future_data": future_inputs}
static_inputs = {"static_data": static_length}

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
)

data_module = ElectricityDataModule(
    train_length=1000,
    val_length=100,
    test_length=100,
    batch_size=128,
    past_length=number_of_past_inputs,
    future_length=number_of_future_inputs,
    static_length=static_length,
    workers=os.cpu_count(),
)

# Set the precision
torch.set_float32_matmul_precision("medium")

# Prepare the data
data_module.setup()


# Create a trainer with TensorBoard for better monitoring
early_stopping = EarlyStopping("val_loss_epoch", patience=50)
trainer = L.Trainer(max_epochs=500, callbacks=[early_stopping])

# Train the model
trainer.fit(tft_module, data_module)
