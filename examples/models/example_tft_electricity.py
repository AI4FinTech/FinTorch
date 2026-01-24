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

# Define feature dimensions based on the ElectricityDataset
# These values match the dataset's feature configuration
num_past_target_features = 1  # electricity consumption
num_past_known_cov_features = 4  # hour, weekday, month, day
num_past_unknown_cov_features = 1  # placeholder unknown features
num_future_known_cov_features = 4  # same time features for future
num_static_real_features = 2  # placeholder static real features
num_static_categorical_features = 1  # placeholder static categorical features
static_categorical_cardinalities = [2]  # cardinalities for categorical features

quantiles = [0.05, 0.5, 0.95]

# Create an instance of TemporalFusionTransformerModule with the new interface
tft_module = TemporalFusionTransformerModule(
    number_of_past_inputs=number_of_past_inputs,
    horizon=number_of_future_inputs,
    embedding_size_inputs=embedding_size_inputs,
    hidden_dimension=hidden_dimension,
    dropout=dropout,
    number_of_heads=number_of_heads,
    # New required parameters based on standardized dataset format
    num_past_target_features=num_past_target_features,
    num_past_known_cov_features=num_past_known_cov_features,
    num_past_unknown_cov_features=num_past_unknown_cov_features,
    num_future_known_cov_features=num_future_known_cov_features,
    num_static_real_features=num_static_real_features,
    num_static_categorical_features=num_static_categorical_features,
    static_categorical_cardinalities=static_categorical_cardinalities,
    batch_size=batch_size,
    device=str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
    quantiles=quantiles,
    # Multi-series handling (ElectricityDataset has series_dim=1, so this doesn't matter much)
    series_selection_method="first",  # Use first (and only) series
)

# Create data module
data_module = ElectricityDataModule(
    batch_size=batch_size,
    past_length=number_of_past_inputs,
    horizon=number_of_future_inputs,
    workers=os.cpu_count() or 1,
)

# Set the precision
torch.set_float32_matmul_precision("medium")

# Prepare the data
data_module.setup()

# Get sample data for plotting
sample_data = data_module.train_dataset.data['target'].flatten()

# Plot sample of the data
plt.figure(figsize=(15, 5))
plt.plot(sample_data[:1000], label="Electricity Consumption (First 1000 points)")
plt.xlabel("Time Step")
plt.ylabel("Normalized Consumption")
plt.title("Sample Electricity Consumption Data")
plt.legend()
plt.show()

# Create a trainer with TensorBoard for better monitoring
early_stopping = EarlyStopping("val_loss", patience=50)
trainer = L.Trainer(
    max_epochs=2000,
    callbacks=[early_stopping],
    gradient_clip_val=0.01,
    log_every_n_steps=50,
    check_val_every_n_epoch=5,
)

# Train the model
print("Starting training...")
trainer.fit(tft_module, data_module)

# Test the model
print("Starting testing...")
trainer.test(tft_module, data_module)

# Make predictions on test data
print("Making predictions...")
predictions = trainer.predict(tft_module, data_module)

# Display some statistics about the predictions
if predictions:
    pred_tensor = torch.cat([p[0] for p in predictions], dim=0)  # Concatenate all predictions
    print("Prediction shape: {}".format(pred_tensor.shape))
    print("Prediction quantiles: {}".format(quantiles))
    print("Sample prediction (first sample, all time steps, median quantile):")
    print(pred_tensor[0, :, 0, 1].detach().cpu().numpy())  # First sample, all time steps, median quantile
