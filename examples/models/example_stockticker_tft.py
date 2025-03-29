import logging
from datetime import date
from pathlib import Path
from random import randint

import lightning as L
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import EarlyStopping

from fintorch.datasets import stockticker
from fintorch.datasets.stockticker import StockTickerDataModule
from fintorch.models.timeseries.tft import TemporalFusionTransformerModule

# Set logging level to INFO
logging.basicConfig(level=logging.INFO)

# Parameters
tickers = ["AAPL"]
data_path = Path("~/.fintorch_data/stocktickers/").expanduser()
start_date = date(2015, 1, 1)
end_date = date(2024, 6, 30)

# Create a dictionary mapping from tickers to index
ticker_index = {ticker: index for index, ticker in enumerate(tickers)}

# Load the stock dataset
stockdata = stockticker.StockTicker(
    data_path,
    tickers=tickers,
    start_date=start_date,
    end_date=end_date,
    mapping=ticker_index,
    force_reload=False,
)

print(stockdata.df_timeseries_dataset.to_pandas().describe())

number_of_past_inputs = 168
number_of_future_inputs = 24
embedding_size_inputs = hidden_dimension = 80
dropout = 0.1
number_of_heads = 2
batch_size = 1024


# Create a datamodule
datamodule = StockTickerDataModule(
    data_path=data_path,
    tickers=tickers,
    start_date=start_date,
    end_date=end_date,
    ticker_index=ticker_index,
    batch_size=batch_size,
    workers=0,
    past_length=number_of_past_inputs,
    future_length=number_of_future_inputs,
)

# Setup the datamodule
datamodule.setup()

# Iterate over the first 3 batches of the train dataset
for batch_idx, (past_inputs, target) in enumerate(datamodule.train_dataloader()):
    if batch_idx >= 3:
        break
    print(f"Batch {batch_idx + 1}:")
    print(f"  Past Inputs shape: {past_inputs['past_data'].shape}")
    print(f"  Target shape: {target.shape}")


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

# Set the precision
torch.set_float32_matmul_precision("medium")

plot_all_data = datamodule.train_dataset.df_timeseries_dataset.select("y")
plot_all_data = plot_all_data.to_series().to_list()


# Plot all data
plt.figure(figsize=(15, 5))
plt.plot(plot_all_data, label="All Data")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.title("All Data")
plt.legend()
plt.show()


# Create a trainer with TensorBoard for better monitoring
early_stopping = EarlyStopping("val_loss_epoch", patience=100)
trainer = L.Trainer(max_epochs=500, callbacks=[early_stopping], log_every_n_steps=4)

# Train the model
trainer.fit(tft_module, datamodule=datamodule)

trainer.test(tft_module, datamodule=datamodule)

# Get predictions
predictions = trainer.predict(tft_module, datamodule=datamodule)

# Concatenate predictions
all_predictions = torch.cat(predictions, dim=0)

# Number of batches to plot
num_batches_to_plot = 5

# Plotting
plt.figure(figsize=(15, 5 * num_batches_to_plot))  # Adjust figure size

for idx in range(0, num_batches_to_plot):
    batch_idx = randint(0, len(all_predictions) - 1)

    selected_batch_predictions = all_predictions[batch_idx]
    _, selected_batch_target = datamodule.test_dataset[batch_idx]

    selected_batch_predictions_inverse_scaled = (
        datamodule.dataset.scaler.inverse_transform(
            selected_batch_predictions.reshape(-1, selected_batch_predictions.shape[-1])
        ).reshape(selected_batch_predictions.shape)
    )

    selected_batch_target_inverse_scaled = datamodule.dataset.scaler.inverse_transform(
        selected_batch_target.reshape(-1, 1)
    ).reshape(selected_batch_target.shape)

    plt.subplot(
        num_batches_to_plot,
        1,
        idx + 1,
    )  # Create subplots

    plt.plot(
        selected_batch_target_inverse_scaled, label="Target", marker="o", linestyle="-"
    )

    for i, quantile in enumerate(quantiles):
        plt.plot(
            selected_batch_predictions_inverse_scaled[:, 0, i],
            label=f"Predicted Quantile {quantile}",
            marker="x",
            linestyle="--",
        )

    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title(f"Target vs Predicted Quantiles for Batch {batch_idx + 1}")
    plt.legend()

plt.tight_layout()  # Adjust subplot parameters for a tight layout
plt.show()
