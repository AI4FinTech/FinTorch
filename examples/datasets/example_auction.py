import logging
from pathlib import Path

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS

from fintorch.datasets.auctiondata import AuctionDataset

logging.basicConfig(level=logging.INFO)

data_path = Path("~/.fintorch_data/auctiondata-optiver/").expanduser()
auction_data = AuctionDataset(data_path, False)

print(auction_data.train.describe())
print(auction_data.train.collect_schema())


# Initialize the model
model = NBEATS(
    input_size=30,
    h=7,
    futr_exog_list=["wap"],
)

# Create a NeuralForecast object
nf = NeuralForecast(models=[model], freq=1)

# Define validation and test sizes
val_size = 100  # Number of days for validation
test_size = 100  # Number of days for testing

# Perform cross-validation
Y_hat_df = nf.cross_validation(
    df=auction_data.train,
    val_size=val_size,
    test_size=test_size,
    n_windows=None,  # Uses expanding window if None
)
