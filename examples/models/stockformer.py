import logging
from datetime import date

import neuralforecast
from neuralforecast.losses.pytorch import MAE

from fintorch.datasets import stockticker
from fintorch.models.graph.stockformer.stockformer import StockFormerModule

logging.basicConfig(level=logging.INFO)


def main():

    # Load the stock dataset
    tickers = ["AAPL", "MSFT", "GOOG", "NVDA"]
    tickers_to_indices = {ticker: index for index, ticker in enumerate(tickers)}

    # Load the stock dataset
    stockdata = stockticker.StockTickerDataModule(
        tickers=tickers,
        start_date=date(2023, 1, 1),
        end_date=date(2023, 6, 30),
        mapping=tickers_to_indices,
    )

    # Construct the data
    stockdata.setup()

    model = StockFormerModule(
        h=10,
        n_series=len(tickers),
        input_size=20,
        scaler_type="robust",
        max_steps=10,
        early_stop_patience_steps=-1,
        val_check_steps=10,
        learning_rate=1e-3,
        loss=MAE(),
        valid_loss=None,
        batch_size=32,
    )

    stockdata.stocktickerdata.df_timeseries_dataset = (
        stockdata.stocktickerdata.df_timeseries_dataset.rename(
            {"unique_ticker": "unique_id"}
        )
    )

    fcst = neuralforecast.NeuralForecast(models=[model], freq="1d")
    fcst.fit(df=stockdata.stocktickerdata.df_timeseries_dataset, val_size=20)

    # dataloader = stockdata.train_dataloader()

    # print(f"Length dataset:{len(dataloader)} should be equal to the number of stocks")
    # for idx, batch in enumerate(dataloader):
    #     print(
    #         f"Dimensionalities of batch {idx}: {batch['static']}, {batch['static_cols'].shape}, {batch['temporal'].shape}, {batch['temporal_cols'].shape}, {batch['y_idx']}"
    #     )
    #     print(f"First 10 values of temporal: {batch['temporal'][:10]}")

    # model = StockFormerModule()

    # trainer = L.Trainer(max_epochs=100, accelerator="auto")

    # trainer.fit(model, datamodule=stockdata)


if __name__ == "__main__":
    main()
