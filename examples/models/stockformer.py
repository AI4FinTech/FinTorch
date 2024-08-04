import logging
from datetime import date

import lightning as L

from fintorch.datasets import stockticker
from fintorch.models.graph.stockformer.stockformer import StockFormerModule

logging.basicConfig(level=logging.INFO)


def main():
    try:
        # Load the stock dataset
        tickers = ["AAPL", "MSFT", "GOOG", "NVDA"]
        tickers_to_indices = {ticker: index for index, ticker in enumerate(tickers)}

        # Load the stock dataset
        stockdata = stockticker.StockTickerDataModule(
            tickers=tickers,
            start_date=date(2015, 1, 1),
            end_date=date(2023, 6, 30),
            mapping=tickers_to_indices,
        )

        # Construct the data
        stockdata.setup()

        model = StockFormerModule()

        trainer = L.Trainer(max_epochs=100, accelerator="auto")

        trainer.fit(model, datamodule=stockdata)
    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
