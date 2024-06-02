# """Console script for fintorch."""

import click
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.cli import LightningCLI

from fintorch.datasets import elliptic as e
from fintorch.datasets import ellipticpp as epp
from fintorch.datasets.ellipticpp import EllipticppDataModule


@click.group()
def fintorch():
    """FinTorch CLI - Your financial AI toolkit"""
    pass


@fintorch.command()
@click.argument("dataset")
def datasets(dataset):
    """Download financial datasets"""
    # Implement your dataset download logic here
    click.echo(f"Downloading dataset: {dataset}")
    if dataset == "elliptic":
        e.TransactionDataset("~/.fintorch_data", force_reload=True)
    elif dataset == "ellipticpp":
        epp.TransactionActorDataset("~/.fintorch_data", force_reload=True)


@fintorch.command()
@click.argument("model")
def train(model):
    """Train financial models"""
    # Implement your model training logic here
    click.echo(f"Training model: {model}")
    cli = LightningCLI(
        datamodule_class=EllipticppDataModule,
        run=False,
        save_config_callback=None,
        trainer_defaults={
            "max_epochs":
            10,
            "callbacks": [
                ModelCheckpoint(monitor="val_loss", mode="min"),
                EarlyStopping(monitor="train_loss", mode="min"),
            ],
        },
    )

    cli.trainer.fit(cli.model, datamodule=cli.datamodule)


# if __name__ == "__main__":
#     fintorch()

# from lightning.pytorch.callbacks import ModelCheckpoint
# from lightning.pytorch.callbacks.early_stopping import EarlyStopping
# from lightning.pytorch.cli import LightningCLI

# from fintorch.datasets.ellipticpp import EllipticppDataModule


def trainer():
    cli = LightningCLI(
        run=False,
        save_config_callback=None,
        trainer_defaults={
            "max_epochs":
            10,
            "callbacks": [
                ModelCheckpoint(monitor="val_loss", mode="min"),
                EarlyStopping(monitor="train_loss", mode="min"),
            ],
        },
    )

    cli.trainer.fit(cli.model, datamodule=cli.datamodule)


# if __name__ == "__main__":
#     main()
