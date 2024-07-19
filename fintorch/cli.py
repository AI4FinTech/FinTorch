# """Console script for fintorch."""

import click
import optuna
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.cli import LightningCLI
from optuna.integration import WeightsAndBiasesCallback

from fintorch.datasets import elliptic as e
from fintorch.datasets import ellipticpp as epp
from fintorch.models.graph.graphbean.wand.sweep import objective


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
@click.option("--model", required=True, help="Name of the model to use.")
@click.option("--predict", required=True, help="Type of prediction to perform.")
@click.option("--max_epochs", required=False, help="Max epochs")
def sweep(model, predict, max_epochs):
    """Sweep your financial models"""
    # Implement your model training logic here
    click.echo(f"Starting sweep for model: {model}")

    if model == "graphbean_elliptic":
        wandb_kwargs = {"project": f"graphbean-{predict}"}
        wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)

        study = optuna.create_study(
            direction="maximize", pruner=optuna.pruners.MedianPruner()
        )

        def wrapped_objective(trial):
            return objective(trial, max_epochs, predict)

        # study.optimize(objective, n_trials=100)
        study.optimize(wrapped_objective, n_trials=100, callbacks=[wandbc])

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))


def trainer():
    cli = LightningCLI(
        run=False,
        save_config_callback=None,
        trainer_defaults={
            "max_epochs": 10,
            "callbacks": [
                ModelCheckpoint(monitor="val_loss", mode="min"),
                EarlyStopping(monitor="train_loss", mode="min"),
            ],
        },
    )

    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
