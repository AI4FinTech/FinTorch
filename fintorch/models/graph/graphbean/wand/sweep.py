import argparse
import os

import lightning as L
import lightning.pytorch as pl
import optuna
import torch
from packaging import version

# import wandb
from fintorch.datasets.ellipticpp import EllipticppDataModule
from fintorch.graph.layers.beanconv import BEANConvSimple
from fintorch.models.graph.graphbean.graphBEAN import GraphBEANModule

torch.set_float32_matmul_precision('medium')

if version.parse(pl.__version__) < version.parse("1.6.0"):
    raise RuntimeError(
        "PyTorch Lightning>=1.6.0 is required for this example.")

PERCENT_VALID_EXAMPLES = 0.1
BATCHSIZE = 128
CLASSES = 10
EPOCHS = 10
DIR = os.getcwd()


def objective(trial: optuna.trial.Trial) -> float:
    # We optimize the number of layers, hidden units in each layer and dropouts.
    encoder_layers = trial.suggest_int("encoder_layers", 1, 3)
    decoder_layers = trial.suggest_int("decoder_layers", 1, 3)
    learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.001)
    # output_dims = [
    #     trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True) for i in range(n_layers)
    # ]

    model = GraphBEANModule(
        ("wallets", "to", "transactions"),
        edge_types=[("wallets_to_transactions"), ("transactions_to_wallets")],
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        learning_rate=learning_rate,
        class_head_layers=3,
        hidden_layers=10,
        conv_type=BEANConvSimple,
        classifier=True,
        predict='transactions',
    )
    datamodule = EllipticppDataModule(("wallets", "to", "transactions"))

    trainer = L.Trainer(
        enable_checkpointing=False,
        max_epochs=1,
        accelerator="auto",
        devices=1,
        logger=L.pytorch.loggers.WandbLogger(project="graphbean-wallets"),
        # callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc_step")],
    )
    hyperparameters = dict(encoder_layers=encoder_layers,
                           decoder_layers=decoder_layers,
                           learning_rate=learning_rate)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=datamodule)

    print(f'Return: {trainer.callback_metrics["val_acc_step"].item()}')
    return trainer.callback_metrics["val_acc_step"].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Lightning example.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    args = parser.parse_args()

    wandb_kwargs = {"project": "graphbean-wallets"}
    # wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)

    pruner = optuna.pruners.MedianPruner(
    ) if args.pruning else optuna.pruners.NopPruner()

    study = optuna.create_study(direction="maximize",
                                pruner=optuna.pruners.NopPruner())
    study.optimize(objective, n_trials=100)
    # study.optimize(objective, n_trials=100, callbacks=[wandbc])

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
