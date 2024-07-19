import lightning as L
import optuna
import torch
from optuna.integration import PyTorchLightningPruningCallback

# import wandb
from fintorch.datasets.ellipticpp import EllipticppDataModule
from fintorch.graph.layers.beanconv import BEANConvSimple
from fintorch.models.graph.graphbean.graphBEAN import GraphBEANModule

torch.set_float32_matmul_precision("medium")


def objective(trial: optuna.trial.Trial, max_epochs, predict) -> float:
    # We optimize the number of layers, hidden units in each layer and dropouts.
    encoder_layers = trial.suggest_int("encoder_layers", 1, 4)
    decoder_layers = trial.suggest_int("decoder_layers", 1, 4)
    class_head_layers = trial.suggest_int("class_head_layers", 1, 10)
    hidden_layers = trial.suggest_int("hidden_layers", 8, 128)
    learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.005)
    structure_decoder_head_layers = trial.suggest_int(
        "structure_decoder_head_layers", 2, 16
    )
    structure_decoder_head_out_channel = trial.suggest_int(
        "structure_decoder_head_out_channel", 8, 64
    )

    if predict == "transactions":
        edge = ("wallets", "to", "transactions")
    else:
        edge = ("transactions", "to", "wallets")

    model = GraphBEANModule(
        edge,
        edge_types=[("wallets_to_transactions"), ("transactions_to_wallets")],
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        learning_rate=learning_rate,
        class_head_layers=class_head_layers,
        hidden_layers=hidden_layers,
        structure_decoder_head_layers=structure_decoder_head_layers,
        structure_decoder_head_out_channels=structure_decoder_head_out_channel,
        conv_type=BEANConvSimple,
        classifier=True,
        predict=predict,
    )
    datamodule = EllipticppDataModule(edge, batch_size=2056)

    trainer = L.Trainer(
        enable_checkpointing=False,
        max_epochs=50,
        accelerator="auto",
        devices=1,
        logger=L.pytorch.loggers.WandbLogger(project=f"graphbean-sweep-{predict}"),
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc_step")],
    )
    hyperparameters = dict(
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        learning_rate=learning_rate,
        class_head_layers=class_head_layers,
        hidden_layers=hidden_layers,
    )
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics["val_f1"].item()
