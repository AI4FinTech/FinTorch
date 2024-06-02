import argparse

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.cli import LightningCLI

from fintorch.datasets.ellipticpp import EllipticDataModule
from fintorch.training.GraphBeanTrainer import GraphBEANModule

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_layers", type=int)
    parser.add_argument("--decoder_layers", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--hidden_layers", type=int)
    parser.add_argument("--class_head_layers", type=int)
    parser.add_argument("--predict", type=str)
    args = parser.parse_args()

    # wandb.init(project='graph')
    # TODO: how does this work for LightnigCLI?
    lightning_model = GraphBEANModule(
        ("transactions", "to", "transactions"),
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        learning_rate=args.learning_rate,
        class_head_layers=args.class_head_layers,
        hidden_layers=args.hidden_layers,
        classifier=True,
        predict=args.predict,
    )
    datamodule = EllipticDataModule(("transactions", "to", "transactions"))
    # trainer = Trainer(
    #     callbacks=[
    #         ModelCheckpoint(monitor="val_loss", mode="min"),
    #         EarlyStopping(monitor="train_loss", mode="min"),
    #     ],
    #     logger=L.pytorch.loggers.WandbLogger(project="graph"),
    #     max_epochs=5,
    #     accelerator="gpu",
    # )
    # trainer.fit(lightning_model, datamodule=datamodule)

    cli = LightningCLI(
        model_class=GraphBEANModule,
        datamodule_class=EllipticDataModule,
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
