import subprocess

import lightning as L
import torch
import torch.nn as nn

from fintorch.datasets.ellipticpp import EllipticppDataModule
from fintorch.graph.layers.beanconv import BEANConvSimple
from fintorch.models.graph.graphbean.graphBEAN import GraphBEANModule


def install(package):
    subprocess.check_call(["pip", "install", package])


def install_pyg_and_dependencies():
    install(
        f"pyg-lib -f https://data.pyg.org/whl/torch-{torch.__version__}.html")
    install(
        f"torch-scatter -f https://data.pyg.org/whl/torch-{torch.__version__}.html"
    )
    install(
        f"torch-sparse -f https://data.pyg.org/whl/torch-{torch.__version__}.html"
    )


# Detect PyTorch version
if torch.__version__ >= "1.13.0":
    print(
        "PyTorch version 1.13.0 or newer detected. Installing PyG and dependencies..."
    )
    install_pyg_and_dependencies()
else:
    print(
        "PyTorch version is older than 1.13.0. PyG might not work correctly. Please upgrade PyTorch"
    )

# Verify installation
try:
    import torch_geometric

    print(
        f"PyTorch Geometric successfully installed. Version: {torch_geometric.__version__}"
    )
except ImportError:
    print("PyTorch Geometric not found. Installation might have failed.")


def test_graphbeanmodule_init():
    edge = ("type1", "to", "type2")
    edge_types = [("type1", "to", "type2")]
    mapping = {"type1": 64, "type2": 128}
    loss_fn = nn.MSELoss()
    learning_rate = 0.01
    encoder_layers = 2
    decoder_layers = 2
    hidden_layers = 128
    structure_decoder_head_out_channels = 50
    structure_decoder_head_layers = 10
    classifier = False
    class_head_layers = 3
    classes = 2
    predict = None
    conv_type = BEANConvSimple
    aggr = "sum"
    node_types = ["type1", "type2"]
    model = GraphBEANModule(
        edge,
        edge_types,
        mapping,
        loss_fn=loss_fn,
        learning_rate=learning_rate,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        hidden_layers=hidden_layers,
        structure_decoder_head_out_channels=structure_decoder_head_out_channels,
        structure_decoder_head_layers=structure_decoder_head_layers,
        classifier=classifier,
        class_head_layers=class_head_layers,
        classes=classes,
        predict=predict,
        conv_type=conv_type,
        aggr=aggr,
        node_types=node_types,
    )

    assert model.edge == edge
    assert model.edge_types == edge_types
    assert model.mapping == mapping
    assert model.loss_fn == loss_fn
    assert model.learning_rate == learning_rate
    assert model.encoder_layers == encoder_layers
    assert model.decoder_layers == decoder_layers
    assert model.hidden_layers == hidden_layers
    assert (model.structure_decoder_head_out_channels ==
            structure_decoder_head_out_channels)
    assert model.structure_decoder_head_layers == structure_decoder_head_layers
    assert model.classifier == classifier
    assert model.class_head_layers == class_head_layers
    assert model.classes == classes
    assert model.predict == predict
    assert model.conv_type == conv_type
    assert model.aggr == aggr
    assert model.node_types == node_types


def test_graphbeanmodule_forward():
    # We use an example data module from the elliptic dataset which is bipartite
    data_module = EllipticppDataModule(("wallets", "to", "transactions"),
                                       force_reload=False)
    try:
        data_module.setup()
        # Get the dimensionalities for the auto-encoder part
        mapping = dict()
        for key in data_module.dataset.metadata()[0]:
            mapping[key] = data_module.dataset[key].x.shape[1]
    except Exception as e:
        print(f"Error setting up data module or retrieving metadata: {e}")
        raise

    # Create an instance of the GraphBEANModule
    module = GraphBEANModule(
        ("wallets", "to", "transactions"),
        edge_types=[
            ("wallets", "to", "transactions"),
            ("transactions", "to", "wallets"),
        ],
        mapping=mapping,
        learning_rate=0.001,
        conv_type=BEANConvSimple,
        encoder_layers=5,
        decoder_layers=5,
        hidden_layers=50,
    )

    # Create a PyTorch Lightning Trainer and train the module
    trainer = L.Trainer(max_epochs=1, accelerator="auto")

    # Train the module using the dataloaders
    trainer.fit(module, datamodule=data_module)

    # Check if the trainer finished training
    assert trainer.state.status, "finished"
    # Check if the edge_types are correct
    assert module.edge_types == [
        ("wallets", "to", "transactions"),
        ("transactions", "to", "wallets"),
    ]
