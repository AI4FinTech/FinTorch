import lightning as L
import torch
from torch_geometric.data import HeteroData

from fintorch.models.graph.graphBEAN import GraphBEANLoss, GraphBEANModule

# Create feature matrices for node types A and B
x = HeteroData()
x["A"].x = torch.randn(100, 10)
x["B"].x = torch.randn(200, 20)

# Add feature dimension 'y' for node type A and B with random assignment of class A, B, and C
x["A"].y = torch.randint(0, 3, (100, ))
x["B"].y = torch.randint(0, 3, (100, ))

# Add edge index for connections between nodes of A and B
# TODO: fix edge construction
edge_index = torch.randint(0, 100, (2, 200))
x["A", "to", "B"].edge_index = edge_index
x["B", "to", "A"].edge_index = edge_index

print(f"Heterodata: {x}")

# Create an instance of the GraphBEANModule
module = GraphBEANModule(
    ("A", "to", "B"),
    loss_fn=GraphBEANLoss,
    learning_rate=0.001,
    encoder_layers=2,
    data=x,
    classifier=False,
)

# Create a PyTorch Lightning Trainer and train the module
trainer = L.Trainer(max_epochs=10)

# Use the dataset to create train and validation dataloaders
train_dataloader = x
val_dataloader = x

# Train the module using the dataloaders
trainer.fit(module, x)
