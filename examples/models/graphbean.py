import pytorch_lightning as pl
from torch import optim
from torch_geometric.datasets import OGB_MAG

from fintorch.models.graph import graphBEAN
from fintorch.models.graph.graphBEAN import GraphBEANLoss
from fintorch.training.GraphBeanTrainer import GraphBEANModule

# Define your GraphBEAN model, loss function, and optimizer
model = graphBEAN()
loss_fn = GraphBEANLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create an instance of the GraphBEANModule
module = GraphBEANModule(model, loss_fn, optimizer)

dataset = OGB_MAG(root="./data", preprocess="metapath2vec")
data = dataset[0]

# Create a PyTorch Lightning Trainer and train the module
trainer = pl.Trainer(max_epochs=10)

# Use the dataset to create train and validation dataloaders
train_dataloader = dataset["train"].loader(batch_size=32, shuffle=True)
val_dataloader = dataset["val"].loader(batch_size=32, shuffle=False)

# Train the module using the dataloaders
trainer.fit(module, train_dataloader, val_dataloader)
