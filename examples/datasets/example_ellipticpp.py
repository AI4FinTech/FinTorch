import torch
from torch_geometric.nn import SAGEConv, to_hetero

from fintorch.datasets import ellipticpp

# Load the elliptic dataset
actor_transaction_graph = ellipticpp.TransactionActorDataset("~/.fintorch_data")

data = actor_transaction_graph[0]


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


model = GNN(hidden_channels=64, out_channels=3)
model = to_hetero(model, data.metadata(), aggr="sum")

with torch.no_grad():
    out = model(data.x_dict, data.edge_index_dict)
    print(out)
