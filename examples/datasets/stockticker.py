from datetime import date

from fintorch.datasets import stockticker

# Load the stock dataset
tickers = ["AAPL", "MSFT", "GOOG"]
# Load the stock dataset
stockdata = stockticker.StockTicker(
    "~/.fintorch_data/stocktickers/",
    tickers=tickers,
    start_date=date(2015, 1, 1),
    end_date=date(2023, 6, 30),
)

# class GNN(torch.nn.Module):

#     def __init__(self, hidden_channels, out_channels):
#         super().__init__()
#         self.conv1 = SAGEConv((-1, -1), hidden_channels)
#         self.conv2 = SAGEConv((-1, -1), out_channels)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = self.conv2(x, edge_index)
#         return x

# model = GNN(hidden_channels=64, out_channels=3)
# model = to_hetero(model, data.metadata(), aggr='sum')

# with torch.no_grad():
#     out = model(data.x_dict, data.edge_index_dict)
#     print(out)
