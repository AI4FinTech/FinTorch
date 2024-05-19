from typing import Any, List, Optional, Union

import torch
from torch import Tensor
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from torch_geometric.utils import spmm

VERBOSE = False


class MyMeanAggregation(Aggregation):
    r"""Custom aggregator"""

    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
    ) -> Tensor:
        if VERBOSE:
            print(
                f"MyMeanAggregation..... x: {x.shape}, index: {index}, ptr: {ptr}, dim_size: {dim_size}, dim: {dim}"
            )
        output_mean = self.reduce(x, index, ptr, dim_size, dim, reduce="mean")
        output_sum = self.reduce(x, index, ptr, dim_size, dim, reduce="sum")
        output = torch.cat((output_mean, output_sum), dim=1)
        if VERBOSE:
            print(f"output dim:{output.shape}")
        return output


class BEANConv(MessagePassing):

    def __init__(
        self,
        in_channels,
        out_channels,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        **kwargs,
    ):

        self.in_channels = in_channels
        self.out_channels = out_channels
        aggr = MyMeanAggregation()
        super().__init__(aggr, **kwargs)
        if VERBOSE:
            print(f"in:{in_channels}")
        self.lin_l = Linear(-1, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_l.reset_parameters()

    def forward(self,
                x: Union[Tensor, OptPairTensor],
                edge_index: Adj,
                size: Size = None):
        if VERBOSE:
            print(
                f"BEANConv.forward x:{x[0].shape, x[1].shape} edge_index:{edge_index.shape}"
            )

        output = self.propagate(edge_index, x=x, size=size)
        if VERBOSE:
            print(f"Output:{output.shape}")
            print(self.output_shape, self.out_channels)
        return self.lin_l(output)

    def message(self, x_j: Tensor) -> Tensor:
        if VERBOSE:
            print(f"BEANConv.message x_j:{x_j.shape}")
        return super().message(x_j)

    def propagate(self,
                  edge_index: Adj,
                  size: Size = None,
                  **kwargs: Any) -> Tensor:
        if VERBOSE:
            print(
                f"BEANConv.propagate edge_index:{edge_index.shape} size:{size}"
            )
        return super().propagate(edge_index, size, **kwargs)

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor,
                              size: Size) -> Tensor:
        if VERBOSE:
            print(
                f"BEANConv.message_and_aggregate adj_t:{adj_t.sparse_sizes} x0:{x[0].shape} x1:{x[1].shape} size:{size}"
            )
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)

    def update(self, inputs: Tensor) -> Tensor:
        if VERBOSE:
            print(f"BEANConv.update inputs:{inputs.shape}")
        self.output_shape = inputs.size(0)
        return super().update(inputs)

    def edge_updater(self,
                     edge_index: Adj,
                     size: Size = None,
                     **kwargs: Any) -> Tensor:
        if VERBOSE:
            print("BEANConv.edge_updater")
        return super().edge_updater(edge_index, size, **kwargs)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}({self.in_channels}, "
                f"{self.out_channels}, aggr={self.aggr})")


# class MyMeanAggregation(Aggregation):
#     r"""An aggregation operator that averages features across a set of
#     elements.

#     .. math::
#         \mathrm{mean}(\mathcal{X}) = \frac{1}{|\mathcal{X}|}
#         \sum_{\mathbf{x}_i \in \mathcal{X}} \mathbf{x}_i.
#     """

#     def forward(
#         self,
#         x: Tensor,
#         index: Optional[Tensor] = None,
#         ptr: Optional[Tensor] = None,
#         dim_size: Optional[int] = None,
#         dim: int = -2,
#     ) -> Tensor:
#         print(
#             f"MyMeanAggregation..... x: {x.shape}, index: {index}, ptr: {ptr}, dim_size: {dim_size}, dim: {dim}"
#         )

#         output_mean = self.reduce(x, index, ptr, dim_size, dim, reduce="mean")
#         output_sum = self.reduce(x, index, ptr, dim_size, dim, reduce="sum")

#         output = torch.cat((output_mean, output_sum), dim=1)

#         print(f"output dim:{output.shape}")
#         return output

# class BEANConv(MessagePassing):

#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
#         **kwargs,
#     ):

#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         aggr = MyMeanAggregation()

#         super().__init__(aggr, **kwargs)
#         print(f"in:{in_channels}")
#         self.lin_l = Linear(-1, out_channels)

#         self.reset_parameters()

#     def reset_parameters(self):
#         super().reset_parameters()
#         self.lin_l.reset_parameters()

#     def forward(
#         self, x: Union[Tensor, OptPairTensor], edge_index: Adj, size: Size = None
#     ):
#         print(
#             f"BEANConv.forward x:{x[0].shape, x[1].shape} edge_index:{edge_index.shape}"
#         )

#         # propagate flexible arguments
#         # x = features
#         # size = size
#         # the message_and_aggregate function should align with this signature

#         adj = SparseTensor(
#             row=edge_index[0],
#             col=edge_index[1],
#             sparse_sizes=(x[1].shape[0], x[1].shape[0]),
#             trust_data=True,
#         )

#         output = self.propagate(edge_index, x=x, size=size)
#         print(f"Output:{output.shape}")

#         # print(f"out_channesl:{self.out_channels}")
#         print(self.output_shape, self.out_channels)

#         return self.lin_l(output)

#     # TODO: ensure that the aggregation function returns something with the same dimensionality as the number of nodes
#     # def aggregate(self, x, edge_index):
#     #     print(f"BEANConv.aggregate x:{x.shape} edge_index:{edge_index.shape}")
#     #     return self.aggr(x, edge_index[1])

#     def message(self, x_j: Tensor) -> Tensor:
#         print(f"BEANConv.message x_j:{x_j.shape}")
#         return super().message(x_j)

#     def propagate(self, edge_index: Adj, size: Size = None, **kwargs: Any) -> Tensor:
#         print(f"BEANConv.propagate edge_index:{edge_index.shape} size:{size}")
#         return super().propagate(edge_index, size, **kwargs)

#     def message_and_aggregate(
#         self, adj_t: SparseTensor, x: Tensor, size: Size
#     ) -> Tensor:
#         """
#         Performs message passing and aggregation in a memory-efficient manner
#         using a sparse edge_index.

#         Args:
#             adj_t (SparseTensor): The sparse adjacency tensor representing the graph.
#             x (Tensor): The input tensor.
#             size (Size): The size of the output tensor.

#         Returns:
#             Tensor: The aggregated tensor after message passing.

#         Raises:
#             NotImplementedError: This method is not implemented yet.
#         """
#         print(
#             f"BEANConv.message_and_aggregate adj_t:{adj_t.sparse_sizes} x0:{x[0].shape} x1:{x[1].shape} size:{size}"
#         )

#         # msg_mean = matmul(adj_t, x[0], reduce="mean")
#         # msg_sum = matmul(adj_t, x[0], reduce="sum")

#         # print(f"msg_mean:{msg_mean.shape} msg_sum:{msg_sum.shape}")

#         # msg = torch.cat((msg_mean, msg_sum), dim=1)

#         # print(f"msg:{msg.shape}")

#         # return msg
#         if isinstance(adj_t, SparseTensor):
#             adj_t = adj_t.set_value(None, layout=None)
#         return spmm(adj_t, x[0], reduce=self.aggr)

#     def update(self, inputs: Tensor) -> Tensor:
#         print(f"BEANConv.update inputs:{inputs.shape}")
#         self.output_shape = inputs.size(0)
#         return super().update(inputs)

#     def edge_updater(self, edge_index: Adj, size: Size = None, **kwargs: Any) -> Tensor:
#         print("BEANConv.edge_updater")
#         return super().edge_updater(edge_index, size, **kwargs)

#     def __repr__(self) -> str:
#         return (
#             f"{self.__class__.__name__}({self.in_channels}, "
#             f"{self.out_channels}, aggr={self.aggr})"
#         )
