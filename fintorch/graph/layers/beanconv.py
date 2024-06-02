from typing import List, Optional, Union

import torch
from torch import Tensor
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size


class BeanAggregation(Aggregation):
    """
    BeanAggregation class performs node and edge aggregation and concatenates the outputs.

    Args:
        Aggregation: The base class for aggregation.

    Attributes:
        None

    Methods:
        forward: Performs the forward pass of the aggregation layer.

    """

    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        edge_attr=None,
        **kwargs,
    ) -> Tensor:
        """
        Performs the forward pass of the aggregation layer.

        Args:
            x (Tensor): The input tensor.
            index (Optional[Tensor]): The index tensor.
            ptr (Optional[Tensor]): The pointer tensor.
            dim_size (Optional[int]): The size of the dimension.
            dim (int): The dimension along which to perform the aggregation. Default is -2.
            edge_attr: The edge attribute tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: The concatenated output tensor.

        """

        # Node aggregation
        output_mean = self.reduce(x, index, ptr, dim_size, dim, reduce="mean")
        output_sum = self.reduce(x, index, ptr, dim_size, dim, reduce="max")

        # Edge aggregation
        output_edge_mean = self.reduce(edge_attr,
                                       index,
                                       ptr,
                                       dim_size,
                                       dim,
                                       reduce="mean")
        output_edge_sum = self.reduce(edge_attr,
                                      index,
                                      ptr,
                                      dim_size,
                                      dim,
                                      reduce="max")

        # Concatenate outputs
        output = torch.cat(
            (output_mean, output_sum, output_edge_mean, output_edge_sum),
            dim=1)

        return output


class BeanAggregationSimple(Aggregation):
    """
    BeanAggregation class performs node and edge aggregation and concatenates the outputs.

    Args:
        Aggregation: The base class for aggregation.

    Attributes:
        None

    Methods:
        forward: Performs the forward pass of the aggregation layer.

    """

    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        **kwargs,
    ) -> Tensor:
        """
        Performs the forward pass of the aggregation layer.

        Args:
            x (Tensor): The input tensor.
            index (Optional[Tensor]): The index tensor.
            ptr (Optional[Tensor]): The pointer tensor.
            dim_size (Optional[int]): The size of the dimension.
            dim (int): The dimension along which to perform the aggregation. Default is -2.
            edge_attr: The edge attribute tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: The concatenated output tensor.

        """

        # Node aggregation
        output_mean = self.reduce(x, index, ptr, dim_size, dim, reduce="mean")
        output_sum = self.reduce(x, index, ptr, dim_size, dim, reduce="max")

        # Concatenate outputs
        output = torch.cat((output_mean, output_sum), dim=1)

        return output


class BEANConvSimple(MessagePassing):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: int = True,
        normalize: bool = True,
        node_self_loop: bool = True,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        **kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.normalize = normalize
        self.node_self_loop = node_self_loop

        aggr = BeanAggregationSimple()

        super().__init__(aggr, **kwargs)
        self.lin_l = Linear(-1, out_channels, bias=bias)

        if self.normalize:
            self.node_normalize = torch.nn.BatchNorm1d(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets the parameters of the layer.
        """
        super().reset_parameters()
        self.lin_l.reset_parameters()
        if self.normalize:
            self.node_normalize.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        size: Size = None,
    ):
        """
        Performs a forward pass of the layer.

        Args:
            x (Tensor or OptPairTensor): The input node features.
            edge_index (Adj): The adjacency matrix of the graph.
            size (Size, optional): The size of the graph. Defaults to None.
            edge_attr (Tensor, optional): The edge attributes. Defaults to None.

        Returns:
            output (Tensor): The output node features.
            edge_attr (Tensor): The updated edge attributes.

        """

        output = self.propagate(edge_index, x=x)
        if self.node_self_loop:
            output = torch.cat((x[1], output), dim=1)

        output = self.lin_l(output)

        if self.normalize:
            output = self.node_normalize(output)

        return output

    def __repr__(self) -> str:
        """
        Returns a string representation of the layer.

        Returns:
            str: The string representation of the layer.

        """
        return (f"{self.__class__.__name__}({self.in_channels}, "
                f"{self.out_channels}, aggr={self.aggr})")


class BEANConv(MessagePassing):
    """
    The BEANConv class represents a graph convolutional layer based on the BEAN algorithm.

    Args:
        in_channels (int): The number of input node features.
        out_channels (int): The number of output node features.
        edge_in_channels (int): The number of input edge features.
        edge_out_channels (int, optional): The number of output edge features. Defaults to -1.
        bias (bool, optional): If set to True, enables the use of a bias term. Defaults to True.
        normalize (bool, optional): If set to True, applies batch normalization to the node and edge features.
        Defaults to True.
        node_self_loop (bool, optional): If set to True, includes self-loops in the graph. Defaults to True.
        aggr (str, List[str], Aggregation, optional): The aggregation method to use. Defaults to "mean".
        **kwargs: Additional keyword arguments.

    Attributes:
        in_channels (int): The number of input node features.
        out_channels (int): The number of output node features.
        bias (bool): If set to True, enables the use of a bias term.
        normalize (bool): If set to True, applies batch normalization to the node and edge features.
        node_self_loop (bool): If set to True, includes self-loops in the graph.
        edge_out_channels (int): The number of output edge features.
        edge_in_channels (int): The number of input edge features.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_in_channels: int = -1,
        edge_out_channels: int = -1,
        bias: int = True,
        normalize: bool = True,
        node_self_loop: bool = True,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        has_edge_features=True,
        **kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.normalize = normalize
        self.node_self_loop = node_self_loop
        self.edge_out_channels = edge_out_channels
        self.edge_in_channels = edge_in_channels

        self.has_edge_features = has_edge_features

        aggr = BeanAggregation()

        super().__init__(aggr, **kwargs)
        self.lin_l = Linear(-1, out_channels, bias=bias)

        if self.normalize:
            self.node_normalize = torch.nn.BatchNorm1d(out_channels)
            self.edge_normalize = torch.nn.BatchNorm1d(edge_in_channels)

        if edge_out_channels != -1:
            self.edge_layer = Linear(-1, edge_out_channels, bias=bias)
            self.edge_normalize = torch.nn.BatchNorm1d(edge_out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets the parameters of the layer.
        """
        super().reset_parameters()
        self.lin_l.reset_parameters()
        if self.normalize:
            self.node_normalize.reset_parameters()
            self.edge_normalize.reset_parameters()

        if self.edge_out_channels != -1:
            self.edge_layer.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        size: Size = None,
        edge_attr=None,
    ):
        """
        Performs a forward pass of the layer.

        Args:
            x (Tensor or OptPairTensor): The input node features.
            edge_index (Adj): The adjacency matrix of the graph.
            size (Size, optional): The size of the graph. Defaults to None.
            edge_attr (Tensor, optional): The edge attributes. Defaults to None.

        Returns:
            output (Tensor): The output node features.
            edge_attr (Tensor): The updated edge attributes.

        """

        output = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        edge_attr = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)

        if self.node_self_loop:
            output = torch.cat((x[1], output), dim=1)

        output = self.lin_l(output)

        if self.edge_out_channels != -1 and edge_attr is not None:
            edge_attr = self.edge_layer(edge_attr)

        if self.normalize:
            output = self.node_normalize(output)
            edge_attr = self.edge_normalize(edge_attr)

        return output, edge_attr

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        edge_attr: Tensor = None,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        """
        Aggregates the messages from neighboring nodes.

        Args:
            inputs (Tensor): The input messages.
            index (Tensor): The indices of the neighboring nodes.
            ptr (Tensor, optional): The pointer tensor. Defaults to None.
            edge_attr (Tensor, optional): The edge attributes. Defaults to None.
            dim_size (int, optional): The size of the dimension. Defaults to None.

        Returns:
            Tensor: The aggregated messages.

        """
        # Overwrite standard aggr_module call
        return self.aggr_module(inputs,
                                index,
                                ptr,
                                dim_size,
                                edge_attr=edge_attr)

    def edge_update(self, x_j: Tensor, x_i: Tensor) -> Tensor:
        """
        Updates the edge attributes.

        Args:
            x_j (Tensor): The input node features of the neighboring nodes.
            x_i (Tensor): The input node features.

        Returns:
            Tensor: The updated edge attributes.

        """
        return torch.cat((x_i, x_j), dim=1)

    def __repr__(self) -> str:
        """
        Returns a string representation of the layer.

        Returns:
            str: The string representation of the layer.

        """
        return (f"{self.__class__.__name__}({self.in_channels}, "
                f"{self.out_channels}, aggr={self.aggr})")
