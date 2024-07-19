from typing import List, Optional, Union

import torch
from torch import Tensor
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size


class BeanAggregation(Aggregation):
    """
    BeanAggregation class performs node and edge aggregation operations.

    Reference:

    Fathony, Rizal, Jenn Ng, and Jia Chen. 2023. “Interaction-Focused Anomaly Detection on Bipartite
    Node-and-Edge-Attributed Graphs.” In 2023 International Joint Conference on Neural Networks (IJCNN), 1–10. IEEE.

    Args:
        x (Tensor): The input tensor.
        index (Optional[Tensor]): The index tensor. Default is None.
        ptr (Optional[Tensor]): The pointer tensor. Default is None.
        dim_size (Optional[int]): The size of the dimension. Default is None.
        dim (int): The dimension along which to perform the aggregation. Default is -2.
        edge_attr: The edge attribute tensor. Default is None.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The aggregated output tensor.

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
        Forward pass of the BeanAggregation layer.

        Args:
            x (Tensor): The input tensor.
            index (Optional[Tensor]): The index tensor. Default is None.
            ptr (Optional[Tensor]): The pointer tensor. Default is None.
            dim_size (Optional[int]): The size of the dimension. Default is None.
            dim (int): The dimension along which to perform the aggregation. Default is -2.
            edge_attr: The edge attribute tensor. Default is None.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: The aggregated output tensor.

        """

        # Node aggregation
        output_mean = self.reduce(x, index, ptr, dim_size, dim, reduce="mean")
        output_sum = self.reduce(x, index, ptr, dim_size, dim, reduce="max")

        # Edge aggregation
        output_edge_mean = self.reduce(
            edge_attr, index, ptr, dim_size, dim, reduce="mean"
        )
        output_edge_sum = self.reduce(
            edge_attr, index, ptr, dim_size, dim, reduce="max"
        )

        # Concatenate outputs
        output = torch.cat(
            (output_mean, output_sum, output_edge_mean, output_edge_sum), dim=1
        )

        return output


class BeanAggregationSimple(Aggregation):
    """
    A simple bean aggregation layer (NOTE: only use this one if you don't have edge attributes).

    This layer performs node aggregation by calculating the mean and maximum values
    along a specified dimension of the input tensor. The outputs are then concatenated
    along the same dimension. Use this Aggregator if you don't have edge features!

    Reference:

    Fathony, Rizal, Jenn Ng, and Jia Chen. 2023. “Interaction-Focused Anomaly Detection on Bipartite
    Node-and-Edge-Attributed Graphs.” In 2023 International Joint Conference on Neural Networks (IJCNN), 1–10. IEEE.

    Args:
        x (Tensor): The input tensor.
        index (Optional[Tensor]): The index tensor for sparse input. Default is None.
        ptr (Optional[Tensor]): The pointer tensor for sparse input. Default is None.
        dim_size (Optional[int]): The size of the dimension along which to perform aggregation.
            Default is None.
        dim (int): The dimension along which to perform aggregation. Default is -2.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: The concatenated output tensor.

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
        Forward pass of the bean aggregation layer.

        Args:
            x (Tensor): The input tensor.
            index (Optional[Tensor]): The index tensor for sparse input. Default is None.
            ptr (Optional[Tensor]): The pointer tensor for sparse input. Default is None.
            dim_size (Optional[int]): The size of the dimension along which to perform aggregation.
                Default is None.
            dim (int): The dimension along which to perform aggregation. Default is -2.
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
    """
    BEANConvSimple is a graph convolutional layer. In comparison with the BEANConv layer, this layer does not
    use edge attributes.

    Reference:

    Fathony, Rizal, Jenn Ng, and Jia Chen. 2023. “Interaction-Focused Anomaly Detection on Bipartite
    Node-and-Edge-Attributed Graphs.” In 2023 International Joint Conference on Neural Networks (IJCNN), 1–10. IEEE.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        bias (bool, optional): If set to True, enables the learnable bias. Default is True.
        normalize (bool, optional): If set to True, applies batch normalization to the output. Default is True.
        node_self_loop (bool, optional): If set to True, includes the node's own features in the output.
        Default is True.
        aggr (str, List[str], Aggregation, optional): The aggregation method to use. Default is "mean".
        **kwargs: Additional keyword arguments to be passed to the parent class.

    Attributes:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        bias (bool): If True, enables the learnable bias.
        normalize (bool): If True, applies batch normalization to the output.
        node_self_loop (bool): If True, includes the node's own features in the output.
        lin_l (Linear): Linear layer for transforming the output.
        node_normalize (BatchNorm1d): Batch normalization layer for normalizing the output.

    Methods:
        reset_parameters(): Resets the parameters of the layer.
        forward(x, edge_index, size=None): Performs the forward pass of the layer.
        __repr__(): Returns a string representation of the layer.

    """

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
        Performs the forward pass of the layer.

        Args:
            x (Tensor or OptPairTensor): The input node features.
            edge_index (Adj): The adjacency matrix of the graph.
            size (Size, optional): The size of the graph. Default is None.

        Returns:
            Tensor: The output node features.

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
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, aggr={self.aggr})"
        )


class BEANConv(MessagePassing):
    r"""
    BEANConv is a message-passing graph convolutional layer implementation.

    Reference:

    Fathony, Rizal, Jenn Ng, and Jia Chen. 2023. “Interaction-Focused Anomaly Detection on Bipartite
    Node-and-Edge-Attributed Graphs.” In 2023 International Joint Conference on Neural Networks (IJCNN), 1–10. IEEE.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        edge_in_channels (int, optional): Number of input channels for edge features. Default is -1.
        edge_out_channels (int, optional): Number of output channels for edge features. Default is -1.
        bias (bool, optional): If set to True, enables the learnable bias. Default is True.
        normalize (bool, optional): If set to True, applies batch normalization to node and edge features.
        Default is True.
        node_self_loop (bool, optional): If set to True, includes self-loops in the graph. Default is True.
        aggr (str, List[str], Aggregation, optional): The aggregation method to use for message passing.
        Default is "mean".
        has_edge_features (bool, optional): If set to True, indicates that the graph has edge features. Default is True.

    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bias (bool): If True, enables the learnable bias.
        normalize (bool): If True, applies batch normalization to node and edge features.
        node_self_loop (bool): If True, includes self-loops in the graph.
        edge_out_channels (int): Number of output channels for edge features.
        edge_in_channels (int): Number of input channels for edge features.
        has_edge_features (bool): If True, indicates that the graph has edge features.

    Methods:
        reset_parameters(): Resets the parameters of the BEANConv layer.
        forward(x, edge_index, size, edge_attr): Performs a forward pass of the BEANConv layer.
        aggregate(inputs, index, ptr, edge_attr, dim_size): Aggregates the input features according to
        the specified aggregation method.
        edge_update(x_j, x_i): Updates the edge features based on the input node features.
        __repr__(): Returns a string representation of the BEANConv layer.

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
        r"""
        Resets the parameters of the BEANConv layer.
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
        r"""
        Performs a forward pass of the BEANConv layer.

        Args:
            x (Tensor or OptPairTensor): Node feature matrix or a tuple of node feature matrices.
            edge_index (Adj): Graph connectivity in COO format.
            size (Size, optional): The size of the graph. Default is None.
            edge_attr (Tensor, optional): Edge feature matrix. Default is None.

        Returns:
            output (Tensor): The output node feature matrix.
            edge_attr (Tensor): The output edge feature matrix.

        """
        # propagate node information
        output = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # propagate edge information
        edge_attr = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)

        if self.node_self_loop:
            # Add the node's own features
            output = torch.cat((x[1], output), dim=1)

        # Node projection
        output = self.lin_l(output)

        # Edge projection
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
        r"""
        Aggregates the input features according to the specified aggregation method, in this case we added
        the edge_attr as an additional parameter to facilitate edge message passing to nodes.

        Args:
            inputs (Tensor): The input features to aggregate.
            index (Tensor): The indices of the nodes to aggregate.
            ptr (Tensor, optional): The pointer tensor pointing to the start and end indices of the nodes.
                Default is None.
            edge_attr (Tensor, optional): The edge features to aggregate. Default is None.
            dim_size (int, optional): The size of the dimension to aggregate. Default is None.

        Returns:
            Tensor: The aggregated output features.

        """
        # Overwrite standard aggr_module call
        return self.aggr_module(inputs, index, ptr, dim_size, edge_attr=edge_attr)

    def edge_update(self, x_j: Tensor, x_i: Tensor) -> Tensor:
        r"""
        Updates the edge features based on the input node features.

        Args:
            x_j (Tensor): The input features of the neighboring nodes.
            x_i (Tensor): The input features of the central nodes.

        Returns:
            Tensor: The updated edge features.

        """
        return torch.cat((x_i, x_j), dim=1)

    def __repr__(self) -> str:
        r"""
        Returns a string representation of the BEANConv layer.

        Returns:
            str: The string representation of the BEANConv layer.

        """
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, aggr={self.aggr})"
        )
