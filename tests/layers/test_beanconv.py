import torch
from torch_geometric.data import HeteroData

from fintorch.graph.layers.beanconv import BeanAggregation, BEANConv, BEANConvSimple


def create_hetero_data():
    torch.manual_seed(43)

    features = torch.randn(3, 3)
    edge_attributes = torch.randn(4, 4)

    data = HeteroData()

    data["U"].x = features[:1, :]
    data["V"].x = features[1:, :]

    # Edge Index (Directed Edges)
    edge_index_U_V = torch.tensor(
        [
            [0, 1],  # Source nodes (U)
            [0, 0],  # Target nodes (V)
        ]
    )

    edge_index_V_U = torch.tensor(
        [
            [0, 0],  # Source nodes (V)
            [0, 1],  # Target nodes (U)
        ]
    )

    data["U", "to", "V"].edge_index = edge_index_V_U
    data["V", "to", "U"].edge_index = edge_index_U_V

    # Set edge attributes
    data["U", "to", "V"].edge_attr = edge_attributes[:2, :]
    data["V", "to", "U"].edge_attr = edge_attributes[2:, :]

    return data


def create_hetero_data_no_edge_attr():
    torch.manual_seed(43)

    features = torch.randn(3, 3)

    data = HeteroData()

    data["U"].x = features[:1, :]
    data["V"].x = features[1:, :]

    # Edge Index (Directed Edges)
    edge_index_U_V = torch.tensor(
        [
            [0, 1],  # Source nodes (U)
            [0, 0],  # Target nodes (V)
        ]
    )

    edge_index_V_U = torch.tensor(
        [
            [0, 0],  # Source nodes (V)
            [0, 1],  # Target nodes (U)
        ]
    )

    data["U", "to", "V"].edge_index = edge_index_V_U
    data["V", "to", "U"].edge_index = edge_index_U_V

    return data


def test_BEANConv_forward_with_hetero_data():
    in_channels = 3
    out_channels = 1
    data = create_hetero_data()
    conv = BEANConv(in_channels, out_channels, edge_in_channels=6)

    x_uv = (data.x_dict["U"], data.x_dict["V"])
    x_vu = (data.x_dict["V"], data.x_dict["U"])

    # V to U
    output_nodes = conv.propagate(
        data["V", "to", "U"].edge_index,
        x=x_vu,
        edge_attr=data["V", "to", "U"].edge_attr,
    )
    output_edges = conv.edge_updater(
        data["V", "to", "U"].edge_index,
        x=x_vu,
        edge_attr=data["V", "to", "U"].edge_attr,
    )

    assert torch.allclose(
        output_nodes,
        torch.tensor(
            [
                [
                    0.9817575216293335,
                    0.9419769644737244,
                    -0.7687521576881409,
                    1.4787991046905518,
                    1.1918498277664185,
                    -0.14461655914783478,
                    -0.4200826585292816,
                    -0.12341845035552979,
                    -0.5287467241287231,
                    1.4372074604034424,
                    -0.3629981577396393,
                    1.5822296142578125,
                    -0.4429505169391632,
                    1.846213459968567,
                ]
            ]
        ),
    )
    assert torch.allclose(
        output_edges,
        torch.tensor(
            [
                [
                    -0.6484010815620422,
                    -0.7058414220809937,
                    0.6432183980941772,
                    1.4787991046905518,
                    1.1918498277664185,
                    -0.14461655914783478,
                ],
                [
                    -0.6484010815620422,
                    -0.7058414220809937,
                    0.6432183980941772,
                    0.48471593856811523,
                    0.6921041011810303,
                    -1.3928877115249634,
                ],
            ]
        ),
    )

    # V to U
    output_nodes_2 = conv.propagate(
        data["U", "to", "V"].edge_index,
        x=x_uv,
        edge_attr=data["U", "to", "V"].edge_attr,
    )
    output_edges_2 = conv.edge_updater(
        data["U", "to", "V"].edge_index,
        x=x_uv,
        edge_attr=data["U", "to", "V"].edge_attr,
    )

    assert torch.allclose(
        output_nodes_2,
        torch.tensor(
            [
                [
                    -0.6484010815620422,
                    -0.7058414220809937,
                    0.6432183980941772,
                    -0.6484010815620422,
                    -0.7058414220809937,
                    0.6432183980941772,
                    0.3082679510116577,
                    -0.2946690022945404,
                    -0.7662270069122314,
                    -0.9961578845977783,
                    0.3082679510116577,
                    -0.2946690022945404,
                    -0.7662270069122314,
                    -0.9961578845977783,
                ],
                [
                    -0.6484010815620422,
                    -0.7058414220809937,
                    0.6432183980941772,
                    -0.6484010815620422,
                    -0.7058414220809937,
                    0.6432183980941772,
                    -0.23451003432273865,
                    -0.5367372632026672,
                    1.1295514106750488,
                    0.10535295307636261,
                    -0.23451003432273865,
                    -0.5367372632026672,
                    1.1295514106750488,
                    0.10535295307636261,
                ],
            ]
        ),
    )
    assert torch.allclose(
        output_edges_2,
        torch.tensor(
            [
                [
                    1.4787991046905518,
                    1.1918498277664185,
                    -0.14461655914783478,
                    -0.6484010815620422,
                    -0.7058414220809937,
                    0.6432183980941772,
                ],
                [
                    0.48471593856811523,
                    0.6921041011810303,
                    -1.3928877115249634,
                    -0.6484010815620422,
                    -0.7058414220809937,
                    0.6432183980941772,
                ],
            ]
        ),
    )


def test_BeanAggregation_forward():
    x = torch.randn((4, 16))
    index = torch.tensor([0, 0, 1, 1])
    ptr = torch.tensor([0, 2, 4])
    edge_attr = torch.randn((4, 8))
    aggregation = BeanAggregation()
    output = aggregation.forward(x, index, ptr, edge_attr=edge_attr)
    assert output.shape == (2, 48)


def test_BEANConv_forward_self_loop():
    in_channels = 3
    out_channels = 2
    data = create_hetero_data()
    conv = BEANConv(
        in_channels,
        out_channels,
        edge_in_channels=6,
        node_self_loop=True,
        normalize=False,
    )

    x_uv = (data.x_dict["U"], data.x_dict["V"])
    x_vu = (data.x_dict["V"], data.x_dict["U"])

    # V to U
    output_nodes, output_edges = conv.forward(
        x_vu, data["V", "to", "U"].edge_index, edge_attr=data["V", "to", "U"].edge_attr
    )
    assert torch.allclose(
        output_nodes, torch.tensor([[-0.8591989278793335, -0.636704683303833]])
    )
    assert torch.allclose(
        output_edges,
        torch.tensor(
            [
                [
                    -0.6484010815620422,
                    -0.7058414220809937,
                    0.6432183980941772,
                    1.4787991046905518,
                    1.1918498277664185,
                    -0.14461655914783478,
                ],
                [
                    -0.6484010815620422,
                    -0.7058414220809937,
                    0.6432183980941772,
                    0.48471593856811523,
                    0.6921041011810303,
                    -1.3928877115249634,
                ],
            ]
        ),
    )

    # V to U
    output_nodes_2, output_edges_2 = conv.forward(
        x_uv, data["U", "to", "V"].edge_index, edge_attr=data["U", "to", "V"].edge_attr
    )
    assert torch.allclose(
        output_nodes_2,
        torch.tensor(
            [
                [0.9347802996635437, 0.8489263653755188],
                [0.01791500672698021, 0.5591261982917786],
            ]
        ),
    )
    assert torch.allclose(
        output_edges_2,
        torch.tensor(
            [
                [
                    1.4787991046905518,
                    1.1918498277664185,
                    -0.14461655914783478,
                    -0.6484010815620422,
                    -0.7058414220809937,
                    0.6432183980941772,
                ],
                [
                    0.48471593856811523,
                    0.6921041011810303,
                    -1.3928877115249634,
                    -0.6484010815620422,
                    -0.7058414220809937,
                    0.6432183980941772,
                ],
            ]
        ),
    )


def test_BEANConv_forward_no_bias():
    in_channels = 3
    out_channels = 2
    data = create_hetero_data()
    conv = BEANConv(
        in_channels,
        out_channels,
        edge_in_channels=6,
        node_self_loop=False,
        normalize=False,
        bias=False,
    )

    x_uv = (data.x_dict["U"], data.x_dict["V"])
    x_vu = (data.x_dict["V"], data.x_dict["U"])

    # V to U
    output_nodes, output_edges = conv.forward(
        x_vu, data["V", "to", "U"].edge_index, edge_attr=data["V", "to", "U"].edge_attr
    )
    assert torch.allclose(
        output_nodes, torch.tensor([[-0.3630157709121704, 1.3030002117156982]])
    )
    assert torch.allclose(
        output_edges,
        torch.tensor(
            [
                [
                    -0.6484010815620422,
                    -0.7058414220809937,
                    0.6432183980941772,
                    1.4787991046905518,
                    1.1918498277664185,
                    -0.14461655914783478,
                ],
                [
                    -0.6484010815620422,
                    -0.7058414220809937,
                    0.6432183980941772,
                    0.48471593856811523,
                    0.6921041011810303,
                    -1.3928877115249634,
                ],
            ]
        ),
    )

    # V to U
    output_nodes_2, output_edges_2 = conv.forward(
        x_uv, data["U", "to", "V"].edge_index, edge_attr=data["U", "to", "V"].edge_attr
    )
    assert torch.allclose(
        output_nodes_2,
        torch.tensor(
            [
                [0.3290987014770508, -0.18378779292106628],
                [0.43931475281715393, -0.92584228515625],
            ]
        ),
    )
    assert torch.allclose(
        output_edges_2,
        torch.tensor(
            [
                [
                    1.4787991046905518,
                    1.1918498277664185,
                    -0.14461655914783478,
                    -0.6484010815620422,
                    -0.7058414220809937,
                    0.6432183980941772,
                ],
                [
                    0.48471593856811523,
                    0.6921041011810303,
                    -1.3928877115249634,
                    -0.6484010815620422,
                    -0.7058414220809937,
                    0.6432183980941772,
                ],
            ]
        ),
    )


def test_BEANConv_forward_normalize():
    in_channels = 3
    out_channels = 2
    data = create_hetero_data()
    conv = BEANConv(
        in_channels,
        out_channels,
        edge_in_channels=6,
        node_self_loop=False,
        normalize=True,
        bias=False,
    )

    x_uv = (data.x_dict["U"], data.x_dict["V"])

    # V to U
    output_nodes_2, output_edges_2 = conv.forward(
        x_uv, data["U", "to", "V"].edge_index, edge_attr=data["U", "to", "V"].edge_attr
    )
    assert torch.allclose(
        output_nodes_2,
        torch.tensor(
            [
                [-0.9983575344085693, 0.9999637007713318],
                [0.9983577728271484, -0.9999637007713318],
            ]
        ),
    )
    assert torch.allclose(
        output_edges_2,
        torch.tensor(
            [
                [
                    0.9999797344207764,
                    0.9999198317527771,
                    0.9999871253967285,
                    2.572351149865426e-06,
                    3.123230271739885e-06,
                    5.0973503675777465e-06,
                ],
                [
                    -0.9999798536300659,
                    -0.9999200105667114,
                    -0.9999871253967285,
                    2.572351149865426e-06,
                    3.123230271739885e-06,
                    5.0973503675777465e-06,
                ],
            ]
        ),
    )


def test_BEANConv_forward_edge_projection():
    in_channels = 3
    out_channels = 2
    data = create_hetero_data()
    conv = BEANConv(
        in_channels,
        out_channels,
        edge_in_channels=6,
        node_self_loop=False,
        normalize=True,
        bias=False,
        edge_out_channels=2,
    )

    x_uv = (data.x_dict["U"], data.x_dict["V"])

    # V to U
    output_nodes_2, output_edges_2 = conv.forward(
        x_uv, data["U", "to", "V"].edge_index, edge_attr=data["U", "to", "V"].edge_attr
    )
    assert torch.allclose(
        output_nodes_2,
        torch.tensor(
            [
                [-0.9983575344085693, 0.9999637007713318],
                [0.9983577728271484, -0.9999637007713318],
            ]
        ),
    )
    assert torch.allclose(
        output_edges_2,
        torch.tensor(
            [
                [0.9999125003814697, 0.9999629855155945],
                [-0.9999125003814697, -0.9999629855155945],
            ]
        ),
    )


def test_BEANConv_no_edge_attributes():
    in_channels = 3
    out_channels = 2
    data = create_hetero_data_no_edge_attr()
    conv = BEANConvSimple(
        in_channels,
        out_channels,
        node_self_loop=False,
        normalize=False,
        bias=False,
    )

    x_uv = (data.x_dict["U"], data.x_dict["V"])

    # V to U
    output_nodes_2 = conv.forward(x_uv, data["U", "to", "V"].edge_index)
    print(f"output_nodes_2:{output_nodes_2.tolist()}")
    assert torch.allclose(
        output_nodes_2,
        torch.tensor(
            [
                [0.16857275366783142, 0.18551811575889587],
                [0.16857275366783142, 0.18551811575889587],
            ]
        ),
    )
