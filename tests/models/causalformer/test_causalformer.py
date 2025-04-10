import torch
from fintorch.models.timeseries.causalformer.CausalFormer import CausalFormer
from fintorch.models.timeseries.causalformer.Encoder import Encoder


def test_causalformer_initialization():
    # Define parameters
    number_of_layers = 2
    number_of_heads = 4
    number_of_series = 5
    length_input_window = 10
    length_output_window = 2
    embedding_size = 16
    feature_dimensionality = 8
    ffn_hidden_dimensionality = 32
    output_dimensionality = 4
    tau = 0.1
    dropout = 0.1

    # Initialize the CausalFormer module
    model = CausalFormer(
        number_of_layers=number_of_layers,
        number_of_heads=number_of_heads,
        number_of_series=number_of_series,
        length_input_window=length_input_window,
        length_output_window=length_output_window,
        embedding_size=embedding_size,
        feature_dimensionality=feature_dimensionality,
        ffn_hidden_dimensionality=ffn_hidden_dimensionality,
        output_dimensionality=output_dimensionality,
        tau=tau,
        dropout=dropout,
    )

    # Assertions
    assert isinstance(
        model.encoder, Encoder
    ), "Encoder should be an instance of Encoder"
    assert model.fully_connected.in_features == feature_dimensionality, (
        f"Expected fully_connected in_features to be {feature_dimensionality}, "
        f"but got {model.fully_connected.in_features}"
    )
    assert model.fully_connected.out_features == output_dimensionality, (
        f"Expected fully_connected out_features to be {output_dimensionality}, "
        f"but got {model.fully_connected.out_features}"
    )
    assert model.output_window == length_output_window, (
        f"Expected output_window to be {length_output_window}, "
        f"but got {model.output_window}"
    )


def test_causalformer_forward_pass():
    # Define parameters
    number_of_layers = 2
    number_of_heads = 4
    number_of_series = 5
    length_input_window = 10
    length_output_window = 2
    embedding_size = 16
    feature_dimensionality = 8
    ffn_hidden_dimensionality = 32
    output_dimensionality = 4
    tau = 0.1
    dropout = 0.1

    # Initialize the CausalFormer module
    model = CausalFormer(
        number_of_layers=number_of_layers,
        number_of_heads=number_of_heads,
        number_of_series=number_of_series,
        length_input_window=length_input_window,
        length_output_window=length_output_window,
        embedding_size=embedding_size,
        feature_dimensionality=feature_dimensionality,
        ffn_hidden_dimensionality=ffn_hidden_dimensionality,
        output_dimensionality=output_dimensionality,
        tau=tau,
        dropout=dropout,
    )

    # Generate fake input tensor
    batch_size = 8
    fake_input = torch.randn(
        batch_size, number_of_series, length_input_window, feature_dimensionality
    )

    # Perform a forward pass
    output = model(fake_input)

    # Assertions
    assert output.shape == (
        batch_size,
        number_of_series,
        length_output_window,
        output_dimensionality,
    ), (
        f"Expected output shape to be {(batch_size, number_of_series, length_output_window, output_dimensionality)}, "
        f"but got {output.shape}"
    )
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"
