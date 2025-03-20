import torch
from fintorch.models.timeseries.tft.tft_module import TemporalFusionTransformerModule

def test_forward_pass():
    # Define model parameters
    number_of_past_inputs = 4
    number_of_future_inputs = 2
    embedding_size_inputs = 8
    hidden_dimension = 16
    dropout = 0.1
    number_of_heads = 2
    past_inputs = {"feature1": 4, "feature2": 3}
    future_inputs = {"feature1": 4}
    static_inputs = {"feature3": 4, "feature4": 3}
    batch_size = 8
    device = "cpu"



    # Create model instance
    model = TemporalFusionTransformerModule(
        number_of_past_inputs,
        number_of_future_inputs,
        embedding_size_inputs,
        hidden_dimension,
        dropout,
        number_of_heads,
        past_inputs,
        future_inputs,
        static_inputs,
        batch_size,
        device,
    )

    # Create dummy inputs
    past_inputs_dict = {
        "feature1": torch.randn(
            batch_size, number_of_past_inputs, past_inputs["feature1"]
        ),
        "feature2": torch.randn(
            batch_size, number_of_past_inputs, past_inputs["feature2"]
        ),
    }
    future_inputs_dict = {
        "feature1": torch.randn(
            batch_size, number_of_future_inputs, future_inputs["feature1"]
        ),
    }
    static_data = {
        "feature3": torch.randn(
            batch_size, static_inputs["feature3"]
        ),
        "feature4": torch.randn(
            batch_size, static_inputs["feature4"]
        ),
    }

    # Forward pass
    output = model(past_inputs_dict, future_inputs_dict, static_data)

    # Check output shape
    assert output is not None, "Output is None"
    assert isinstance(output, tuple), "Output is not a tuple"
    assert len(output) == 2, "Output tuple does not contain two elements"
    assert output[0].shape[0] == batch_size, "Output batch size mismatch"


# def test_unpack_batch_format_1():
#     # Create model instance
#     model = TemporalFusionTransformerModule(
#         4, 2, 8, 16, 0.1, 2, {}, {}, {}, 8, "cpu"
#     )

#     # Create dummy batch in format 1
#     past_inputs_dict = {
#         "feature1": torch.randn(
#             batch_size, number_of_past_inputs, past_inputs["feature1"]
#         ),
#         "feature2": torch.randn(
#             batch_size, number_of_past_inputs, past_inputs["feature2"]
#         ),
#     }
#     future_inputs_dict = {
#         "feature1": torch.randn(
#             batch_size, number_of_future_inputs, future_inputs["feature1"]
#         ),
#     }
#     static_data = {
#         "feature3": torch.randn(
#             batch_size, static_inputs["feature3"]
#         ),
#         "feature4": torch.randn(
#             batch_size, static_inputs["feature4"]
#         ),
#     }
#     target = torch.randn(8, 1)
#     batch = (past_inputs_dict, future_inputs_dict, static_data, target)

#     # Unpack batch
#     unpacked = model._unpack_batch(batch)

#     # Check unpacked values
#     assert len(unpacked) == 4, "Unpacked batch does not contain 4 elements"
#     assert torch.equal(unpacked[0], past_inputs), "Past inputs mismatch"
#     assert torch.equal(unpacked[1], future_inputs), "Future inputs mismatch"
#     assert torch.equal(unpacked[2], static_inputs), "Static inputs mismatch"
#     assert torch.equal(unpacked[3], target), "Target mismatch"


# def test_unpack_batch_format_2():
#     # Create model instance
#     model = TemporalFusionTransformerModule(
#         4, 2, 8, 16, 0.1, 2, [], [], [], 8, "cpu"
#     )

#     # Create dummy batch in format 2
#     inputs = (torch.randn(8, 4, 8), torch.randn(8, 2, 8), torch.randn(8, 1, 8))
#     target = torch.randn(8, 1)
#     batch = (inputs, target)

#     # Unpack batch
#     unpacked = model._unpack_batch(batch)

#     # Check unpacked values
#     assert len(unpacked) == 4, "Unpacked batch does not contain 4 elements"
#     assert torch.equal(unpacked[0], inputs[0]), "Past inputs mismatch"
#     assert torch.equal(unpacked[1], inputs[1]), "Future inputs mismatch"
#     assert torch.equal(unpacked[2], inputs[2]), "Static inputs mismatch"
#     assert torch.equal(unpacked[3], target), "Target mismatch"


def test_training_step():
# Define model parameters
    number_of_past_inputs = 4
    number_of_future_inputs = 2
    embedding_size_inputs = 8
    hidden_dimension = 16
    dropout = 0.1
    number_of_heads = 2
    past_inputs = {"feature1": 4, "feature2": 3}
    future_inputs = {"feature1": 4}
    static_inputs = {"feature3": 4, "feature4": 3}
    batch_size = 8
    device = "cpu"



    # Create model instance
    model = TemporalFusionTransformerModule(
        number_of_past_inputs,
        number_of_future_inputs,
        embedding_size_inputs,
        hidden_dimension,
        dropout,
        number_of_heads,
        past_inputs,
        future_inputs,
        static_inputs,
        batch_size,
        device,
    )

    # Create dummy inputs
    past_inputs_dict = {
        "feature1": torch.randn(
            batch_size, number_of_past_inputs, past_inputs["feature1"]
        ),
        "feature2": torch.randn(
            batch_size, number_of_past_inputs, past_inputs["feature2"]
        ),
    }
    future_inputs_dict = {
        "feature1": torch.randn(
            batch_size, number_of_future_inputs, future_inputs["feature1"]
        ),
    }
    static_data = {
        "feature3": torch.randn(
            batch_size, static_inputs["feature3"]
        ),
        "feature4": torch.randn(
            batch_size, static_inputs["feature4"]
        ),
    }

    # Generate target tensor
    target = torch.randn(batch_size, number_of_future_inputs, 1)

    # Forward pass
    batch = (past_inputs_dict, future_inputs_dict, static_data, target)

    # Perform training step
    loss = model.training_step(batch, 0)

    # Check loss
    assert loss is not None, "Loss is None"
    assert isinstance(loss, torch.Tensor), "Loss is not a tensor"
    assert loss.item() > 0, "Loss is not positive"
