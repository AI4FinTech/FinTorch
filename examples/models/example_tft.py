import os

import lightning as L
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import EarlyStopping

from fintorch.datasets.synthetic import SimpleSyntheticDataModule
from fintorch.models.timeseries.tft import (
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
    TemporalFusionTransformer,
    TemporalFusionTransformerModule,
    VariableSelectionNetwork,
)

# Define hyperparameters
input_dimensions = 10
hidden_dimensions = 64
dropout = 0.1
context_size = 5
batch_size = 32
sequence_length = 10
num_inputs = 3
number_of_heads = 4

# Example usage of GatedResidualNetwork
context = torch.randn(batch_size, context_size)
grn = GatedResidualNetwork(
    input_size=input_dimensions,
    hidden_size=hidden_dimensions,
    output_size=hidden_dimensions,
    dropout=dropout,
    context_size=context_size,
)
grn_input = torch.randn(batch_size, sequence_length, input_dimensions)
grn_output = grn(grn_input, context)
print("GRN Output shape:", grn_output.shape)


# Example usage of VariableSelectionNetwork
variable_selection_network = VariableSelectionNetwork(
    {"a": 3, "b": 4, "c": 5}, hidden_dimensions, dropout, context_size
)

# Create example input data
# Example input tensor with 3 inputs
x_a = torch.randn(batch_size, sequence_length, 3)
x_b = torch.randn(batch_size, sequence_length, 4)
x_c = torch.randn(batch_size, sequence_length, 5)
inputs = {"a": x_a, "b": x_b, "c": x_c}

print("Shape of x_a:", x_a.shape)
print("Shape of x_b:", x_b.shape)
print("Shape of x_c:", x_c.shape)


# Call the forward method
vsn_output = variable_selection_network(inputs, context)

# Print the output shape
print("VSN output shape:", vsn_output.shape)

# Example usage of InterpretableMultiHeadAttention
# Create an instance of InterpretableMultiHeadAttention
attention_module = InterpretableMultiHeadAttention(
    number_of_heads, hidden_dimensions, dropout
)

# Generate example input tensors
q = grn_output  # Use output of GRN as query
k = grn_output  # Use output of GRN as key
v = grn_output  # Use output of GRN as value

# Create a mask (optional)
# This is an example mask to prevent attending to future time steps
mask = torch.tril(torch.ones(sequence_length, sequence_length)).bool()

# Call the forward method of InterpretableMultiHeadAttention
attention_output, attentions = attention_module(q, k, v, mask)

# Print the output shapes
print("Attention Output shape:", attention_output.shape)
print("Attentions shape:", attentions.shape)

# Example of using the InterpretableMultiHeadAttention after the variable selection network
q = vsn_output  # Using the output of the vsn.
k = vsn_output  # Using the output of the vsn.
v = vsn_output  # Using the output of the vsn.

attention_output, attentions = attention_module(q, k, v, mask)

# Print the output shapes
print("Attention Output shape after the vsn:", attention_output.shape)
print("Attentions shape after the vsn:", attentions.shape)


# Example of plotting the attention maps per head
# Assuming 'attentions' is a tensor of shape (batch_size, seq_length, num_heads, seq_length)
# Let's visualize the attention maps for the first element in the batch

batch_index = 0
fig, axes = plt.subplots(
    1, number_of_heads, figsize=(5 * number_of_heads, 5), constrained_layout=True
)

for head in range(number_of_heads):
    attention_map = attentions[batch_index, :, head, :].detach().cpu().numpy()
    ax = axes[head] if number_of_heads > 1 else axes

    # Plot the attention map as a heatmap
    im = ax.imshow(attention_map, cmap="viridis")

    # Set labels and title
    ax.set_xlabel("Key Sequence Position")
    if head == 0:
        ax.set_ylabel("Query Sequence Position")
    ax.set_title(f"Attention Map - Head {head+1}")

fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="Attention Weight")
plt.show()


print("####### TFT ########")

# Example usage of TemporalFusionTransformer
# Define hyperparameters
embedding_size_inputs = 64
hidden_dimension = 64
dropout = 0.1
number_of_heads = 4
past_inputs = {"a": 3, "b": 4, "c": 5}
future_inputs = {"d": 2, "e": 3}
static_inputs = {"f": 4, "g": 5}
quantiles = [0.05, 0.5, 0.95]
sequence_length_past = 6

sequence_length_future = 2

# Create an instance of TemporalFusionTransformer
tft_model = TemporalFusionTransformer(
    sequence_length_past,
    sequence_length_future,
    embedding_size_inputs,
    hidden_dimension,
    dropout,
    number_of_heads,
    past_inputs,
    future_inputs,
    static_inputs,
    quantiles=quantiles,
    batch_size=batch_size,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

# Generate example input tensors
past_inputs_tensor = {
    "a": torch.randn(batch_size, sequence_length_past, 3),
    "b": torch.randn(batch_size, sequence_length_past, 4),
    "c": torch.randn(batch_size, sequence_length_past, 5),
}
future_inputs_tensor = {
    "d": torch.randn(batch_size, sequence_length_future, 2),
    "e": torch.randn(batch_size, sequence_length_future, 3),
}
static_inputs_tensor = {
    "f": torch.randn(batch_size, 4),
    "g": torch.randn(batch_size, 5),
}

# Call the forward method of TemporalFusionTransformer
tft_output, attention_weights = tft_model(
    past_inputs_tensor, future_inputs_tensor, static_inputs_tensor
)

# Print the output shapes
print(
    "TFT Output shape[batch size, horizon, number of targets=1, number of quantiles]:",
    tft_output.shape,
)
print(f"past lenght:{sequence_length} future length:{sequence_length_future}")
print(
    "Attention Weights shape[batch size, heads, sequence length (past length + future length),"
    + " sequence length (past length + future length)]:",
    attention_weights.shape,
)


print("#################### TFT MODULE ###################")

# Example usage of TemporalFusionTransformerModule
# Define hyperparameters
sequence_length_past = 3
number_of_future_inputs = 2
embedding_size_inputs = 64
hidden_dimension = 64
dropout = 0.1
number_of_heads = 4
past_inputs = {"a": 3, "b": 4, "c": 5}
future_inputs = {"d": 2, "e": 3}
static_inputs = {"f": 4, "g": 5}

sequence_length_future = 2

# Create an instance of TemporalFusionTransformerModule
tft_module = TemporalFusionTransformerModule(
    sequence_length_past,
    sequence_length_future,
    embedding_size_inputs,
    hidden_dimension,
    dropout,
    number_of_heads,
    past_inputs,
    future_inputs,
    static_inputs,
    quantiles=quantiles,
    batch_size=batch_size,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

# Generate example input tensors
past_inputs_tensor = {
    "a": torch.randn(batch_size, sequence_length_past, 3),
    "b": torch.randn(batch_size, sequence_length_past, 4),
    "c": torch.randn(batch_size, sequence_length_past, 5),
}
future_inputs_tensor = {
    "d": torch.randn(batch_size, sequence_length_future, 2),
    "e": torch.randn(batch_size, sequence_length_future, 3),
}
static_inputs_tensor = {
    "f": torch.randn(batch_size, 4),
    "g": torch.randn(batch_size, 5),
}
target = torch.randn(batch_size, sequence_length_future).float()

# Create a batch
batch = (past_inputs_tensor, future_inputs_tensor, static_inputs_tensor, target)

# Call the forward method of TemporalFusionTransformerModule
tft_module_output, attention_weights = tft_module(
    past_inputs_tensor, future_inputs_tensor, static_inputs_tensor
)

# Print the output shapes
print("TFT Module Output shape:", tft_module_output.shape)
print("Attention Weights shape:", attention_weights.shape)

# Call the training step
loss = tft_module.training_step(batch, 0)
print("Loss:", loss)

# Call the validation step
loss = tft_module.validation_step(batch, 0)
print("Loss:", loss)

# Call the test step
loss = tft_module.test_step(batch, 0)
print("Loss:", loss)

# Call the predict step
output = tft_module.predict_step(batch, 0)
print("Prediction:", output.shape)


# print("#################### TFT MODULE WITH DATASET ###################")


#
static_length = 2
past_inputs = 1
future_inputs = 1
noise_level = 5
trend_slope = 0.1
seasonality_amplitude = 10
seasonality_period = 100
batch_size = 32

# Define hyperparameters
number_of_past_inputs = 24
number_of_future_inputs = 12
embedding_size_inputs = hidden_dimension = 32
dropout = 0.5
number_of_heads = 1
past_inputs = {"past_data": past_inputs}
future_inputs = {"future_data": future_inputs}
static_inputs = {"static_data": static_length}

# Create an instance of TemporalFusionTransformerModule
tft_module = TemporalFusionTransformerModule(
    number_of_past_inputs,
    number_of_future_inputs,
    embedding_size_inputs,
    hidden_dimension,
    dropout,
    number_of_heads,
    past_inputs,
    future_inputs,
    static_inputs,
    batch_size=batch_size,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

# Create an instance of SimpleSyntheticDataModule
data_module = SimpleSyntheticDataModule(
    train_length=1000,
    val_length=100,
    test_length=100,
    batch_size=batch_size,
    noise_level=noise_level,
    past_length=number_of_past_inputs,
    future_length=number_of_future_inputs,
    static_length=static_length,
    trend_slope=trend_slope,
    seasonality_amplitude=seasonality_amplitude,
    seasonality_period=seasonality_period,
    workers=os.cpu_count(),
)

# Set the precision
torch.set_float32_matmul_precision("medium")

# Prepare the data
data_module.setup()


plot_all_data = data_module.train_dataset.data

# Plot all data
plt.figure(figsize=(15, 5))
plt.plot(plot_all_data, label="All Data")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.title("All Data")
plt.legend()
plt.show()


train_dataloader = data_module.train_dataloader()


# Create a trainer with TensorBoard for better monitoring
early_stopping = EarlyStopping("val_loss_epoch", patience=50)
trainer = L.Trainer(max_epochs=50, callbacks=[early_stopping])

# Train the model
trainer.fit(tft_module, data_module)
trainer.test(tft_module, data_module)
