import matplotlib.pyplot as plt
import torch

from fintorch.models.timeseries.tft import (
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
    TemporalFusionTransformer,
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
    ax.set_title(f"Attention Map - Head {head + 1}")

fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="Attention Weight")
plt.savefig("attention_maps.png")
plt.close()


print("####### TFT ########")

# Example usage of TemporalFusionTransformer
# Define hyperparameters
embedding_size_inputs = 64
hidden_dimension = 64
dropout = 0.1
number_of_heads = 4
# Use standardized key names
past_inputs = {"past_target": 1, "past_covariates_known_future": 2, "past_covariates_unknown_future": 2}
future_inputs = {"future_covariates_known": 5}
static_inputs = {"static_features_real": 4, "static_features_categorical": 5}
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
    "past_target": torch.randn(batch_size, sequence_length_past, 1),
    "past_covariates_known_future": torch.randn(batch_size, sequence_length_past, 2),
    "past_covariates_unknown_future": torch.randn(batch_size, sequence_length_past, 2),
}
future_inputs_tensor = {
    "future_covariates_known": torch.randn(batch_size, sequence_length_future, 5),
}
static_inputs_tensor = {
    "static_features_real": torch.randn(batch_size, 4),
    "static_features_categorical": torch.randn(batch_size, 5),
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
