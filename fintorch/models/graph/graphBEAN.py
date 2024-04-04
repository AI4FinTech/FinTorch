import torch.nn as nn
import torch_geometric.nn as nng


def GraphBEANLoss(predictions, targets):

    total_loss = 0

    # Loop over all node types and add the MSE loss
    for key in predictions.keys():
        total_loss += nn.MSELoss()(predictions[key], targets[key].x)

    # encoder_mse = nn.MSELoss()(predictions[:, 0], targets[:, 0])
    # decoder_mse = nn.MSELoss()(predictions[:, 1], targets[:, 1])
    # structure_mse = nn.MSELoss()(predictions[:, 2], targets[:, 2])

    # return encoder_mse + decoder_mse + structure_mse

    return total_loss


class GraphBEAN(nn.Module):

    def __init__(
        self,
        hetero_data,
        n_encoder_layers: int,
        n_feature_decoder_layers: int,
        n_structure_decoder_layers: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.encoder_layers = nn.ModuleList()

        # Encoder decoder (latent)
        self.encoder_layers.append(
            self.EncoderLayer(hetero_data, in_channels, hidden_channels))
        for _ in range(n_encoder_layers - 1):
            self.encoder_layers.append(
                self.EncoderLayer(hetero_data, in_channels, hidden_channels))

        # FeatureDecoder
        self.decoder_layers = nn.ModuleList()
        for _ in range(n_feature_decoder_layers):
            self.decoder_layers.append(
                self.FeatureDecoderLayer(hetero_data, hidden_channels,
                                         out_channels))

        self.structure_decoder_layers = nn.ModuleList()

        # add the decoder net for the structure
        for _ in range(n_structure_decoder_layers):
            self.structure_decoder_layers.append(
                self.StructureDecoder(hetero_data, hidden_channels,
                                      out_channels))

    def StructureDecoder(self, hetero_data, in_channels, out_channels):
        struct_layer = nng.HeteroLinear(
            in_channels=in_channels,
            out_channels=out_channels,
            num_types=len(hetero_data.node_types),
        )
        return struct_layer

    def FeatureDecoderLayer(self, hetero_data, in_channels, out_channels):
        conv_layer = nng.HeteroConv(
            {
                edge_type: nng.SAGEConv(in_channels, out_channels)
                for edge_type in hetero_data.edge_types
            },
            aggr="sum",
        )

        # return nn.Sequential(conv_layer, batch_layer, nn.ReLU())
        return conv_layer

    def EncoderLayer(self, hetero_data, in_channels, out_channels):
        conv_layer = nng.HeteroConv(
            {
                edge_type: nng.SAGEConv(in_channels, out_channels)
                for edge_type in hetero_data.edge_types
            },
            aggr="sum",
        )

        return conv_layer

    def forward(self, data):
        # loop through the encoder layers to obtain the hidden representation
        x_dict = data.x_dict
        for layer in self.encoder_layers:
            x_dict = layer(
                x_dict,
                data.edge_index_dict,
            )

        # Obtain the feature decoding output after looping through the feature decoding layers
        feature_out = x_dict
        for layer in self.decoder_layers:
            feature_out = layer(
                feature_out,
                data.edge_index_dict,
            )

        # TODO: Figure out how to do the forward pass through the hetero linear layer
        # Obtain the output after loping through the structure prediction network
        # structure_output = x_dict
        # for layer in self.structure_decoder_layers:
        #     structure_output = layer(structure_output, data.edge_index_dict)

        # print(
        #     f"Hidden representation:{x_dict['paper'].shape} \n\n feature_out:{feature_out['paper'].shape} "
        # )

        # # we return the predictions: feature output, and structure output
        # return feature_output, structure_output
        return feature_out
