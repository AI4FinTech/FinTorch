from typing import Dict

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.nn as nng
import torchmetrics
from torch_geometric.nn import SAGEConv

VERBOSE = False


def GraphBEANLoss(feature_predictions, edge_predictions,
                  ground_truth_sampled_data, edge):
    """
    Calculates the loss function for the GraphBEAN model.

    Args:
        feature_predictions (dict): A dictionary containing the predicted features for each node type.
        edge_predictions (torch.Tensor): The predicted edge values.
        ground_truth_sampled_data (dict): A dictionary containing the ground truth sampled data for each node type.
        edge (tuple): A tuple representing the source, target, and destination nodes for the edge.

    Returns:
        torch.Tensor: The total loss value.

    """
    # Loss of the feature reconstruction per node type
    feature_loss = 0
    for key in feature_predictions.keys():
        feature_loss += nn.MSELoss()(feature_predictions[key],
                                     ground_truth_sampled_data[key].x)

    # Edge prediction loss (one edge type)
    src, to, dst = edge
    edge_loss = F.binary_cross_entropy_with_logits(
        edge_predictions, ground_truth_sampled_data[src, to, dst].edge_label)
    # Total loss function
    total_loss = feature_loss + edge_loss
    return total_loss


def GraphBEANLossClassifier(
    feature_predictions,
    edge_predictions,
    ground_truth_sampled_data,
    edge,
    node_pred,
    node_ground_truth,
):
    """
    Calculates the total loss for the GraphBEAN model with a classifier.

    Args:
        feature_predictions (Tensor): Predictions of the node features.
        edge_predictions (Tensor): Predictions of the edge features.
        ground_truth_sampled_data (Tensor): Ground truth sampled data.
        edge (Tensor): Edge tensor.
        node_pred (Tensor): Predictions of the node labels.
        node_ground_truth (Tensor): Ground truth node labels.

    Returns:
        Tensor: The total loss for the GraphBEAN model with a classifier.
    """

    # Loss of the feature reconstruction per node type
    loss = GraphBEANLoss(feature_predictions, edge_predictions,
                         ground_truth_sampled_data, edge)

    classification_loss_fn = torch.nn.CrossEntropyLoss()

    # TODO: fix class loss
    # Filter for illict and licit class
    idx = node_ground_truth != 3
    filtered = node_ground_truth[idx]

    if VERBOSE:
        print(f"node_pre:{node_pred} gt:{filtered} unique:{filtered.unique()}")

    class_loss = classification_loss_fn(node_pred[idx, :], filtered)

    # Total loss function
    # total_loss = loss + class_loss * 100

    weight = 3  # weight > 1, then class loss is weighted more
    # Harmonic mean of feature_loss and total_loss
    harmonic_mean = (1 + weight) * (loss * class_loss) / (weight * loss +
                                                          class_loss)
    total_loss = harmonic_mean

    return total_loss


class StructureDecoder(nn.Module):
    """
    StructureDecoder is a PyTorch module that performs link classification in a graph.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        assert num_layers >= 2

        self.mlp_layers_src = nn.ModuleList()
        self.mlp_layers_src.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.mlp_layers_src.append(
                nn.Linear(hidden_channels, hidden_channels))
        self.mlp_layers_src.append(nn.Linear(hidden_channels, out_channels))

        self.mlp_layers_dst = nn.ModuleList()
        self.mlp_layers_dst.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.mlp_layers_dst.append(
                nn.Linear(hidden_channels, hidden_channels))
        self.mlp_layers_dst.append(nn.Linear(hidden_channels, out_channels))

    def forward(self, src, dst, edge_label_index):
        """
        Performs the forward pass of the LinkClassifier model.

        Args:
            src (torch.Tensor): Source node features.
            dst (torch.Tensor): Destination node features.
            edge_label_index (tuple): Tuple containing the indices of the edge labels.

        Returns:
            torch.Tensor: The result of the forward pass.
        """
        x_src_features = src[edge_label_index[0]]
        x_dst_features = dst[edge_label_index[1]]

        # Loop over the self.mlp_layers_src
        src = x_src_features
        for layer in self.mlp_layers_src:
            src = layer(src)

        # Loop over the self.mlp_layers_dst
        dst = x_dst_features
        for layer in self.mlp_layers_dst:
            dst = layer(dst)

        return torch.sigmoid((src * dst).sum(dim=-1))


class GraphBEAN(nn.Module):
    """
    GraphBEAN (Graph Bipartite Edge and Node) model for interaction-focused anomaly detection on bipartite
    node-and-edge-attributed graphs. We modified the original implementation to include a classification head such
    that the model works for semi-supervised settings.

    Reference:
    Fathony, R., Ng, J., & Chen, J. (2023, June). Interaction-focused anomaly detection on bipartite
    node-and-edge-attributed graphs. In 2023 International Joint Conference on Neural Networks (IJCNN) (pp. 1-10). IEEE.

    Args:
        hetero_data (HeteroData): The heterogeneous graph data.
        n_encoder_layers (int): Number of encoder layers.
        n_feature_decoder_layers (int): Number of feature decoder layers.
        hidden_channels (int): Number of hidden channels.
        features_channels (dict): Dictionary specifying the number of channels for each edge type.

    Attributes:
        encoder_layers (nn.ModuleList): List of encoder layers.
        decoder_layers (nn.ModuleList): List of feature decoder layers.
        graph_edge_prediction (LinkClassifier): Link classifier for graph edge prediction.
    """

    def __init__(
        self,
        heterodata,
        n_encoder_layers: int,
        n_feature_decoder_layers: int,
        hidden_channels: int,
        features_channels: dict,
        conv_type: callable,
        edge_types,
    ):
        super().__init__()

        assert n_encoder_layers > 0, "Number of encoder layers must be greater than 0."
        assert (
            n_feature_decoder_layers
            > 0), "Number of feature decoder layers must be greater than 0."
        assert hidden_channels > 0, "Number of hidden channels must be greater than 0."

        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        first_encoder_layer = nng.HeteroConv(
            {
                edge_type: conv_type(-1, hidden_channels)
                for edge_type in edge_types
            },
            aggr="sum",
        )

        self.encoder_layers.append(first_encoder_layer)

        for _ in range(n_encoder_layers - 1):
            self.encoder_layers.append(
                nng.HeteroConv(
                    {
                        edge_type: conv_type(hidden_channels, hidden_channels)
                        for edge_type in edge_types
                    },
                    aggr="sum",
                ))  # Pass conv_type argument

        # FeatureDecoder layers
        self.decoder_layers = nn.ModuleList()
        for _ in range(n_feature_decoder_layers):
            self.decoder_layers.append(
                nng.HeteroConv(
                    {
                        edge_type: conv_type(hidden_channels, hidden_channels)
                        for edge_type in edge_types
                    },
                    aggr="sum",
                ))

        decoder_last_conv_layer = nng.HeteroConv(
            {
                edge_type:
                conv_type(hidden_channels, features_channels[edge_type[2]])
                for edge_type in edge_types
            },
            aggr="sum",
        )

        self.decoder_layers.append(decoder_last_conv_layer)

        # Graph classifier head
        self.graph_edge_prediction = StructureDecoder(hidden_channels,
                                                      hidden_channels, 50, 10)

    def EncoderLayer(self, hetero_data, in_channels, out_channels, conv_type):

        conv_layer = nng.HeteroConv(
            {
                edge_type: conv_type(in_channels, out_channels)
                for edge_type in hetero_data.edge_types
            },
            aggr="sum",
        )

        return conv_layer

    def forward(self, data, edge):
        """
        Forward pass of the GraphBEAN model.

        Args:
            data (Tensor): The input data.
            edge (Tuple): Tuple containing the source, target, and destination nodes.

        Returns:
            Tuple: A tuple containing the following elements:
                - None: Placeholder for future use.
                - hidden_representation (Tensor): The hidden representation obtained from the encoder layers.
                - feature_out (Tensor): The output of the feature decoding layers.
                - edge_prediction (Tensor): The predicted edge labels.
        """

        # loop through the encoder layers to obtain the hidden representation
        hidden_representation = data.x_dict
        for layer in self.encoder_layers:
            hidden_representation = layer(
                hidden_representation,
                data.edge_index_dict,
            )
            hidden_representation = {
                key: F.relu(x)
                for key, x in hidden_representation.items()
            }

        # Obtain the feature decoding output after looping through the feature decoding layers
        feature_out = hidden_representation
        for layer in self.decoder_layers:
            feature_out = layer(
                feature_out,
                data.edge_index_dict,
            )
            feature_out = {key: F.relu(x) for key, x in feature_out.items()}

        src, to, dst = edge

        edge_prediction = self.graph_edge_prediction(
            hidden_representation[src],
            hidden_representation[dst],
            data[src, to, dst].edge_label_index,
        )

        return None, hidden_representation, feature_out, edge_prediction


class GraphBeanClassifier(nn.Module):

    def __init__(
        self,
        hetero_data,
        n_encoder_layers: int,
        n_feature_decoder_layers: int,
        hidden_channels: int,
        features_channels: Dict,
        class_head_layers: int = 1,
        classes: int = 3,
        conv_type: callable = SAGEConv,
        edge_types=None,
    ):

        super().__init__()

        self.data_types = hetero_data.metadata()[0]

        self.graph_bean_layer = GraphBEAN(
            hetero_data,
            n_encoder_layers,
            n_feature_decoder_layers,
            hidden_channels,
            features_channels,
            conv_type,
            edge_types,
        )

        self.classifierHead = nn.ModuleList([
            nng.HeteroDictLinear(hidden_channels, hidden_channels,
                                 hetero_data.metadata()[0])
        ])

        for _ in range(class_head_layers):
            self.classifierHead.append(
                nng.HeteroDictLinear(hidden_channels, hidden_channels,
                                     hetero_data.metadata()[0]))
        self.classifierHead.append(
            nng.HeteroDictLinear(hidden_channels, classes,
                                 hetero_data.metadata()[0]))

    def forward(self, data, edge):
        _, hidden_representation, feature_out, edge_prediction = self.graph_bean_layer(
            data, edge)

        class_probs = hidden_representation
        for layer in self.classifierHead:
            class_probs = layer(class_probs)

        output = {}
        for key in self.data_types:
            output[key] = torch.sigmoid(class_probs[key])

        return class_probs, hidden_representation, feature_out, edge_prediction


class GraphBEANModule(L.LightningModule):

    # TODO: clarify the api and settings!?
    def __init__(
        self,
        edge,
        edge_types,
        loss_fn=None,
        learning_rate=0.01,
        encoder_layers=2,
        decoder_layers=2,
        hidden_layers=128,
        classifier: bool = False,
        class_head_layers=3,
        classes=2,
        predict="wallets",
        data=None,
        conv_type: callable = SAGEConv,
    ):
        """
        Initializes the GraphBEAN model.

        Args:
            edge (str): The edge type.
            model (torch.nn.Module, optional): The underlying model architecture. Defaults to None.
            loss_fn (callable, optional): The loss function. Defaults to None.
            learning_rate (float, optional): The learning rate. Defaults to 0.01.
            encoder_layers (int, optional): The number of encoder layers. Defaults to 2.
            decoder_layers (int, optional): The number of decoder layers. Defaults to 2.
            hidden_layers (int, optional): The number of hidden layers. Defaults to 128.
            classifier (bool, optional): Whether the model is a classifier. Defaults to False.
            class_head_layers (int, optional): The number of layers in the classification head. Defaults to 3.
            classes (int, optional): The number of classes. Defaults to 2.
            predict (str, optional): The prediction type. Defaults to "wallets".
            data (torch_geometric.data.Data, optional): The dataset. Defaults to None.
            conv_type (callable, optional): The convolutional layer type. Defaults to SAGEConv.
        """
        super().__init__()
        self.edge = edge
        self.predict = predict

        if data:
            self.dataset = data
        else:
            self.dataset = None

        if classifier:
            self.loss_fn = loss_fn if loss_fn is not None else GraphBEANLossClassifier
        else:
            self.loss_fn = loss_fn if loss_fn is not None else GraphBEANLoss

        self.learning_rate = learning_rate
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.hidden_layers = hidden_layers
        self.classifier = classifier
        self.class_head_layers = class_head_layers
        self.classes = classes
        self.conv_type = conv_type

        self.edge_types = edge_types
        if isinstance(self.edge_types, str):
            self.edge_types = [self.edge_types]

        # Convert edge_type strings to tuples
        self.edge_types = [
            tuple(edge_type.split('_'))
            if isinstance(edge_type, str) else edge_type
            for edge_type in self.edge_types
        ]

        print(f"Edge types:{self.edge_types}")

        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=classes, average="macro")

        self.f1 = torchmetrics.classification.F1Score(task="multiclass",
                                                      num_classes=classes,
                                                      average="macro")

        self.recall = torchmetrics.classification.Recall(task="multiclass",
                                                         num_classes=classes,
                                                         average="macro")

        self.precision = torchmetrics.classification.Precision(
            task="multiclass", num_classes=classes, average="macro")

        self.confmat = torchmetrics.classification.ConfusionMatrix(
            task="multiclass", num_classes=classes)

        self.aucroc = torchmetrics.classification.AUROC(task="multiclass",
                                                        num_classes=classes,
                                                        average="macro")

        self.validation_step_outputs = []

        self.save_hyperparameters()

    def setup(self, stage=None):
        """
        Function called before validate and fit.
        Sets up the model and optimizers based on the provided dataset and configuration.

        Args:
            stage: Optional[str], the current stage of training. Defaults to None.

        Returns:
            None
        """

        # Get the dataset from the datamodule
        if self.dataset is None:
            self.dataset = self.trainer.datamodule.dataset

        # Required for the feature encoder-decoder
        mapping = dict()
        for key in self.dataset.metadata()[0]:
            mapping[key] = self.dataset[key].x.shape[1]

        if self.classifier:
            # Initialize your model using the dataset
            self.model = GraphBeanClassifier(
                self.dataset,
                self.encoder_layers,
                self.decoder_layers,
                self.hidden_layers,
                mapping,
                self.class_head_layers,
                self.classes,
                self.conv_type,
                self.edge_types,
            )

        else:
            # Initialize your model using the dataset
            self.model = GraphBEAN(
                self.dataset,
                self.encoder_layers,
                self.decoder_layers,
                self.hidden_layers,
                mapping,
                self.conv_type,
                self.edge_types,
            )

        self.optimizers = optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate)

    def forward(self, batch, edge):
        """
        Forward pass of the graphBEAN model.

        Args:
            batch: The input batch.
            edge: The input edge.

        Returns:
            The output of the model.
        """
        if VERBOSE:
            print(f"forward module batch:{batch} edge:{edge}")
        return self.model(batch, edge)

    def loss(self, batch, class_probs, pred_features, pred_edges):
        if self.classifier:
            loss = self.loss_fn(
                pred_features,
                pred_edges,
                batch,
                self.edge,
                class_probs[self.predict],
                batch[self.predict].y,
            )
        else:
            loss = self.loss_fn(pred_features, pred_edges, batch, self.edge)

        return loss

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step on the given batch of data.

        Args:
            batch: The input batch of data.
            batch_idx: The index of the current batch.

        Returns:
            The loss value computed during the training step.
        """
        class_probs, _, pred_features, pred_edges = self(batch, self.edge)

        loss = self.loss(batch, class_probs, pred_features, pred_edges)

        self.log("train_loss", loss, on_step=True, prog_bar=True)

        node_ground_truth = batch[self.predict].y
        idx = node_ground_truth != 2
        batch_subset = node_ground_truth[idx]

        if self.classifier:
            output_class = torch.argmax(class_probs[self.predict],
                                        dim=1).long()
            output_class_subset = output_class[idx]
            self.accuracy(output_class_subset, batch_subset)
            self.log(
                "train_accuracy",
                self.accuracy,
                prog_bar=True,
                on_step=True,
                on_epoch=False,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Performs a validation step on the given batch of data.

        Args:
            batch: The batch of data for validation.
            batch_idx: The index of the current batch.

        Returns:
            None
        """
        class_probs, _, pred_features, pred_edges = self(batch, self.edge)
        loss = self.loss(batch, class_probs, pred_features, pred_edges)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=64,
        )

        node_ground_truth = batch[self.predict].y
        idx_filter = node_ground_truth != 2
        batch_subset = node_ground_truth[idx_filter]

        if self.classifier:
            output_class = torch.argmax(class_probs[self.predict],
                                        dim=1).long()
            output_class_subset = output_class[idx_filter]
            self.accuracy(output_class_subset, batch_subset)
            self.log(
                "val_acc_step",
                self.accuracy,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=64,
            )
            self.f1(output_class_subset, batch_subset)
            self.log(
                "val_f1",
                self.f1,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=64,
            )
            self.recall(output_class_subset, batch_subset)
            self.log(
                "val_recall",
                self.recall,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=64,
            )
            self.precision(output_class_subset, batch_subset)
            self.log(
                "val_precision",
                self.precision,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=64,
            )

    def on_validation_epoch_end(self):
        """
        Performs operations at the end of each validation epoch.

        This method clears the validation step outputs.

        Returns:
            None
        """

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """
        Configures the optimizers for the model.

        Returns:
            optimizers: The configured optimizers for the model.
        """
        return self.optimizers
