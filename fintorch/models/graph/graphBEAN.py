from typing import Dict

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.nn as nng
import torchmetrics
from torch_geometric.nn import SAGEConv


def GraphBEANLoss(feature_predictions, edge_predictions,
                  ground_truth_sampled_data, edge):
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
    # Loss of the feature reconstruction per node type
    loss = GraphBEANLoss(feature_predictions, edge_predictions,
                         ground_truth_sampled_data, edge)

    classification_loss = F.binary_cross_entropy_with_logits(
        node_pred, node_ground_truth)

    # Total loss function
    total_loss = loss + classification_loss

    return total_loss


class LinkClassifier(nn.Module):
    """
    LinkClassifier is a PyTorch module that performs link classification in a graph.
    """

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

        return (x_src_features * x_dst_features).sum(dim=-1)


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
        hetero_data,
        n_encoder_layers: int,
        n_feature_decoder_layers: int,
        hidden_channels: int,
        features_channels: dict,
        conv_type: callable,
    ):
        # GATConv, GATv2Conv, SAGEConv, GraphConv, ResGatedGraphConv, TransformerConv, MFConv, RGCNConv, GMMConv,
        # SplineConv, NNConv, CGConv, PointTransformerConv, LEConv, GENConv, FiLMConv, GeneralConv,
        super().__init__()

        assert n_encoder_layers > 0, "Number of encoder layers must be greater than 0."
        assert (
            n_feature_decoder_layers
            > 0), "Number of feature decoder layers must be greater than 0."
        assert hidden_channels > 0, "Number of hidden channels must be greater than 0."

        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        self.encoder_layers.append(
            self.EncoderLayer(hetero_data, (-1, -1), hidden_channels,
                              conv_type))  # Pass conv_type argument

        for _ in range(n_encoder_layers - 1):
            self.encoder_layers.append(
                nng.HeteroConv(
                    {
                        edge_type: conv_type(hidden_channels, hidden_channels)
                        for edge_type in hetero_data.edge_types
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
                        for edge_type in hetero_data.edge_types
                    },
                    aggr="sum",
                ))

        decoder_last_conv_layer = nng.HeteroConv(
            {
                edge_type:
                SAGEConv(hidden_channels, features_channels[edge_type[2]])
                for edge_type in hetero_data.edge_types
            },
            aggr="sum",
        )

        self.decoder_layers.append(decoder_last_conv_layer)

        # Graph classifier head
        self.graph_edge_prediction = LinkClassifier()

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

        # Obtain the feature decoding output after looping through the feature decoding layers
        feature_out = hidden_representation
        for layer in self.decoder_layers:
            feature_out = layer(
                feature_out,
                data.edge_index_dict,
            )

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
    ):

        super().__init__()

        self.data_types = hetero_data.metadata()[0]

        self.graph_bean_layer = GraphBEAN(
            hetero_data,
            n_encoder_layers,
            n_feature_decoder_layers,
            hidden_channels,
            features_channels,
            SAGEConv,
        )

        self.classifierHead = nn.ModuleList([
            nng.HeteroDictLinear(hidden_channels, hidden_channels,
                                 hetero_data.metadata()[0])
        ])

        for i in range(class_head_layers):
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

    def __init__(
        self,
        edge,
        model=None,
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
    ):
        super().__init__()
        self.model = model
        self.edge = edge
        self.predict = predict

        if data:
            self.dataset = data
        else:
            self.dataset = None

        if classifier:
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            self.loss_fn = loss_fn if loss_fn is not None else GraphBEANLoss

        self.learning_rate = learning_rate
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.hidden_layers = hidden_layers
        self.classifier = classifier
        self.class_head_layers = class_head_layers
        self.classes = classes

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
        # Get the dataset from the datamodule
        if self.dataset is None:
            self.dataset = self.trainer.datamodule.dataset

        mapping = dict()
        for key in self.dataset.metadata()[0]:
            mapping[key] = self.dataset[key].x.shape[1]

        if self.classifier:
            # Initialize your model using the dataset
            self.model = (self.model
                          if self.model is not None else GraphBeanClassifier(
                              self.dataset,
                              self.encoder_layers,
                              self.decoder_layers,
                              self.hidden_layers,
                              mapping,
                              self.class_head_layers,
                              self.classes,
                          ))
        else:
            # Initialize your model using the dataset
            self.model = (self.model if self.model is not None else GraphBEAN(
                self.dataset,
                self.encoder_layers,
                self.decoder_layers,
                self.hidden_layers,
                mapping,
                SAGEConv,
            ))
        print(f"Model:{self.model}")
        self.optimizers = optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        class_probs, hidden_representation, pred_features, pred_edges = self.model(
            batch, self.edge)

        if self.classifier:

            loss = self.loss_fn(
                class_probs[self.predict],
                batch[self.predict].y,
            )
        else:
            loss = self.loss_fn(pred_features, pred_edges, batch, self.edge)

        self.log("train_loss", loss)

        # print(f"Class probs:{class_probs} \n\n batch")
        if self.classifier:
            output_class = torch.argmax(class_probs[self.predict],
                                        dim=1).long()
            self.accuracy(output_class, batch[self.predict].y)
            self.log("train_acc_step", self.accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        class_probs, hidden_representation, pred_features, pred_edges = self.model(
            batch, self.edge)
        if self.classifier:
            loss = self.loss_fn(
                class_probs[self.predict],
                batch[self.predict].y,
            )
        else:
            loss = self.loss_fn(pred_features, pred_edges, batch, self.edge)

        self.log("val_loss",
                 loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 batch_size=16)

        if self.classifier:
            # print(f"target:{batch['wallets'].y - 1}")
            output_class = torch.argmax(class_probs[self.predict],
                                        dim=1).long()
            self.accuracy(output_class, batch[self.predict].y)
            self.log(
                "val_acc_step",
                self.accuracy,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=16,
            )
            self.f1(output_class, batch[self.predict].y)
            self.log(
                "val_f1",
                self.f1,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=16,
            )
            self.recall(output_class, batch[self.predict].y)
            self.log(
                "val_recall",
                self.recall,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=16,
            )
            self.precision(output_class, batch[self.predict].y)
            self.log(
                "val_precision",
                self.precision,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=16,
            )
            self.aucroc(
                class_probs[self.predict],
                batch[self.predict].y,
            )
            self.log(
                "val_aucroc",
                self.aucroc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=16,
            )
        self.validation_step_outputs.append({
            "labels":
            batch[self.predict].y,
            "logits":
            class_probs[self.predict],
        })

    def on_validation_epoch_end(self):
        # print(f"all val outputs:{len(self.validation_step_outputs)}")

        # labels = torch.cat([x["labels"] for x in self.validation_step_outputs])
        # logits = torch.cat([x["logits"] for x in self.validation_step_outputs])
        # preds = torch.argmax(logits, dim=1).long()
        # print(f"labs:{labels.cpu().numpy()} log:{logits}")

        # wandb.log({
        #     "conf_mat":
        #     wandb.plot.confusion_matrix(
        #         y_true=labels.cpu().numpy(),
        #         preds=preds.cpu().numpy(),
        #         class_names=["licit", "illicit"],
        #     )
        # })

        # wandb.log({
        #     "pr":
        #     wandb.plot.pr_curve(
        #         labels.cpu().numpy(),
        #         logits.cpu().numpy(),
        #         labels=["licit", "illicit"],
        #         classes_to_plot=None,
        #     )
        # })

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return self.optimizers
