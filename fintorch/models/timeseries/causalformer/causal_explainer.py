import torch
from typing import Tuple, Any


class causal_explanation:
    def __init__(self, model: Any) -> None:
        self.model = model

    def generate_relevance_propagation_for_batch(
        self, batch_size: int, input: torch.Tensor, interpreted_series: int
    ) -> None:
        # TODO: maybe we can make this part of the module
        # Create batches of the inputs
        pass

    def _generate_RP(self, input: torch.Tensor, interpreted_series: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # input: [total_batch, input_window, number_of_series, feature_dim]
        # index of the series we interpret (find causal relatinship with)

        # pass through the model
        output = self.model(input)

        # Create a one-hot tensor for the interpreted series
        one_hot = torch.zeros_like(output, dtype=torch.float).to(output.device)
        one_hot[:, :, interpreted_series, :] = 1
        # Clone the one-hot tensor and set requires_grad to True for gradient computation
        one_hot_vector = one_hot.clone()
        one_hot.requires_grad_(True)
        # Compute the dot product of one-hot tensor and model output
        one_hot = torch.sum(one_hot * output)
        # Reset gradients and perform backward pass
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)  # type: ignore
        # Apply regression relevance propagation to calculate relevance scores
        self.model.relprop(one_hot_vector)
        relAs = []
        relKs = []
        # collect causal scores from each encoder layers (in practice, there is only one encoder layer)
        for layer in self.model.encoder.layers:
            # gradient modulation
            relA = layer.attention.attention.get_rel() * torch.abs(
                layer.attention.attention.get_grad()
            )
            relK = layer.attention.Wv.get_rel() * torch.abs(
                layer.attention.Wv.get_grad()
            )

            # w/o interpretation
            # relA = layer.attention.attention.get_wgt()
            # relK = layer.attention.Wv.get_wgt()

            relA = relA.clamp(
                min=0
            )  # only the positive causal scores are taken into consideration
            relK = relK.clamp(
                min=0
            )  # only the positive causal scores are taken into consideration
            relAs.append(relA.mean((0, 1)))  # mean for sample and head
            relKs.append(relK.mean(0))  # mean for head
        relA = torch.stack(relAs).prod(
            0
        )  # multiply each sub-tensor along the `encoder layer` dimension
        relK = torch.stack(relKs).prod(
            0
        )  # multiply each sub-tensor along the `encoder layer` dimension
        return relA, relK
