from copy import deepcopy
from typing import Literal, Union

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

PoolingType = Literal["last", "avg"]

# TODO: add a model name to rootmodel name dictionary so discriminator support fine-tuned model

"""
tensor shape: 
B: Batch size 
M: model dimension / residual stream dimension
S: sequence length
C: number of class in classification task

"""


class Discriminator(nn.Module):
    """Transformer Followed by a Linear Prob"""

    def __init__(
        self,
        pretrained_model: str,
        pooling_type: PoolingType,
        device: str,
    ) -> "Discriminator":

        super(Discriminator, self).__init__()
        if pooling_type not in ["last", "avg"]:
            raise ValueError(f"Invalid pooling type {pooling_type}")

        self.pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"

        if "mGPT" in pretrained_model:
            d_resid = self.pretrained_model.config.n_embd
        elif "bloom" in pretrained_model:
            d_resid = self.pretrained_model.config.hidden_size
        else:
            raise ValueError(f"Model not supported: {pretrained_model}")

        # Probabilistic head for binary classification
        self.probe = nn.Linear(in_features=d_resid, out_features=1, bias=False)

        # Disable grad for pretrained_model, turn on eval mode
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        self.pretrained_model.eval()

        self.pretrained_model.to(device)
        self.probe.to(device)

        self.pooling_type = pooling_type
        self.device = device

    def forward(self, sentences) -> Tensor:
        tokenizer_output: Tensor = self.tokenizer(
            sentences, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        token_ids_BSM: Tensor = tokenizer_output.input_ids
        attn_mask_BS: Tensor = tokenizer_output.attention_mask

        assert hasattr(self.pretrained_model, "transformer")

        lm_output_BSM: Tensor = self.pretrained_model.transformer(token_ids_BSM)[0]
        if self.pooling_type == "avg":
            probe_input_BM = self.avg_pool(lm_output_BSM, attn_mask_BS)
        else:
            raise NotImplementedError(f"{self.pooling_type} not implemented yet")

        probe_output_BC: Tensor = self.probe(probe_input_BM)

        output_BC: Tensor = torch.sigmoid(probe_output_BC)
        return output_BC

    def avg_pool(self, hidden_states_BSM: Tensor, attn_mask_BS: Tensor):
        """Compute sentence embedding by taking the average of all non-padding embeddings"""
        attn_mask_BSM: Tensor = attn_mask_BS.unsqueeze(dim=-1).expand_as(
            hidden_states_BSM
        )
        masked_hidden_states_BSM = attn_mask_BSM * hidden_states_BSM
        sum_hidden_states_BM = masked_hidden_states_BSM.sum(dim=1)
        token_counts_B = attn_mask_BS.sum(dim=1, keepdim=True)
        avg_hidden_states_BM = sum_hidden_states_BM / token_counts_B
        return avg_hidden_states_BM

    def predict(self, sentences):
        """Predict class label given texts"""
        prob = self(sentences)
        return int(prob >= 0.5)

    @property
    def probe_model(self) -> nn.Module:
        return deepcopy(self.probe).to("cpu")

    @classmethod
    def from_cache(
        cls,
        pretrained_model: str,
        probe_path_or_module: Union[str, nn.Module],
        pooling_type: PoolingType = "avg",
        device: str = "cpu",
    ) -> "Discriminator":

        new_discriminator = Discriminator(
            pretrained_model=pretrained_model, pooling_type=pooling_type, device=device
        )

        if isinstance(probe_path_or_module, str):
            probe = torch.load(probe_path_or_module, map_location=device)
        elif isinstance(probe_path_or_module, nn.Module):
            probe = probe_path_or_module
        else:
            raise ValueError("Not supported probe passed in")
        new_discriminator.probe = probe
        return new_discriminator

    def to(self, device):
        self.device = device
        self.probe = self.probe.to(device)
        self.pretrained_model = self.pretrained_model.to(device)
        return self
