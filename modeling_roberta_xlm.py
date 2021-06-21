#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import RobertaModel, BertPreTrainedModel, RobertaConfig, XLMRobertaModel, BertPreTrainedModel, XLMRobertaConfig
import torch
import torch.nn as nn
# from torch.nn import CrossEntropyLoss, KLDivLoss

from modeling_roberta import RobertaForTokenClassification_v2
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, KLDivLoss
from transformers.modeling_outputs import (
    TokenClassifierOutput,
)
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel

XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "xlm-roberta-base",
    "xlm-roberta-large",
    "xlm-roberta-large-finetuned-conll02-dutch",
    "xlm-roberta-large-finetuned-conll02-spanish",
    "xlm-roberta-large-finetuned-conll03-english",
    "xlm-roberta-large-finetuned-conll03-german",
    # See all XLM-RoBERTa models at https://huggingface.co/models?filter=xlm-roberta
]
#from modules.transformer import TransformerEncoder
#
#after_norm=True
#n_heads=6
#head_dims=128
#num_layers=2
#attn_type='transformer'
#trans_dropout=0.45
#d_model = n_heads * head_dims
#dim_feedforward = int(2 * d_model)
#dropout=0.45

class XLMRobertaForTokenClassification_v2(RobertaForTokenClassification_v2):
    """
    This class overrides :class:`~transformers.RobertaForTokenClassification`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config=XLMRobertaConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
#        self.transformer = TransformerEncoder(num_layers, d_model, n_heads, dim_feedforward, dropout,
#                                              after_norm=after_norm, attn_type=attn_type,
#                                              scale=attn_type == 'transformer', dropout_attn=None,
#                                              pos_embed='sin')

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        label_mask=None,
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        final_embedding = outputs[0]
        sequence_output = self.dropout(final_embedding)
        logits = self.classifier(sequence_output)

        outputs = (logits, final_embedding, ) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:

            # Only keep active parts of the loss
            if attention_mask is not None or label_mask is not None:
                active_loss = True
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                if label_mask is not None:
                    active_loss = active_loss & label_mask.view(-1)
                active_logits = logits.view(-1, self.num_labels)[active_loss]


            if labels.shape == logits.shape:
                loss_fct = KLDivLoss()
                if attention_mask is not None or label_mask is not None:
                    active_labels = labels.view(-1, self.num_labels)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits, labels)
            else:
                loss_fct = CrossEntropyLoss()
                if attention_mask is not None or label_mask is not None:
                    active_labels = labels.view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))


            outputs = (loss,) + outputs

        return outputs  # (loss), scores, final_embedding, (hidden_states), (attentions)
