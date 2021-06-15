from transformers import RobertaModel, BertPreTrainedModel, RobertaConfig, XLMRobertaModel, BertPreTrainedModel, XLMRobertaConfig
import torch
import torch.nn as nn
# from torch.nn import CrossEntropyLoss, KLDivLoss

from modeling_roberta import RobertaForTokenClassification_v2


XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "xlm-roberta-base",
    "xlm-roberta-large",
    "xlm-roberta-large-finetuned-conll02-dutch",
    "xlm-roberta-large-finetuned-conll02-spanish",
    "xlm-roberta-large-finetuned-conll03-english",
    "xlm-roberta-large-finetuned-conll03-german",
    # See all XLM-RoBERTa models at https://huggingface.co/models?filter=xlm-roberta
]

class XLMRobertaForTokenClassification_v2(RobertaForTokenClassification_v2):
    """
    This class overrides :class:`~transformers.RobertaForTokenClassification`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig
