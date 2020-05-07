import torch
import torch.nn as nn

from fairseq.models import register_model

import models
from fairseq.models import ARCH_MODEL_REGISTRY
from models.encoder.roberta import RobertaWrapper

@register_model('multi_model')
class MultiModel(object):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""

        RobertaWrapper.add_args(parser)

    @classmethod
    def build_model(cls, args, task):
        encoder = RobertaWrapper.build_model(args, task)
        model_dict = nn.ModuleDict()
        for task_name, sub_task in task.tasks:
            model_dict[task_name] = ARCH_MODEL_REGISTRY[args.arch].build_model(sub_task.args, sub_task, encoder=encoder)
        return model_dict



@register_model_architecture('encoder_triplet', 'encoder_triplet__roberta_base')
def triplet_base_architecture(args):
    base_architecture(args)


@register_model_architecture('encoder_triplet', 'encoder_triplet__roberta_large')
def triplet_large_architecture(args):
    large_architecture(args)


@register_model_architecture('encoder_triplet', 'encoder_triplet__roberta_small')
def triplet_small_architecture(args):
    small_architecture(args)
