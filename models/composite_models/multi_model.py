import torch
import torch.nn as nn

from fairseq.models import register_model, register_model_architecture, BaseFairseqModel

import models
from fairseq.models import ARCH_MODEL_REGISTRY
from models.encoder.roberta import RobertaWrapper

@register_model('multi_model')
class MultiModel(BaseFairseqModel):
    def __init__(self, args, encoder, model_dict):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.model_dict = model_dict

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        RobertaWrapper.add_args(parser)

    @classmethod
    def build_model(cls, args, task):
        encoder = RobertaWrapper.build_model(args, task)
        model_dict = nn.ModuleDict()
        for task_name, sub_task in task.tasks.items():
            task_override_args = args.tasks[task_name]
            if 'arch' in task_override_args:
                model_dict[task_name] = ARCH_MODEL_REGISTRY[task_override_args['arch']].build_model(sub_task.args, sub_task, encoder=encoder)
            else:
                model_dict[task_name] = encoder
        return cls(args, encoder, model_dict)

@register_model_architecture('multi_model', 'multi_model')
def multi_model(args):
    pass
