import copy
import logging
import torch

from fairseq.optim.adam import Adam
from fairseq.optim import FairseqOptimizer, register_optimizer
from fairseq.optim.fused_adam import get_fused_adam_class


logger = logging.getLogger(__name__)


def get_param_group_index(groups, param_name):
    index = None
    for i, group in enumerate(groups):
        if (
            (len(group['prefix']) == 0 or group['prefix'] in param_name)
            and (index is None or len(groups[index]['prefix']) < len(group['prefix']))
        ):
            index = i
    return index


@register_optimizer('multi_adam')
class MultiAdamOptimizer(FairseqOptimizer):
    def __init__(self, args, params):
        super().__init__(args)
        param_groups = copy.deepcopy(self.args.optimizers)
        for param_group in param_groups:
            param_group['params'] = []
        for param_name, param in params:
            param_index = get_param_group_index(param_groups, param_name)
            param_groups[param_index]['params'].append(param)
        for param_group in param_groups:
            if len(param_group['params']) == 0:
                logging.error('Parameters %s' % ','.join([x[0] for x in params]))
                logging.warning('Failed to match any parameter for the group %s' % param_group)

        fused_adam_cls = get_fused_adam_class()
        use_fused_adam = fused_adam_cls is not None and torch.cuda.is_available()
        if use_fused_adam:
            logger.info('using FusedAdam')
            self._optimizer = fused_adam_cls(param_groups, **self.optimizer_config)
        else:
            self._optimizer = Adam(param_groups, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--adam-betas', default='(0.9, 0.999)', metavar='B',
                            help='betas for Adam optimizer')
        parser.add_argument('--adam-eps', type=float, default=1e-8, metavar='D',
                            help='epsilon for Adam optimizer')
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        # fmt: on

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            'lr': self.args.lr[0],
            'betas': eval(self.args.adam_betas),
            'eps': self.args.adam_eps,
            'weight_decay': self.args.weight_decay,
            # TODO: This is probably not correct
            # 'optimizers': self.args.optimizers,
        }

    def average_params(self):
        """Reduce Params is only used during BMUF distributed training."""
        state_dict = self.optimizer.state_dict()
        total_gpus = float(dist.get_world_size())

        for _, value in state_dict["state"].items():
            value["exp_avg"] /= total_gpus
            value["exp_avg_sq"] /= total_gpus
            dist.all_reduce(value["exp_avg"], op=dist.ReduceOp.SUM)
            dist.all_reduce(value["exp_avg_sq"], op=dist.ReduceOp.SUM)
