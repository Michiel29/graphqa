import logging
import os

from fairseq.tasks import register_task
from fairseq import metrics

from tasks import BaseTask
from datasets import (
    AnnotatedText,
    TACREDDataset,
    PrependTokenDataset,
    FixedSizeDataset,
)
from utils.data_utils import (
    safe_load_indexed_dataset,
)
from utils.dictionary import CustomDictionary
from utils.logging_utils import compute_confusion_matrix, MicroF1Meter
from utils.numpy_utils import MMapNumpyArray

logger = logging.getLogger(__name__)


@register_task('tacred')
class TACREDTask(BaseTask):
    """Task for training inference models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""

        """Required either in config or cl"""
        parser.add_argument('--data_path', help='path to data')

    @classmethod
    def setup_task(cls, args, **kwargs):
        dict_path = os.path.join(args.data_path, 'dict.txt')
        dictionary = CustomDictionary.load(dict_path)
        logger.info('dictionary: {} types'.format(len(dictionary)))
        return cls(args, dictionary, None)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        text_data = safe_load_indexed_dataset(
            os.path.join(self.args.data_path, split + '.text'),
        )
        annotation_data = MMapNumpyArray(
            os.path.join(self.args.data_path, split + '.annotations.npy'),
        )
        annotated_text = AnnotatedText(
            text_data=text_data,
            annotation_data=annotation_data,
            dictionary=self.dictionary,
            mask_type=self.args.mask_type,
            non_mask_rate=self.args.non_mask_rate,
        )
        relation_dataset = safe_load_indexed_dataset(
            os.path.join(self.args.data_path, split + '.relations')
        )

        dataset = TACREDDataset(
            annotation_text=annotated_text,
            relation_dataset=relation_dataset,
            dictionary=self.dictionary,
            seed=self.seed,
        )
        dataset = PrependTokenDataset(dataset, self.dictionary.bos(), ['text'])

        n_examples = getattr(self.args, 'n_' + split + '_examples', None)
        if n_examples is not None:
            dataset = FixedSizeDataset(
                dataset=dataset,
                size=n_examples,
                seed=self.seed,
            )

        self.datasets[split] = dataset

    def reporter(self, target, pred, logging_output):
        fn, tp, fp = compute_confusion_matrix(
            target=target.cpu().numpy(),
            pred=pred.detach().cpu().numpy(),
            avg='micro',
            num_classes=self.args.num_classes,
            ignore_classes=[self.args.num_classes-1]
        )
        logging_output['fn'] = fn
        logging_output['tp'] = tp
        logging_output['fp'] = fp
        return logging_output

    def reduce_metrics(self, logging_outputs, criterion, prefix=''):
        super().reduce_metrics(logging_outputs, criterion)

        sample_size = sum(log.get(prefix + 'sample_size', 0) for log in logging_outputs)
        weight = 0 if self.split == 'train' else sample_size

        fn = sum(log.get(prefix + 'fn', 0) for log in logging_outputs)
        tp = sum(log.get(prefix + 'tp', 0) for log in logging_outputs)
        fp = sum(log.get(prefix + 'fp', 0) for log in logging_outputs)
        metrics.log_custom(MicroF1Meter, 'micro_f1', fn, tp, fp, self.split, weight)