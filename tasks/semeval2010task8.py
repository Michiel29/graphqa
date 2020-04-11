import logging
import os
import collections

from fairseq.tasks import register_task
from fairseq import metrics

from tasks import BaseTask
from datasets import (
    AnnotatedText,
    SemEval2010Task8Dataset,
    PrependTokenDataset,
    FixedSizeDataset,
)
from utils.data_utils import (
    safe_load_indexed_dataset,
)
from utils.dictionary import CustomDictionary
from utils.logging_utils import (
    compute_confusion_matrix, 
    reduce_macro_mcm, 
    MacroF1Meter
)
from utils.numpy_utils import MMapNumpyArray

logger = logging.getLogger(__name__)


@register_task('semeval2010task8')
class SemEval2010Task8Task(BaseTask):
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

        dataset = SemEval2010Task8Dataset(
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

    def reporter(self, pred, target, logging_output):
        fn, tp, fp = compute_confusion_matrix(
            target=target.cpu().numpy(),
            pred=pred.detach().cpu().numpy(),
            avg='macro',
            num_classes=self.args.num_classes
        )
        for i in fn.keys():
            logging_output['fn_' + str(i)] = fn[i]
            logging_output['tp_' + str(i)] = tp[i]
            logging_output['fp_' + str(i)] = fp[i]
        return logging_output

    def reduce_metrics(self, logging_outputs, criterion, prefix=''):
        super().reduce_metrics(logging_outputs, criterion)

        fn, tp, fp = reduce_macro_mcm(logging_outputs, self.args.num_classes, prefix)
        sample_size = sum(log.get(prefix + 'sample_size', 0) for log in logging_outputs)
        weight = 0 if self.split == 'train' else sample_size

        metrics.log_custom(MacroF1Meter, 'macro_f1', fn, tp, fp, self.split, [self.args.num_classes-1], True, weight)
        if self.split == 'train':
            metrics.log_custom(MacroF1Meter, 'macro_f1_avg', fn, tp, fp, self.split, [self.args.num_classes-1], True, sample_size)