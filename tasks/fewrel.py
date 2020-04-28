import logging
import os

from fairseq.tasks import register_task

from tasks import BaseTask
from datasets import (
    AnnotatedText,
    FewRelDataset,
    FixedSizeDataset,
    PrependTokenDataset,
)
from utils.data_utils import (
    safe_load_indexed_dataset,
)
from utils.numpy_utils import MMapNumpyArray
from utils.dictionary import CustomDictionary


logger = logging.getLogger(__name__)


@register_task('fewrel')
class FewRelTask(BaseTask):
    """Task for training inference models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""

        """Required either in config or cl"""
        parser.add_argument('--data_path', help='path to data')
        parser.add_argument('--n_way', default=5, help='number of few-shot classes')
        parser.add_argument('--n_shot', default=1, help='number of few-shot examples')
        # parser.add_argument('--n_train_relations', default=64, help='number of total classes')
        # parser.add_argument('--n_train_examples_per_relation', default=700, help='number of total examples per class')

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

        dataset = FewRelDataset(
            annotation_text=annotated_text,
            relation_dataset=relation_dataset,
            dictionary=self.dictionary,
            n_way=self.args.n_way,
            n_shot=self.args.n_shot,
            # n_train_relations=self.args.n_train_relations,
            # n_train_examples_per_relation=self.args.n_train_examples_per_relation,
            seed=self.seed,
        )
        # if split == 'train':
            # dataset.prune_by_prune_by_num_relations()
        dataset = PrependTokenDataset(dataset, self.dictionary.bos(), ['text', 'exemplars'])

        n_examples = getattr(self.args, 'n_' + split + '_examples', None)
        if n_examples is not None:
            dataset = FixedSizeDataset(
                dataset=dataset,
                size=n_examples,
                seed=self.seed,
            )

        self.datasets[split] = dataset
