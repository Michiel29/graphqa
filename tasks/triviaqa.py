import logging
import os

from fairseq.tasks import register_task

from tasks import BaseTask
from datasets import (
    TriviaQADataset,
    FixedSizeDataset,
    PrependTokenDataset,
)
from utils.data_utils import (
    safe_load_indexed_dataset,
)
from utils.numpy_utils import MMapNumpyArray
from utils.dictionary import CustomDictionary


logger = logging.getLogger(__name__)


@register_task('triviaqa')
class TriviaQATask(BaseTask):
    """Task for training inference models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""

        """Required either in config or cl"""
        parser.add_argument('--data_path', help='path to data')
        parser.add_argument('--n_way', default=5, help='number of few-shot classes')
        parser.add_argument('--n_shot', default=1, help='number of few-shot examples')

    @classmethod
    def setup_task(cls, args, **kwargs):
        dict_path = os.path.join(args.data_path, 'dict.txt')
        dictionary = CustomDictionary.load(dict_path)
        logger.info('dictionary: {} types'.format(len(dictionary)))
        return cls(args, dictionary, None)

    def load_dataset(self, split, prune_type=None, prune_param=None, epoch=0, combine=False, **kwargs):

        questions = safe_load_indexed_dataset(
            os.path.join(self.args.data_path, split + '.questions_entities'),
        )

        answers = MMapNumpyArray(
            os.path.join(self.args.data_path, split + '.answer_entities.npy'),
        )



        dataset = PrependTokenDataset(dataset, self.dictionary.bos(), ['text', 'exemplars'])

        n_examples = getattr(self.args, 'n_' + split + '_examples', None)
        if n_examples is not None:
            dataset = FixedSizeDataset(
                dataset=dataset,
                size=n_examples,
                seed=self.seed,
            )

        self.datasets[split] = dataset
