import logging
import os
import json

from fairseq.tasks import register_task

from tasks import BaseTask
from datasets import (
    TriviaQADataset,
    ETPDownstreamDataset,
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

    def __init__(self, args, dictionary, entity_dictionary):
        super().__init__(args, dictionary, entity_dictionary)
        self.sample_sizes_mean = None

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""

        """Required either in config or cl"""
        parser.add_argument('--data_path', help='path to data')

    def load_dataset(self, split, prune_type=None, prune_param=None, epoch=0, combine=False, **kwargs):

        questions = safe_load_indexed_dataset(
            os.path.join(self.args.qa_data_path, split + '.questions_entities'),
        )

        answers = MMapNumpyArray(
            os.path.join(self.args.qa_data_path, split + '.answer_entities.npy'),
        )

        with open(os.path.join(self.args.qa_data_path, split + '.annotations.json')) as f:
            annotations = json.load(f)

        dataset = TriviaQADataset(questions, answers, annotations)

        task_framing = self.args.task_framing

        dataset = PrependTokenDataset(dataset, self.dictionary.bos(), ['question'])

        if task_framing == 'predict_mask':
            edges = safe_load_indexed_dataset(
            os.path.join(self.args.data_path, split + '.graph'),
        )
            dataset = ETPDownstreamDataset(
                dataset=dataset,
                edges=edges,
                dictionary=self.dictionary,
                n_entities=len(self.entity_dictionary),
                total_negatives=self.args.total_negatives,
                seed=self.args.seed,
                split=split,
                )
        else:
            raise Exception

        n_examples = getattr(self.args, 'n_' + split + '_examples', None)
        if n_examples is not None:
            dataset = FixedSizeDataset(
                dataset=dataset,
                size=n_examples,
                seed=self.seed,
            )

        self.datasets[split] = dataset
