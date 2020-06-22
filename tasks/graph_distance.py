from fairseq.tasks import register_task
import logging
import os
# import numpy as np

from datasets import (
    AnnotatedText,
    EpochSplitDataset,
    FixedSizeDataset,
    GraphDataset,
    PrependTokenDataset,
    GraphDistanceDataset,
)
from utils.data_utils import safe_load_indexed_dataset
from utils.numpy_utils import MMapNumpyArray
from tasks import RelationInferenceTask

logger = logging.getLogger(__name__)


@register_task('graph_distance')
class GraphDistanceTask(RelationInferenceTask):
    """Task for training inference models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""

        """Required either in config or cl"""
        parser.add_argument('--n_tries_entity', type=int,
                            help='number of attempts to sample entity candidates')
        parser.add_argument('--split_mode', default=False,
                            help='number of attempts to sample entity candidates')

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):

        text_data_A = safe_load_indexed_dataset(
            os.path.join(self.args.data_path, split + '.text'),
        )
        annotation_data_A = MMapNumpyArray(
            os.path.join(self.args.data_path, split + '.annotations.npy')
        )
        annotated_text_A = AnnotatedText(
            text_data=text_data_A,
            annotation_data=annotation_data_A,
            dictionary=self.dictionary,
            mask_type=self.args.mask_type,
            non_mask_rate=self.args.non_mask_rate,
        )

        graph_data_A = safe_load_indexed_dataset(
            os.path.join(self.args.data_path, 'mtb_' + split + '.graph'),
        )
        graph_A = GraphDataset(
            edges=graph_data_A,
            subsampling_strategy=self.args.subsampling_strategy,
            subsampling_cap=self.args.subsampling_cap,
            seed=self.args.seed,
        )

        if self.args.split_mode:
            annotated_text_B = annotated_text_A
            graph_data_B = safe_load_indexed_dataset(
                os.path.join(self.args.data_path, split + '.graph'),
            )
        else:
            text_data_B = safe_load_indexed_dataset(
                os.path.join(self.args.data_path, 'train.text'),
            )
            annotation_data_B = MMapNumpyArray(
                os.path.join(self.args.data_path, 'train.annotations.npy')
            )
            annotated_text_B = AnnotatedText(
                text_data=text_data_B,
                annotation_data=annotation_data_B,
                dictionary=self.dictionary,
                mask_type=self.args.mask_type,
                non_mask_rate=self.args.non_mask_rate,
            )
            graph_data_B = safe_load_indexed_dataset(
                os.path.join(self.args.data_path, 'train.graph'),
            )

        graph_B = GraphDataset(
            edges=graph_data_B,
            subsampling_strategy=self.args.subsampling_strategy,
            subsampling_cap=self.args.subsampling_cap,
            seed=self.args.seed,
        )

        dataset = GraphDistanceDataset(
            split=split,
            annotated_text_A=annotated_text_A,
            annotated_text_B=annotated_text_B,
            graph_A=graph_A,
            graph_B=graph_B,
            seed=self.args.seed,
            dictionary=self.dictionary,
            class_probabilities=self.args.class_probabilities,
            n_tries_entity=self.args.n_tries_entity
        )
        if split == 'train' and self.args.epoch_size is not None:
            dataset = EpochSplitDataset(
                dataset=dataset,
                epoch_size=self.args.epoch_size,
                seed=self.args.seed,
            )

        dataset = PrependTokenDataset(
            dataset,
            self.dictionary.bos(),
            ['textA', 'textB']
        )

        n_examples = getattr(self.args, 'n_' + split + '_examples', None)
        if n_examples is not None:
            dataset = FixedSizeDataset(
                dataset=dataset,
                size=n_examples,
                seed=self.seed,
            )

        self.datasets[split] = dataset