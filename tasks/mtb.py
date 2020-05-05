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
    MTBDataset,
    MTBTripletsDataset,
)
from utils.data_utils import safe_load_indexed_dataset
from utils.numpy_utils import MMapNumpyArray
from tasks import RelationInferenceTask

logger = logging.getLogger(__name__)


@register_task('mtb')
class MTBTask(RelationInferenceTask):
    """Task for training inference models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""

        """Required either in config or cl"""
        parser.add_argument('--k_weak_negs', type=float,
                            help='number of weak negatives per positive')
        parser.add_argument('--n_tries_entity', type=int,
                            help='number of attempts to sample entity candidates')

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        train_text_data = safe_load_indexed_dataset(
            os.path.join(self.args.data_path, 'train.text'),
        )
        train_annotation_data = MMapNumpyArray(
            os.path.join(self.args.data_path, 'train.annotations.npy')
        )
        train_annotated_text = AnnotatedText(
            text_data=train_text_data,
            annotation_data=train_annotation_data,
            dictionary=self.dictionary,
            mask_type=self.args.mask_type,
            non_mask_rate=self.args.non_mask_rate,
        )

        if split == 'train':
            split_annotated_text = train_annotated_text
        else:
            split_text_data = safe_load_indexed_dataset(
                os.path.join(self.args.data_path, split + '.text'),
            )
            split_annotation_data = MMapNumpyArray(
                os.path.join(self.args.data_path, split + '.annotations.npy')
            )
            split_annotated_text = AnnotatedText(
                text_data=split_text_data,
                annotation_data=split_annotation_data,
                dictionary=self.dictionary,
                mask_type=self.args.mask_type,
                non_mask_rate=self.args.non_mask_rate,
            )

        train_graph_data = safe_load_indexed_dataset(
            os.path.join(self.args.data_path, 'train.graph'),
        )
        train_graph = GraphDataset(
            edges=train_graph_data,
            subsampling_strategy=self.args.subsampling_strategy,
            subsampling_cap=self.args.subsampling_cap,
            seed=self.args.seed,
        )

        split_graph_data = safe_load_indexed_dataset(
            os.path.join(self.args.data_path, 'mtb_' + split + '.graph'),
        )
        split_graph = GraphDataset(
            edges=split_graph_data,
            subsampling_strategy=self.args.subsampling_strategy,
            subsampling_cap=self.args.subsampling_cap,
            seed=self.args.seed,
        )

        dataset = MTBDataset(
            split=split,
            split_annotated_text=split_annotated_text,
            train_annotated_text=train_annotated_text,
            split_graph=split_graph,
            train_graph=train_graph,
            seed=self.args.seed,
            dictionary=self.dictionary,
            k_weak_negs=self.args.k_weak_negs,
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