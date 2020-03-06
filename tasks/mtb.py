import logging
import os
from itertools import combinations
import numpy as np
import torch

from fairseq.data import (
    data_utils,
    iterators,
    FairseqDataset,
    PrependDataset,
    PrependTokenDataset,
    Dictionary
)
from fairseq.tasks import FairseqTask, register_task

from tasks import RelationInferenceTask

from datasets import (
    GraphDataset,
    MTBDataset,
    MTBTripletsDataset,
    AnnotatedTextDataset,
    SelectDictionaryDataset,
    filter_by_max_length,
    prune_dataset_size,
    ShuffledDataset,
)

from utils.data_utils import (
    CustomDictionary,
    EntityDictionary,
    load_annotated_text,
    safe_load_indexed_dataset,
)

logger = logging.getLogger(__name__)


@register_task('mtb')
class MTBTask(RelationInferenceTask):
    """Task for training inference models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""

        """Required either in config or cl"""
        parser.add_argument('--data-path', help='path to data')
        parser.add_argument('--strong_prob', default=0.8, type=float,
                            help='probability of sampling a strong negative')
        parser.add_argument('--n_tries_neighbor', default=100, type=int,
                            help='number of attempts to sample neighbors for a given case')
        parser.add_argument('--n_tries_text', default=100, type=int,
                            help='number of attempts to sample texts for a given case')
        parser.add_argument('--alpha', default=0.7, type=float,
                            help='probability of not masking the entity with a [BLANK] token')

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        train_text_data, train_annotation_data = load_annotated_text(
            self.args.data_path,
            'train',
            self.dictionary.bos(),
        )

        train_annotated_text_dataset = AnnotatedTextDataset(
            text_data=train_text_data,
            annotation_data=train_annotation_data,
            dictionary=self.dictionary,
            shift_annotations=1,
            mask_type=self.args.mask_type,
            assign_head_tail=None,
            seed=self.seed,
            alpha=self.args.alpha,
        )

        if split == 'train':
            split_annotated_text_dataset = train_annotated_text_dataset
        else:
            text_data, annotation_data = load_annotated_text(
                self.args.data_path,
                split,
                self.dictionary.bos(),
            )

            split_annotated_text_dataset = AnnotatedTextDataset(
                text_data=text_data,
                annotation_data=annotation_data,
                dictionary=self.dictionary,
                shift_annotations=1,
                mask_type=self.args.mask_type,
                assign_head_tail=None,
                seed=self.seed,
                alpha=self.args.alpha,
            )

        mtb_triplets_path = os.path.join(self.args.data_path, 'mtb_triplets_' + split + '.npy')
        mtb_triplets_loaded = np.load(mtb_triplets_path, mmap_mode='r')

        split_dataset = MTBTripletsDataset(split_annotated_text_dataset, mtb_triplets_loaded)

        split_dataset = self.filter_by_max_positions(split_dataset)

        # train_annotated_text_dataset, sentences_to_keep = filter_by_max_length(
        #     train_annotated_text_dataset,
        #     self.args.max_positions,
        # )
        # self.graph = self.create_graph(
        #     train_annotation_data,
        #     len(self.entity_dictionary),
        #     sentences_to_keep,
        # )

        dataset = MTBDataset(
            split_dataset=split_dataset,
            train_dataset=train_annotated_text_dataset,
            graph=self.graph,
            n_entities=len(self.entity_dictionary),
            dictionary=self.dictionary,
            strong_prob=self.args.strong_prob,
            n_tries_neighbor=self.args.n_tries_neighbor,
            n_tries_text=self.args.n_tries_text,
            max_positions=self.args.max_positions,
            seed=self.seed,
        )

        n_examples = int(getattr(self.args, 'n_' + split + '_examples', -1))
        dataset = prune_dataset_size(
            dataset,
            n_examples,
            self.seed,
        )
        self.datasets[split] = dataset
