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
        parser.add_argument('--case0_prob', default=0.5, type=float,
                            help='probability of sampling a pair of sentences which share both head and tail entities')
        parser.add_argument('--case1_prob', default=0.35, type=float,
                            help='probability of sampling a pair of sentences which share only one entity')
        parser.add_argument('--n_tries', default=10, type=int,
                            help='number of attempts to sample mentions for a given case')
        parser.add_argument('--alpha', default=0.7, type=float,
                            help='probability of not masking the entity with a [BLANK] token')

    def create_graph(self, annotation_data, n_entities, indices_to_keep):
        entity_neighbors = [list() for entity in range(n_entities)]
        entity_edges = [list() for entity in range(n_entities)]

        for sentence_idx in indices_to_keep:
            entity_ids = np.unique(annotation_data[sentence_idx].reshape(3, -1)[0])

            for a, b in combinations(entity_ids, 2):
                entity_neighbors[a].append(b)
                entity_neighbors[b].append(a)

                entity_edges[a].append(sentence_idx)
                entity_edges[b].append(sentence_idx)

        return GraphDataset(entity_neighbors, entity_edges)

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

        split_dataset = filter_by_max_length(
            split_dataset,
            self.args.max_positions,
        )[0]
        train_annotated_text_dataset, sentences_to_keep = filter_by_max_length(
            train_annotated_text_dataset,
            self.args.max_positions,
        )
        self.graph = self.create_graph(
            train_annotation_data,
            len(self.entity_dictionary),
            sentences_to_keep,
        )

        dataset = MTBDataset(
            split_dataset=split_dataset,
            train_dataset=train_annotated_text_dataset,
            graph=self.graph,
            n_entities=len(self.entity_dictionary),
            dictionary=self.dictionary,
            case0_prob=self.args.case0_prob,
            case1_prob=self.args.case1_prob,
            n_tries=self.args.n_tries,
            seed=self.seed,
        )

        n_examples = int(getattr(self.args, 'n_' + split + '_examples', -1))
        dataset = prune_dataset_size(
            dataset,
            n_examples,
            self.seed,
        )
        self.datasets[split] = dataset