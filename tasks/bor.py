from fairseq.tasks import register_task
import logging
import os

import numpy as np

from datasets import (
    AnnotatedText,
    EpochSplitDataset,
    FixedSizeDataset,
    GraphDataset,
    PrependTokenDataset,
    BoRDataset,
)
from utils.data_utils import safe_load_indexed_dataset
from utils.numpy_utils import MMapNumpyArray
from tasks import RelationInferenceTask

logger = logging.getLogger(__name__)


@register_task('bor')
class BoRTask(RelationInferenceTask):
    """Task for training inference models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""

        """Required either in config or cl"""
        parser.add_argument('--k_weak_negs', type=float,
                            help='number of weak negatives per positive')
        parser.add_argument('--n_tries_entity', type=int,
                            help='number of attempts to sample entity candidates')
        parser.add_argument('--split_mode', default=False,
                            help='whether train and validation sets have disjoint entities')
        parser.add_argument('--use_strong_negs', default=True,
                            help='whether to use strong negatives')
        parser.add_argument('--replace_tail', default=False,
                            help='whether to always replace tail when sampling strong negatives')
        parser.add_argument('--mutual-neighbors', default=False,
                            help='whether the sampled candidate entity must be a mutual neighbor of keep_entity and replace_entity')

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

        if self.args.data_path == '../data/nki/bin-v5-threshold20':
            similar_entities = np.load(os.path.join(self.args.data_path, 'entity.train.1133_25.candidates.idx.npy'))
            similarity_scores = np.load(os.path.join(self.args.data_path, 'entity.train.1133_25.candidates.score.npy'))
        elif self.args.data_path == '../data/nki/bin-v6':
            similar_entities = np.load(os.path.join(self.args.data_path, 'entity.train.1109_22.candidates.idx.npy'))
            similarity_scores = np.load(os.path.join(self.args.data_path, 'entity.train.1109_22.candidates.score.npy'))
        else:
            raise Exception("Top 1000 similar entities/scores data not available for the given dataset.")


        dataset = BoRDataset(
            split=split,
            annotated_text_A=annotated_text_A,
            annotated_text_B=annotated_text_B,
            graph_A=graph_A,
            graph_B=graph_B,
            similar_entities=similar_entities,
            similarity_scores=similarity_scores,
            seed=self.args.seed,
            dictionary=self.dictionary,
            k_weak_negs=self.args.k_weak_negs,
            n_tries_entity=self.args.n_tries_entity,
            use_strong_negs=self.args.use_strong_negs,
            replace_tail=self.args.replace_tail,
            mutual_neighbors=self.args.mutual_neighbors,
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