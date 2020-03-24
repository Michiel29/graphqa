import logging
import numpy as np
import os
from fairseq.tasks import register_task

from datasets import (
    AnnotatedText,
    GraphDataset,
    PrependTokenDataset,
    prune_dataset_size,
    TripletDataset,
)
from utils.data_utils import safe_load_indexed_dataset
from tasks import RelationInferenceTask

logger = logging.getLogger(__name__)


@register_task('triplet_inference')
class TripletInferenceTask(RelationInferenceTask):
    """Task for training inference models."""

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        text_data = safe_load_indexed_dataset(
            os.path.join(self.args.data_path, split + '.text'),
        )
        annotation_data = np.load(
            os.path.join(self.args.data_path, split + '.annotations.npy'),
            mmap_mode='r',
        )
        annotated_text = AnnotatedText(
            text_data=text_data,
            annotation_data=annotation_data,
            dictionary=self.dictionary,
            entity_dictionary=self.entity_dictionary,
            mask_type=self.args.mask_type,
            non_mask_rate=self.args.non_mask_rate,
        )

        graph_data = safe_load_indexed_dataset(
            os.path.join(self.args.data_path, split + '.graph'),
        )
        graph = GraphDataset(
            edges=graph_data,
            subsampling_strategy=self.args.subsampling_strategy,
            subsampling_cap=self.args.subsampling_cap,
            epoch_splits=self.args.epoch_splits,
            seed=self.args.seed,
        )

        dataset = TripletDataset(
            annotated_text=annotated_text,
            graph=graph,
            k_negative=self.args.k_negative,
            n_entities=len(self.entity_dictionary),
            seed=self.args.seed,
            dictionary=self.dictionary,
        )
        dataset = PrependTokenDataset(dataset, self.dictionary.bos(), 'text')

        n_examples = int(getattr(self.args, 'n_' + split + '_examples', -1))
        if n_examples != -1:
            dataset = prune_dataset_size(dataset, n_examples, self.seed)

        self.datasets[split] = dataset
