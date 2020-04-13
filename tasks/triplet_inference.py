from fairseq.tasks import register_task
import logging
import os

from datasets import (
    AnnotatedText,
    EpochSplitDataset,
    FixedSizeDataset,
    GraphDataset,
    PrependTokenDataset,
    TripletDataset,
)
from utils.data_utils import safe_load_indexed_dataset
from utils.numpy_utils import MMapNumpyArray
from tasks import RelationInferenceTask

logger = logging.getLogger(__name__)


@register_task('triplet_inference')
class TripletInferenceTask(RelationInferenceTask):
    """Task for training inference models."""

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        text_data = safe_load_indexed_dataset(
            os.path.join(self.args.data_path, split + '.text'),
        )
        annotation_data = MMapNumpyArray(
            os.path.join(self.args.data_path, split + '.annotations.npy')
        )
        annotated_text = AnnotatedText(
            text_data=text_data,
            annotation_data=annotation_data,
            dictionary=self.dictionary,
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
            seed=self.args.seed,
        )

        dataset = TripletDataset(
            annotated_text=annotated_text,
            graph=graph,
            k_negative=self.args.k_negative,
            n_entities=len(self.entity_dictionary),
            seed=self.args.seed,
            dictionary=self.dictionary,
            same_replace_heads_for_all_negatives=self.args.arch.startswith('encoder_dual'),
            negative_split_probs=self.args.negative_split_probs or [1, 0, 0],
        )
        if split == 'train' and self.args.epoch_size is not None:
            dataset = EpochSplitDataset(
                dataset=dataset,
                epoch_size=self.args.epoch_size,
                seed=self.args.seed,
            )

        dataset = PrependTokenDataset(dataset, self.dictionary.bos(), 'text')

        n_examples = getattr(self.args, 'n_' + split + '_examples', None)
        if n_examples is not None:
            dataset = FixedSizeDataset(
                dataset=dataset,
                size=n_examples,
                seed=self.seed,
            )

        self.datasets[split] = dataset
