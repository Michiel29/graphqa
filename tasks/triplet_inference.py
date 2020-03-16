import logging

from fairseq.tasks import register_task

from datasets import (
    AnnotatedTextDataset,
    filter_by_max_length,
    prune_dataset_size,
    TripletDataset,
)
from tasks import RelationInferenceTask
from utils.data_utils import load_annotated_text

logger = logging.getLogger(__name__)


@register_task('triplet_inference')
class TripletInferenceTask(RelationInferenceTask):
    """Task for training inference models."""

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        text_data, annotation_data = load_annotated_text(
            self.args.data_path,
            split,
            self.dictionary.bos(),
        )
        annotated_text_dataset = AnnotatedTextDataset(
            text_data=text_data,
            annotation_data=annotation_data,
            dictionary=self.dictionary,
            entity_dictionary=self.entity_dictionary,
            shift_annotations=1,
            mask_type=self.args.mask_type,
            assign_head_tail='random',
            seed=self.args.seed,
            alpha=self.args.alpha,
        )
        dataset = TripletDataset(
            annotated_text_dataset=annotated_text_dataset,
            dictionary=self.dictionary,
            graph=self.graph,
            k_negative=self.args.k_negative,
            n_entities=len(self.entity_dictionary),
            subsampling_strategy=self.args.subsampling_strategy,
            subsampling_cap=self.args.subsampling_cap,
            max_positions=self.args.max_positions - 4,
            seed=self.args.seed,
        )
        n_examples = int(getattr(self.args, 'n_' + split + '_examples', -1))
        if self.args.subsampling_strategy is None:
            dataset = self.filter_by_max_positions(dataset)
        dataset = prune_dataset_size(dataset, n_examples, self.seed)
        self.datasets[split] = dataset
