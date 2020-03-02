import logging

from fairseq.tasks import register_task

from tasks import RelationInferenceTask
from datasets import TripletDataset


logger = logging.getLogger(__name__)


@register_task('triplet_inference')
class TripletInferenceTask(RelationInferenceTask):
    """Task for training inference models."""

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        text_data, annotation_data = self.load_annotated_text(split)

        self.datasets[split] = TripletDataset(
            text_data,
            annotation_data,
            self.graph,
            self.args.k_negative,
            len(self.entity_dictionary),
            self.dictionary,
            shift_annotations=1, # because of the PrependTokenDataset
            mask_type=self.mask_type,
        )