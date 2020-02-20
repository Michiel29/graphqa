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

        # Don't reload datasets that were already setup earlier
        if split in self.datasets:
            return

        text_data, annotation_data = self.load_annotated_text(split)

        self.datasets[split] = TripletDataset(
            text_data,
            annotation_data,
            self.args.k_negative,
            len(self.entity_dictionary),
            self.dictionary,
            shift_annotations=1, # because of the PrependTokenDataset
        )
