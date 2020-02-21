import logging

from fairseq.data import (
    IdDataset,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PadDataset,
)
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.tasks import register_task

from datasets import AnnotatedTextDataset, SelectDictionaryDataset
from tasks import RelationInferenceTask


logger = logging.getLogger(__name__)


@register_task('masked_lm_em')
class MaskedLMEMTask(RelationInferenceTask):
    """Task for masked language model (entity mention) models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('--data-path', type=str, help='path to data')
        parser.add_argument('--mask-prob', default=0.15, type=float,
                            help='probability of replacing a token with mask')
        parser.add_argument('--leave-unmasked-prob', default=0.1, type=float,
                            help='probability that a masked token is unmasked')
        parser.add_argument('--random-token-prob', default=0.1, type=float,
                            help='probability of replacing a token with a random token')
        parser.add_argument('--freq-weighted-replacement', default=False, action='store_true',
                            help='sample random replacement words based on word frequencies')
        parser.add_argument('--mask-whole-words', default=True, action='store_true',
                            help='mask whole words; you may also want to set --bpe')

    def __init__(self, args, dictionary, entity_dictionary):
        super().__init__(args, dictionary, entity_dictionary)
        self.mask_idx = self.dictionary.index('<mask>')

    def get_mask_seed(self, epoch):
        return self.seed + epoch

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        # Don't reload datasets that were already setup earlier
        if split in self.datasets:
            return

        text_data, annotation_data = self.load_annotated_text(split)
        dataset = SelectDictionaryDataset(
            AnnotatedTextDataset(
                text_data,
                annotation_data,
                self.dictionary,
                shift_annotations=1, # because of the PrependTokenDataset
                assign_head_tail_randomly=True,
            ),
            'mention',
        )

        # create masked input and targets
        mask_whole_words = get_whole_word_mask(self.args, self.source_dictionary) \
            if self.args.mask_whole_words else None

        src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
            dataset,
            self.dictionary,
            pad_idx=self.dictionary.pad(),
            mask_idx=self.mask_idx,
            seed=self.get_mask_seed(epoch),
            mask_prob=self.args.mask_prob,
            leave_unmasked_prob=self.args.leave_unmasked_prob,
            random_token_prob=self.args.random_token_prob,
            freq_weighted_replacement=self.args.freq_weighted_replacement,
            mask_whole_words=mask_whole_words,
        )

        self.datasets[split] = NestedDictionaryDataset(
            {
                'id': IdDataset(),
                'net_input': {
                    'src_tokens': PadDataset(
                        src_dataset,
                        pad_idx=self.source_dictionary.pad(),
                        left_pad=False,
                    ),
                    'src_lengths': NumelDataset(src_dataset, reduce=False),
                },
                'target': PadDataset(
                    tgt_dataset,
                    pad_idx=self.source_dictionary.pad(),
                    left_pad=False,
                ),
                'nsentences': NumSamplesDataset(),
                'ntokens': NumelDataset(src_dataset, reduce=True),
            },
            sizes=[src_dataset.sizes],
        )

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary