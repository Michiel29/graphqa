import logging
import numpy as np
import os

from fairseq.data import (
    data_utils,
    Dictionary,
    IdDataset,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PadDataset,
)
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.tasks import register_task

from datasets import (
    AnnotatedTextDataset,
    SelectDictionaryDataset,
    filter_by_max_length,
    prune_dataset_size,
    ShuffledDataset,
)
from tasks import BaseTask
from utils.data_utils import (
    CustomDictionary,
    EntityDictionary,
    load_annotated_text,
    safe_load_indexed_dataset,
)


logger = logging.getLogger(__name__)


@register_task('masked_lm_em')
class MaskedLMEMTask(BaseTask):
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


    @classmethod
    def setup_task(cls, args, **kwargs):
        dict_path = os.path.join(args.data_path, 'dict.txt')
        dictionary = CustomDictionary.load(dict_path)

        logger.info('dictionary: {} types'.format(len(dictionary)))

        task = cls(args, dictionary, None)
        return task

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
            shift_annotations=1,
            mask_type=self.args.mask_type,
            assign_head_tail='random',
            seed=self.seed,
            alpha=self.args.alpha,
        )

        n_examples = int(getattr(self.args, 'n_' + split + '_examples', -1))
        dataset = SelectDictionaryDataset(
            prune_dataset_size(
                filter_by_max_length(
                    annotated_text_dataset,
                    self.args.max_positions,
                )[0],
                n_examples,
                self.seed,
            ),
            'text',
        )

        # create masked input and targets
        mask_whole_words = get_whole_word_mask(self.args, self.source_dictionary) \
            if self.args.mask_whole_words else None

        src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
            dataset,
            self.dictionary,
            pad_idx=self.dictionary.pad(),
            mask_idx=self.dictionary.mask(),
            seed=self.seed,
            mask_prob=self.args.mask_prob,
            leave_unmasked_prob=self.args.leave_unmasked_prob,
            random_token_prob=self.args.random_token_prob,
            freq_weighted_replacement=self.args.freq_weighted_replacement,
            mask_whole_words=mask_whole_words,
        )

        self.datasets[split] = ShuffledDataset(
                NestedDictionaryDataset(
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
            ),
            sizes=src_dataset.sizes,
        )