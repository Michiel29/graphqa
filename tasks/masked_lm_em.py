import logging
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
    PrependTokenDataset,
)
from utils.data_utils import CustomDictionary, EntityDictionary
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.tasks import FairseqTask, register_task

from datasets import (
    AnnotatedTextDataset,
    FixedSizeDataset,
    SelectDictionaryDataset,
)

logger = logging.getLogger(__name__)


@register_task('masked_lm_em')
class MaskedLMEMTask(FairseqTask):
    """Task for masked language model (entity mention) models."""
    def __init__(self, args, dictionary, entity_dictionary):
        super().__init__(args)
        self.seed = args.seed
        self.dictionary = dictionary
        self.mask_idx = self.dictionary.index('<mask>')
        self.mask_type = args.mask_type

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

    def get_mask_seed(self, epoch):
        return self.seed + epoch

    # TODO(urikz): refactor this
    def load_annotated_text(self, split):
        text_path = os.path.join(self.args.data_path, split + '.text')
        annotation_path = os.path.join(self.args.data_path, split + '.annotations')

        text_data =  data_utils.load_indexed_dataset(
            text_path,
            None,
            dataset_impl='mmap',
        )

        if text_data is None:
            raise FileNotFoundError('Dataset (text) not found: {}'.format(text_path))

        annotation_data =  data_utils.load_indexed_dataset(
            annotation_path,
            None,
            dataset_impl='mmap',
        )

        if annotation_data is None:
            raise FileNotFoundError('Dataset (annotation) not found: {}'.format(annotation_path))

        text_data = PrependTokenDataset(text_data, self.dictionary.bos())

        n_examples = int(getattr(self.args, 'n_' + split + '_examples', -1))

        text_data = FixedSizeDataset(text_data, n_examples)
        annotation_data = FixedSizeDataset(annotation_data, n_examples)

        return text_data, annotation_data

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
                text_data=text_data,
                annotation_data=annotation_data,
                dictionary=self.dictionary,
                mask_type=self.mask_type,
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

    @classmethod
    def setup_task(cls, args, **kwargs):
        dict_path = os.path.join(args.data_path, 'dict.txt')
        dictionary = CustomDictionary.load(dict_path)

        entity_dict_path = os.path.join(args.data_path, 'entity.dict.txt')
        entity_dictionary = EntityDictionary.load(entity_dict_path)

        logger.info('dictionary: {} types'.format(len(dictionary)))
        logger.info('entity dictionary: {} types'.format(len(entity_dictionary)))

        task = cls(args, dictionary, entity_dictionary)
        task.load_dataset('train')

        return task

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary