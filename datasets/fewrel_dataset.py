from collections import defaultdict
import numpy as np
import numpy.random as rd
import torch
from torch.nn.utils.rnn import pad_sequence

from fairseq.data import data_utils, FairseqDataset

from datasets import AnnotatedText


class FewRelDataset(FairseqDataset):

    def __init__(
        self,
        annotation_text,
        relation_dataset,
        dictionary,
        n_way,
        n_shot,
        seed,
    ):
        self.annotation_text = annotation_text
        self.relation_dataset = relation_dataset
        self.dictionary = dictionary
        self.n_way = n_way
        assert self.n_way > 1
        self.n_shot = n_shot
        assert self.n_way > 0
        self.seed = seed
        self.epoch = 0

        self.relation_index = defaultdict(list)
        for idx in range(len(self.relation_dataset)):
            self.relation_index[self.relation_dataset[idx].item()].append(idx)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        with data_utils.numpy_seed(271828, self.seed, self.epoch, index):
            target_item = self.annotation_text.annotate_sentence(index, head_entity=0, tail_entity=1)
            target_relation = self.relation_dataset[index]
            relations = rd.choice(
                list(self.relation_index.keys()),
                size=self.n_way,
                replace=False,
            ).tolist()
            if target_relation in relations:
                relations.remove(target_relation)
            else:
                relations = relations[:self.n_way - 1]

            relations = [target_relation.item()] + relations

            exemplars = []
            for rel in relations:
                rel_examplar_idxs = rd.choice(self.relation_index[rel], size=self.n_shot, replace=False)
                exemplars += [
                    self.annotation_text.annotate_sentence(idx, head_entity=0, tail_entity=1)
                    for idx in rel_examplar_idxs
                ]

            ntokens, nsentences = len(target_item), 1
            for exemplar in exemplars:
                nsentences += 1
                ntokens += len(exemplars)

        item = {
            'text': target_item,
            'exemplars': exemplars,
            'ntokens': ntokens,
            'nsentences': nsentences,
        }
        return item

    def __len__(self):
        return len(self.annotation_text)

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @property
    def sizes(self):
        return self.annotation_text.sizes

    def ordered_indices(self):
        return np.argsort([10 * (np.random.random(len(self.sizes)) - 0.5) + self.sizes])[0]

    def collater(self, instances):
        batch_size = len(instances)

        if batch_size == 0:
            return None

        text, annotation = [], []
        exemplars, exemplars_annotations = [], []
        ntokens, nsentences = 0, 0

        for instance in instances:
            text.append(instance['text'])
            if 'annotation' in instance:
                annotation.append(instance['annotation'])
            exemplars += instance['exemplars']
            if 'exemplars_annotations' in instance:
                exemplars_annotations += instance['exemplars_annotations']
            ntokens += len(instance['text']) + sum([len(s) for s in instance['exemplars']])
            nsentences += 1 + len(instance['exemplars'])

        padded_text = pad_sequence(text, batch_first=True, padding_value=self.dictionary.pad())
        padded_exemplars = pad_sequence(exemplars, batch_first=True, padding_value=self.dictionary.pad())

        item = {
            'text': padded_text,
            'exemplars': padded_exemplars,
            'target': torch.zeros(len(instances), dtype=torch.long),
            'batch_size': len(instances),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'size': batch_size,
        }

        if len(annotation) > 0:
            import pdb; pdb.set_trace()
            assert len(annotation) == len(text)
            padded_annotation = pad_sequence(annotation, batch_first=True, padding_value=self.dictionary.pad())
            item['annotation'] = padded_annotation
        if len(exemplars_annotations) > 0:
            assert len(exemplars_annotations) == len(exemplars)
            padded_exemplars_annotations = pad_sequence(exemplars_annotations, batch_first=True, padding_value=self.dictionary.pad())
            item['exemplars_annotations'] = padded_exemplars_annotations

        return item

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False
