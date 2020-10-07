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
        self.relations = list(self.relation_index.keys())
        self.filtered_relation_dataset = None

    def prune_by_num_relations(self, n_train_relations):
        if self.filtered_relation_dataset is not None:
            raise Exception('FewRel dataset has already been pruned!')
        assert n_train_relations >= 5 and n_train_relations <= 64
        with data_utils.numpy_seed(271829, self.seed):
            self.filtered_relation_dataset = []
            self.filtered_item_index = []
            self.relations = rd.choice(self.relations, size=n_train_relations, replace=False)
            for idx in range(len(self.relation_dataset)):
                cur_rel = self.relation_dataset[idx].item()
                if cur_rel in self.relations:
                    self.filtered_relation_dataset.append(cur_rel)
                    self.filtered_item_index.append(idx)

    def prune_by_num_examples_per_relation(self, n_train_examples_per_relation):
        if self.filtered_relation_dataset is not None:
            raise Exception('FewRel dataset has already been pruned!')
        assert n_train_examples_per_relation >= 5 and n_train_examples_per_relation <= 700
        with data_utils.numpy_seed(271830, self.seed):
            filtered_relation_index = {}
            for rel, rel_indices in self.relation_index.items():
                cur_filter_indices = rd.choice(len(rel_indices), size=n_train_examples_per_relation, replace=False)
                filtered_relation_index[rel] = np.array(rel_indices)[cur_filter_indices]
            self.relation_index = filtered_relation_index

            self.filtered_relation_dataset = []
            self.filtered_item_index = []
            for idx in range(len(self.relation_dataset)):
                cur_rel = self.relation_dataset[idx].item()
                if idx in self.relation_index[cur_rel]:
                    self.filtered_relation_dataset.append(cur_rel)
                    self.filtered_item_index.append(idx)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        with data_utils.numpy_seed(271828, self.seed, self.epoch, index):
            text_index = index if self.filtered_relation_dataset is None else self.filtered_item_index[index]
            target_item, target_annotation = self.annotation_text.annotate_sentence(text_index, head_entity=0, tail_entity=1)
            target_relation = self.relation_dataset[index].item() if self.filtered_relation_dataset is None else self.filtered_relation_dataset[index]
            relations = rd.choice(
                self.relations,
                size=self.n_way,
                replace=False,
            ).tolist()
            if target_relation in relations:
                relations.remove(target_relation)
            else:
                relations = relations[:self.n_way - 1]

            relations = [target_relation] + relations

            exemplars = []
            exemplars_annotation = []
            for rel in relations:
                rel_examplar_idxs = rd.choice(self.relation_index[rel], size=self.n_shot, replace=False)

                for idx in rel_examplar_idxs:
                    text, exemplar_annotation = self.annotation_text.annotate_sentence(idx, head_entity=0, tail_entity=1)
                    exemplars.append(text)
                    exemplars_annotation.append(exemplar_annotation)

            ntokens, nsentences = len(target_item), 1
            for exemplar in exemplars:
                nsentences += 1
                ntokens += len(exemplar)

        item = {
            'text': target_item,
            'annotation': target_annotation,
            'exemplars': exemplars,
            'exemplars_annotation': exemplars_annotation,
            'ntokens': ntokens,
            'nsentences': nsentences,
        }
        return item

    def __len__(self):
        if self.filtered_relation_dataset is None:
            return len(self.relation_dataset)
        else:
            return len(self.filtered_relation_dataset)

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @property
    def sizes(self):
        if self.filtered_relation_dataset is None:
            return self.annotation_text.sizes
        else:
            return self.annotation_text.sizes[self.filtered_item_index]

    def ordered_indices(self):
        return np.argsort([10 * (np.random.random(len(self.sizes)) - 0.5) + self.sizes])[0]

    def collater(self, instances):
        batch_size = len(instances)

        if batch_size == 0:
            return None

        text = []
        annotation = []
        exemplars = []
        exemplars_annotation = []
        ntokens, nsentences = 0, 0

        for instance in instances:
            text.append(instance['text'])
            annotation.append(instance['annotation'])
            exemplars += instance['exemplars']
            exemplars_annotation.append(instance['exemplars_annotation'])
            ntokens += len(instance['text']) + sum([len(s) for s in instance['exemplars']])
            nsentences += 1 + len(instance['exemplars'])

        padded_text = pad_sequence(text, batch_first=True, padding_value=self.dictionary.pad())
        padded_exemplars = pad_sequence(exemplars, batch_first=True, padding_value=self.dictionary.pad())

        if len(annotation) > 0 and annotation[0] is not None:
            annotation = torch.LongTensor(annotation)
            exemplars_annotation = torch.LongTensor(exemplars_annotation)
        else:
            annotation = None
            exemplars_annotation = None

        item = {
            'text': padded_text,
            'annotation': annotation,
            'exemplars': padded_exemplars,
            'exemplars_annotation': exemplars_annotation,
            'target': torch.zeros(len(instances), dtype=torch.long),
            'batch_size': len(instances),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'size': batch_size,
        }
        return item

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return False
