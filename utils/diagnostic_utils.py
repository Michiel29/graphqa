import numpy as np
np.set_printoptions(suppress=True)

import torch
import torch.nn.functional as F

from fairseq.data.encoders.gpt2_bpe import get_encoder
from fairseq import file_utils

from .data_utils import CustomDictionary as cd


class Diagnostic():
    DEFAULT_ENCODER_JSON = 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
    DEFAULT_VOCAB_BPE = 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'


    encoder_json = file_utils.cached_path(
        getattr(None, 'gpt2_encoder_json', DEFAULT_ENCODER_JSON)
    )
    vocab_bpe = file_utils.cached_path(
        getattr(None, 'gpt2_vocab_bpe', DEFAULT_VOCAB_BPE)
    )

    filter_tokens = [cd.pad_df]
    replace_tokens = [cd.head_token_df, cd.tail_token_df]

    def __init__(self, task):
        self.bpe = get_encoder(self.encoder_json, self.vocab_bpe)
        self.task = task

    def decode_sentence(self, sentence):
        token_ids = self.task.dictionary.string(sentence).split()
        processed_ids = []
        for token in token_ids:
            if token in self.replace_tokens:
                processed_ids += self.bpe.encode(' ' + token)
            elif token not in self.filter_tokens:
                processed_ids.append(token)

        decoded_sentence = self.bpe.decode([int(x) for x in processed_ids])

        return decoded_sentence

    def inspect_batch(self, batch, ent_filter=None, scores=None):

        batch_size = batch['nsentences']
        mention_id = batch['mention']
        target = batch['target']

        if self.task.args.task == 'triplet_inference':
            head_id = batch['head']
            tail_id = batch['tail']
        elif self.task.args.task == 'fewrel':
            n_way = self.task.args.n_way
            n_shot = self.task.args.n_shot
            exemplars_id = batch['exemplars'].reshape(batch_size, n_way, n_shot, -1)

        for i in range(batch_size):

            # cur_mention_id = task.dictionary.string(mention_id[i]).split()
            # decoded_mention = self.decode_sentence(cur_mention_id)

            decoded_mention = self.decode_sentence(mention_id[i])

            if self.task.args.task == 'triplet_inference':
                if head_id[i,0] not in ent_filter or tail_id[i,0] not in ent_filter:
                    continue
                pos_head_ent = self.task.entity_dictionary[head_id[i,0]]
                pos_tail_ent = self.task.entity_dictionary[tail_id[i,0]]
                neg_head_ent = [self.task.entity_dictionary[head_id[i,j]] for j in range(1, head_id.shape[1])]
                neg_tail_ent = [self.task.entity_dictionary[tail_id[i,j]] for j in range(1, tail_id.shape[1])]

            elif self.task.args.task == 'fewrel':
                decoded_exemplars = {}
                for j in range(n_way):
                    decoded_exemplars[j] = set()
                    for k in range(n_shot):
                        # cur_exemplar_id = self.task.dictionary.string(exemplars_id[i,j,k,:]).split()
                        decoded_exemplars[j].add(self.decode_sentence(exemplars_id[i,j,k,:]))

            print('\n\nMENTION ID LIST:\n {}\n'.format(self.task.dictionary.string(mention_id[i]).split()))
            print('DECODED MENTION:\n {}\n'.format(decoded_mention))
            if self.task.args.task == 'triplet_inference':
                print('POSITIVE <head> ENTITY:\n {} (ID={})\n'.format(pos_head_ent, head_id[i,0].item()))
                print('POSITIVE <tail> ENTITY:\n {} (ID={})\n'.format(pos_tail_ent, tail_id[i,0].item()))
                print('NEGATIVE <head> ENTITIES:\n {} (ID={})\n'.format(neg_head_ent, head_id[i,1:].cpu().numpy()))
                print('NEGATIVE <tail> ENTITIES:\n {} (ID={})\n'.format(neg_tail_ent, tail_id[i,1:].cpu().numpy()))
            elif self.task.args.task == 'fewrel':
                print('DECODED EXEMPLARS (w/o ENTITIES):')
                for j in range(n_way):
                    print('Class {0}: {1}\n'.format(j, decoded_exemplars[j]))
            print('TARGET: \n {}\n'.format(target[i].cpu().detach().numpy()))
            if scores is not None:
                print('SCORES: \n {}\n'.format(F.softmax(scores[i,:], dim=-1).cpu().detach().numpy()))
            else:
                print('\n')

