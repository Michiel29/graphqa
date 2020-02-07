from pudb import set_trace

import numpy as np
np.set_printoptions(suppress=True)

import torch
import torch.nn.functional as F

from fairseq.data.encoders.gpt2_bpe import get_encoder
from fairseq import file_utils

DEFAULT_ENCODER_JSON = 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
DEFAULT_VOCAB_BPE = 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'

encoder_json = file_utils.cached_path(
    getattr(None, 'gpt2_encoder_json', DEFAULT_ENCODER_JSON)
)
vocab_bpe = file_utils.cached_path(
    getattr(None, 'gpt2_vocab_bpe', DEFAULT_VOCAB_BPE)
)
bpe = get_encoder(encoder_json, vocab_bpe)

def inspect_batch(batch, task, score=None):

    batch_size = batch['batch_size']
    mention_id = batch['mention']
    head_id = batch['head']
    tail_id = batch['tail']

    decoded_mentions = []

    for i in range(batch_size):

        cur_mention_id = task.dictionary.string(mention_id[i]).split()  

        decode = bpe.decode([int(x) for x in cur_mention_id if (x != '<head>' and x != '<tail>' and x != '<pad>')])

        pos_head_ent = task.entity_dictionary[head_id[i,0]]
        pos_tail_ent = task.entity_dictionary[tail_id[i,0]]

        neg_head_ent = [task.entity_dictionary[head_id[i,j]] for j in range(1, head_id.shape[1])] 
        neg_tail_ent = [task.entity_dictionary[tail_id[i,j]] for j in range(1, tail_id.shape[1])] 

        print('\n\nMENTION ID LIST:\n {}\n'.format(cur_mention_id))
        print('POSITIVE <head> ENTITY:\n {} (ID={})\n'.format(pos_head_ent, head_id[i,0].item()))
        print('POSITIVE <tail> ENTITY:\n {} (ID={})\n'.format(pos_tail_ent, tail_id[i,0].item()))
        print('NEGATIVE <head> ENTITIES:\n {} (ID={})\n'.format(neg_head_ent, head_id[i,1:].cpu().numpy()))
        print('NEGATIVE <tail> ENTITIES:\n {} (ID={})\n'.format(neg_tail_ent, tail_id[i,1:].cpu().numpy()))
        print('DECODED MENTION (w/o ENTITIES):\n {}\n'.format(decode))

        if score is not None:
            print('SCORE: \n {}\n\n'.format(F.softmax(score[i,:], dim=-1).cpu().detach().numpy()))
        else:
            print('\n')

