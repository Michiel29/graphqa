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
    replace_tokens = [cd.head_token_df, cd.tail_token_df, cd.blank_token_df,
                        cd.e1_start_token_df, cd.e1_end_token_df, 
                        cd.e2_start_token_df, cd.e2_end_token_df]

    def __init__(self, dictionary, entity_dictionary, task=None):
        self.bpe = get_encoder(self.encoder_json, self.vocab_bpe)
        self.dictionary = dictionary
        self.entity_dictionary = entity_dictionary
        self.task = task

    def decode_sentence(self, sentence):
        token_ids = self.dictionary.string(sentence).split()
        processed_ids = []
        for token in token_ids:
            if token in self.replace_tokens:
                processed_ids += self.bpe.encode(' ' + token)
            elif token not in self.filter_tokens:
                processed_ids.append(token)

        decoded_sentence = self.bpe.decode([int(x) for x in processed_ids])

        return decoded_sentence

    def inspect_item(self, text_id, head_id=None, tail_id=None):

        decoded_text = self.decode_sentence(text_id)    
        print('\n\nTEXT ID LIST:\n {}\n'.format(self.dictionary.string(text_id).split()))    
        print('DECODED TEXT:\n {}\n'.format(decoded_text))

        if head_id is not None:
            head_ent = self.entity_dictionary[head_id]
            print('<head> ENTITY:\n {} (ID={})\n'.format(head_ent, head_id))
        if tail_id is not None:
            tail_ent = self.entity_dictionary[tail_id]
            print('<tail> ENTITY:\n {} (ID={})\n'.format(tail_ent, tail_id))
        print('\n')

    def inspect_mtb_pairs(self, A_dict=None, B_dict=None):

        if A_dict is not None:
            print('\n')
            for key, val in A_dict.items():
                if key == 'textA':
                    decoded_text = self.decode_sentence(val)    
                    print('{} ID LIST:\n {}\n'.format(key, self.dictionary.string(val).split()))    
                    print('{} DECODED TEXT:\n {}\n'.format(key, decoded_text))
                else:
                    ent = self.entity_dictionary[val]
                    print('<{}> ENTITY:\n {} (ID={})\n'.format(key, ent, val))  
            print('\n')
            
        if B_dict is not None:
            for i in range(len(B_dict)):
                pair_type = 'POSITIVE' if i == 0 else 'NEGATIVE {}'.format(i) 
                decoded_text = self.decode_sentence(B_dict['textB'][i])    
                print('\n{} ID LIST ({}):\n {}\n'.format('textB', pair_type, self.dictionary.string(B_dict['textB'][i]).split()))    
                print('{} DECODED TEXT ({}):\n {}\n'.format('textB', pair_type, decoded_text))

                head_ent = self.entity_dictionary[B_dict['headB'][i]]
                print('<{}> ENTITY ({}):\n {} (ID={})\n'.format('headB', pair_type, head_ent, B_dict['headB'][i]))

                tail_ent = self.entity_dictionary[B_dict['tailB'][i]]
                print('<{}> ENTITY ({}):\n {} (ID={})\n\n'.format('tailB', pair_type, tail_ent, B_dict['tailB'][i]))

        print('------------------------------------------------------------')

    def inspect_batch(self, batch, ent_filter=None, scores=None):

        batch_size = batch['size']
        target = batch['target']

        if self.task.args.task == 'triplet_inference':
            text_id = batch['text']
            head_id = batch['head']
            tail_id = batch['tail']
        elif self.task.args.task == 'fewrel':
            text_id = batch['text']
            n_way = self.task.args.n_way
            n_shot = self.task.args.n_shot
            exemplars_id = batch['exemplars'].reshape(batch_size, n_way, n_shot, -1)
        elif self.task.args.task == 'mtb':
            textA_id = batch['textA']
            textB_id = []
            for cluster_id, cluster_texts in batch['textB'].items():
                textB_chunks = list(torch.chunk(cluster_texts, cluster_texts.shape[0], dim=0))
                textB_id += textB_chunks
            headA_id = batch['headA']
            tailA_id = batch['tailA']
            headB_id = batch['headB']
            tailB_id = batch['tailB']
            n_pairs = int(batch['A2B'].numel() / batch['size'])

        for i in range(batch_size):
            if self.task.args.task == 'triplet_inference':
                if ent_filter is None:
                    pass
                elif head_id[i,0] not in ent_filter or tail_id[i,0] not in ent_filter:
                    continue
                decoded_text = self.decode_sentence(text_id[i])
                pos_head_ent = self.task.entity_dictionary[head_id[i,0]]
                pos_tail_ent = self.task.entity_dictionary[tail_id[i,0]]
                neg_head_ent = [self.task.entity_dictionary[head_id[i,j]] for j in range(1, head_id.shape[1])]
                neg_tail_ent = [self.task.entity_dictionary[tail_id[i,j]] for j in range(1, tail_id.shape[1])]

                print('\n\nTEXT ID LIST:\n {}\n'.format(self.task.dictionary.string(text_id[i]).split()))
                print('DECODED TEXT:\n {}\n'.format(decoded_text))
                print('POSITIVE <head> ENTITY:\n {} (ID={})\n'.format(pos_head_ent, head_id[i,0].item()))
                print('POSITIVE <tail> ENTITY:\n {} (ID={})\n'.format(pos_tail_ent, tail_id[i,0].item()))
                print('NEGATIVE <head> ENTITIES:\n {} (ID={})\n'.format(neg_head_ent, head_id[i,1:].cpu().numpy()))
                print('NEGATIVE <tail> ENTITIES:\n {} (ID={})\n'.format(neg_tail_ent, tail_id[i,1:].cpu().numpy()))

                print('TARGET: \n {}\n'.format(target[i].cpu().detach().numpy()))
                if scores is not None:
                    print('SCORES: \n {}\n'.format(F.softmax(scores[i,:], dim=-1).cpu().detach().numpy()))
                else:
                    print('\n')

            elif self.task.args.task == 'fewrel':
                decoded_text = self.decode_sentence(text_id[i])
                decoded_exemplars = {}
                for j in range(n_way):
                    decoded_exemplars[j] = set()
                    for k in range(n_shot):
                        # cur_exemplar_id = self.task.dictionary.string(exemplars_id[i,j,k,:]).split()
                        decoded_exemplars[j].add(self.decode_sentence(exemplars_id[i,j,k,:]))

                print('\n\nTEXT ID LIST:\n {}\n'.format(self.task.dictionary.string(text_id[i]).split()))
                print('DECODED TEXT:\n {}\n'.format(decoded_text))
                print('DECODED EXEMPLARS (w/o ENTITIES):')
                for j in range(n_way):
                    print('Class {0}: {1}\n'.format(j, decoded_exemplars[j]))

                print('TARGET: \n {}\n'.format(target[i].cpu().detach().numpy()))
                if scores is not None:
                    print('SCORES: \n {}\n'.format(F.softmax(scores[i,:], dim=-1).cpu().detach().numpy()))
                else:
                    print('\n')

            elif self.task.args.task == 'mtb':
                if ent_filter is None:
                    pass
                elif headA_id[i] not in ent_filter or tailA_id[i] not in ent_filter or headB_neg_id[i] not in ent_filter or tailB_neg_id[i] not in ent_filter:
                    continue

                # Print textA, headA, and tailA
                cur_textA = textA_id[i]
                decoded_textA = self.decode_sentence(cur_textA) 
                print('\nTEXTA ID LIST:\n {}\n'.format(self.task.dictionary.string(cur_textA).split()))
                print('DECODED TEXTA:\n {}\n'.format(decoded_textA))

                headA_ent = self.task.entity_dictionary[headA_id[i]]
                print('<headA> ENTITY:\n {} (ID={})\n'.format(headA_ent, headA_id[i].item()))

                tailA_ent = self.task.entity_dictionary[tailA_id[i]]
                print('<tailA> ENTITY:\n {} (ID={})\n'.format(tailA_ent, tailA_id[i].item()))
        
                # Print textB, headB, and tailB
                for j in range(n_pairs):
                    cur_textB = textB_id[batch['A2B'][i * n_pairs + j]]
                    decoded_textB = self.decode_sentence(cur_textB)
                    pair_type = 'POSITIVE' if j == 0 else 'NEGATIVE {}'.format(j)  
                    print('\nTEXTB ID LIST ({}):\n {}\n'.format(pair_type, self.task.dictionary.string(cur_textB).split()))
                    print('DECODED TEXTB ({}):\n {}\n'.format(pair_type, decoded_textB))

                    headB_ent = self.task.entity_dictionary[headB_id[i, j]]
                    print('<headB> ENTITY ({}):\n {} (ID={})\n'.format(pair_type, headB_ent, headB_id[i, j].item()))

                    tailB_ent = self.task.entity_dictionary[tailB_id[i, j]]
                    print('<tailB> ENTITY ({}):\n {} (ID={})\n'.format(pair_type, tailB_ent, tailB_id[i, j].item()))             

                # Print targets and scores
                print('TARGETS: \n {}\n'.format(np.array([1] + (n_pairs-1) * [0])))
                if scores is not None:
                    print('SCORES: \n {}\n\n'.format(np.round(torch.sigmoid(scores.reshape(batch_size, n_pairs)[i]).detach().cpu().numpy(), decimals=5)))
                else:
                    print('\n') 
                print('------------------------------------------------------------\n')
            
