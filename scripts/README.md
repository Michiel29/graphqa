# Wikipedia Corpus

### Latest dataset location: ravenclaw.usc.edu:/data/urikz/nki/bin-v5-threshold20 and ravenclaw.usc.edu:/data/urikz/nki/bin-v5-threshold20-small

1. Create output directory
```console
$ mkdir /data/urikz/nki/bin-v5-threshold20/
```
2. Build dictionary of all entities
```console
$ python scripts/prepare_wiki.py  --roberta /data/urikz/nki/roberta --tmp /tmp --entity-vocab /data/urikz/nki/bin-v5-threshold20/entity.dict.txt --limit-set-of-entities --append-eos --nworkers 60 --data "/data/urikz/nki/wiki/annotated_el_unquote_expand2_v2/*/wiki*" --entity-count-threshold 20 --min-entities-per-sentence 0
-- Found 42562 files
-- Build entity vocab mode: ON
Processing Wiki: 100%|█| 42562/42562 [02:32<00:00 , s=1.21e+8, p=2.87e+8, ann=1.23e+8, f_ed=4.1e+7, f_h_overlap=3.03e+7, f_self_overlap=112435, f_cross_s_bd=1534931, f_solo_s=0, f_xao=134301, f_vocab=0]
-- Successfully saved 650625 entities (out of 7168865) to /data/urikz/nki/bin-v5-threshold20/entity.dict.txt
```

3. Construct validation dataset
```console
$ python scripts/prepare_wiki.py --roberta /data/urikz/nki/roberta --tmp /tmp --output /data/urikz/nki/bin-v5-threshold20/valid --entity-vocab /data/urikz/nki/bin-v5-threshold20/entity.dict.txt --limit-set-of-entities --append-eos --nworkers 60 --data "/data/urikz/nki/wiki/annotated_el_unquote_expand2_v2/*/wiki99*"  --min-entities-per-sentence 0
-- Found 425 files
-- Build entity vocab mode: OFF
Processing Wiki: 100%|█| 425/425 [00:22<00:00 , s=1263488, d=55548, ann=1014396, f_ed=407930, f_h_overlap=307946, f_self_overlap=1159, f_cross_s_bd=8255, f_solo_s=0, f_xao=1057, f_vocab=216704]

$ python scripts/prepare_graph.py --data-path /data/urikz/nki/bin-v5-threshold20 --document-sep-len 1 --max-entity-pair-distance 40 --prefix valid --max-positions 123
-- 650629 entities
Collecting entity pairs: 100%|█|  1014396/1014396 [00:18<00:00, 54514.03it/s, entities_queue_sz=3, num_documents=55058, num_sentences=1257078, num_undir_edges=886562, num_undir_edges_same_s=886562]
-- num documents 55547, num sentences 1263481, num undirected edges 889842 (within same sentence 889842)
Building graph dataset: 100%|█|  650629/650629 [00:06<00:00, 94026.92it/s]
```

4. Construct training dataset
```console
$ python scripts/prepare_wiki.py --roberta /data/urikz/nki/roberta --tmp /tmp --output /data/urikz/nki/bin-v5-threshold20/train --entity-vocab /data/urikz/nki/bin-v5-threshold20/entity.dict.txt --limit-set-of-entities --append-eos --nworkers 60 --min-entities-per-sentence 0 --data "/data/urikz/nki/wiki/annotated_el_unquote_expand2_v2/*/wiki[0-8]*,/data/urikz/nki/wiki/annotated_el_unquote_expand2_v2/*/wiki9[0-8]*"
-- Found 42137 files
-- Build entity vocab mode: OFF
Processing Wiki: 100%|█| 42137/42137 [23:26<00:00 , s=1.25e+8, d=5546646, ann=1e+8, f_ed=4.06e+7, f_h_overlap=3e+7, f_self_overlap=111276, f_cross_s_bd=767234, f_solo_s=0, f_xao=133244, f_vocab=2.15e+7]

$ python scripts/prepare_graph.py --data-path /data/urikz/nki/bin-v5-threshold20 --document-sep-len 1 --max-entity-pair-distance 40 --prefix train --max-positions 123
-- 650629 entities
Collecting entity pairs: 100%|█| 100373302/100373302 [1:03:00<00:00, 26548.91it/s, entities_queue_sz=1, num_documents=5546256, num_sentences=1.25e+8, num_undir_edges=8.86e+7, num_undir_edges_same_s=8.86e+7]
-- num documents 5546645, num sentences 124895749, num undirected edges 88651776 (within same sentence 88651776)
Building graph dataset: 100%|█| 650629/650629 [05:04<00:00, 2137.25it/s]
```
# FewRel Dataset

[FewRel v1 Official Website](https://thunlp.github.io/1/fewrel1.html)

### Latest dataset location: ravenclaw.usc.edu:/data/urikz/fewrel/bin-v2

1. Download training and validation data
```console
$ mkdir /data/urikz/fewrel/bin-v2/train
$ wget https://github.com/thunlp/FewRel/raw/master/data/train_wiki.json
$ wget https://github.com/thunlp/FewRel/raw/master/data/val_wiki.json
```
2. Preprocess the data
```console
$ python scripts/prepare_fewrel.py --data /data/urikz/fewrel/train_wiki.json --output /data/urikz/fewrel/bin-v2/train --append-eos
-- Loaded data from /data/urikz/fewrel/train_wiki.json
Processing Wiki: 100%|█| 44800/44800 [00:23<00:00 ]

$ python scripts/prepare_fewrel.py --data /data/urikz/fewrel/val_wiki.json --output /data/urikz/fewrel/bin-v2/valid --append-eos
-- Loaded data from /data/urikz/fewrel/val_wiki.json
Processing Wiki: 100%|█| 11200/11200 [00:06<00:00 ]
```

# KBP37 Dataset

[KBP37 Paper](https://arxiv.org/abs/1508.01006)

### Latest dataset location: waldstein.usc.edu:/data1/aarchan/self_inference/data/kbp37/bin

1. Download train/dev/test data
```console
$ cd path/to/data
$ mkdir kbp37 && cd kbp37
$ wget https://github.com/zhangdongxu/kbp37/blob/master/train.txt
$ wget https://github.com/zhangdongxu/kbp37/blob/master/dev.txt
$ wget https://github.com/zhangdongxu/kbp37/blob/master/test.txt
```

2. Preprocess the data
```console
$ cd path/to/self_inference

$ python scripts/prepare_kbp37.py --split train --root-dir ../data/kbp37 --roberta-dir ../data/roberta --append-eos
-- Loaded data from ../data/kbp37/train.txt
Processing Wiki: 100%|█| 15917/15917 [00:18<00:00 ]

$ python scripts/prepare_kbp37.py --split dev --root-dir ../data/kbp37 --roberta-dir ../data/roberta --append-eos
-- Loaded data from ../data/kbp37/dev.txt
Processing Wiki: 100%|█| 1724/1724 [00:02<00:00 ]

$ python scripts/prepare_kbp37.py --split test --root-dir ../data/kbp37 --roberta-dir ../data/roberta --append-eos
-- Loaded data from ../data/kbp37/test.txt
Processing Wiki: 100%|█| 3405/3405 [00:05<00:00 ]
```

3. Copy dict.txt into data/kbp37/bin
```console
$ cd path/to/self_inference
$ cp ../data/roberta/roberta.base/dict.txt ../data/kbp37/bin
```

# SemEval 2010 Task 8 Dataset

[SemEval 2010 Task 8 Paper](https://arxiv.org/abs/1911.10422)
[SemEval 2010 Task 8 Website 1](http://www.kozareva.com/downloads.html)
[SemEval 2010 Task 8 Website 2](http://semeval2.fbk.eu/semeval2.php?location=tasks)

### Latest dataset location: waldstein.usc.edu:/data1/aarchan/self_inference/data/SemEval2010_task8_all_data/bin

1. Download and extract data
```console
$ cd path/to/data
$ wget https://github.com/davidsbatista/Annotated-Semantic-Relationships-Datasets/raw/master/datasets/SemEval2010_task8_all_data.tar.gz
$ tar xvzf SemEval2010_task8_all_data.tar.gz
```

2. Preprocess the data
```console
$ cd path/to/self_inference

$ python scripts/prepare_semeval2010task8.py --split train --root-dir ../data/SemEval2010_task8_all_data --roberta-dir ../data/roberta --append-eos
-- Loaded data from ../data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT
Processing Wiki: 100%|█| 6500/6500 [00:05<00:00 ]

$ python scripts/prepare_semeval2010task8.py --split dev --root-dir ../data/SemEval2010_task8_all_data --roberta-dir ../data/roberta --append-eos
-- Loaded data from ../data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT
Processing Wiki: 100%|█| 1500/1500 [00:02<00:00 ]

$ python scripts/prepare_semeval2010task8.py --split test --root-dir ../data/SemEval2010_task8_all_data --roberta-dir ../data/roberta --append-eos
-- Loaded data from ../data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT
Processing Wiki: 100%|█| 2717/2717 [00:02<00:00 ]
```

3. Copy dict.txt into data/SemEval2010_task8_all_data/bin
```console
$ cd path/to/self_inference
$ cp ../data/roberta/roberta.base/dict.txt ../data/SemEval2010_task8_all_data/bin
```

# MTB Triplet Datasets

## What is an MTB triplet?
An MTB triplet is a directed triplet of the form (sentence_id, entity_1, entity_2), which has been verified to satisfy case0 (i.e, there exists at least one other sentence containing entity_1 and entity_2) and case1 (i.e., there exists at least one other sentence containing entity_1 but not entity_2).

Thus, an MTB triplet can be used as the first sentence in a positive sentence pair (case0) or a strong negative sentence pair (case1), with respect to entity_1 and entity_2. Recall that weak negative pairs are those in which the two constituent sentences share no entities (case2).

## How to use the MTB triplet dataset
First, we assume here that the dataset is `bin-v3-threshold20`, but we also generate MTB triplets for `bin-v3-threshold20-small`.

__TODO: Compress Steps 1-2 into a single script and fewer data files__

#### 1. Create the helper data files
Run the following scripts to create the helper data files:
```
python scripts/prepare_unique_graph.py --data-path ../data/nki/bin-v3-threshold20
python scripts/prepare_unique_entities.py --data-path ../data/nki/bin-v3-threshold20
```
This will save the following data files to your data directory:
- `unique_neighbors.bin`, `unique_neighbors.idx`, `unique_neighbors_len.npy`
- `unique_edges.bin`, `unique_edges.idx`, `unique_edges_len.npy`
- `unique_entities.bin`, `unique_entities.idx`

#### 2. Build the MTB triplets array
Run the following script to build the MTB triplets array:
```
python scripts/prepare_mtb_triplets.py --data-path ../data/nki/bin-v3-threshold20
```
This will save the following data files to your data directory:
- `ent_pair_counts.pkl` (pre-processing step for case0)
- `mtb_triplet_train.npy`, `mtb_triplet_valid.npy`

The `mtb_triplet_train.npy` and `mtb_triplet_valid.npy` files each contain a numpy array of all MTB triplets in the train and validation sets, respectively.

#### 3. Run the MTB dataloader
For MTB training/validation, the dataloader will create an `MTBTripletsDataset` class, which samples triplets from the corresponding MTB triplet array.

The the `__getitem__` method in `MTBDataset` is responsible for sampling sentence pairs satisfying either case0, case1, or case2. To sample a sentence pair, the `__getitem__` method will first sample a triplet via `MTBTripletsDataset`. This triplet corresponds to the first sentence of the sentence pair.

Then, a case is randomly selected, based on predetermined probabilities for case0, case1, and case2.

After that, the `__getitem__` method will use the sentence and entity IDs from that triplet to sample a second sentence satisfying the selected case.

## bin-v3-threshold20
#### train
- __data path__: `waldstein.usc.edu:/data1/aarchan/self_inference/data/nki/bin-v3-threshold20/mtb_triplets_train.npy`
- __size of data__: 2.9GB
- __number of triplets__: 125,755,179
- __number of sentences__: 23,520,420
- __triplet dataset generation time__: 5:38:55 (05:12 for ent_pair_dict; 5:26:43 for mtb_triplets)
- __epoch runtime__:
#### valid
- __data path__: `waldstein.usc.edu:/data1/aarchan/self_inference/data/nki/bin-v3-threshold20/mtb_triplets_valid.npy`
- __size of data__: 25MB
- __number of triplets__: 1,065,258
- __number of sentences__: 237,147
- __triplet dataset generation time__: 07:58 (05:12 for ent_pair_dict; 02:56 for mtb_triplets)
- __epoch runtime__:

## bin-v3-threshold20-small
#### train
- __data path__: `waldstein.usc.edu:/data1/aarchan/self_inference/data/nki/bin-v3-threshold20-small/mtb_triplets_train.npy`
- __size of data__: 173MB
- __number of triplets__: 7,528,652
- __number of sentences__: 2,138,107
- __triplet dataset generation time__: 05:37 (00:26 for ent_pair_dict; 05:11 for mtb_triplets)
- __epoch runtime__:
#### valid
- __data path__: `waldstein.usc.edu:/data1/aarchan/self_inference/data/nki/bin-v3-threshold20-small/mtb_triplets_valid.npy`
- __size of data__: 13MB
- __number of triplets__: 556,327
- __number of sentences__: 237,147
- __triplet dataset generation time__: 00:54 (00:26 for ent_pair_dict; 00:28 for mtb_triplets)
- __epoch runtime__:

## Note
Epoch runtime estimates computed using the following settings:
- __machine__: waldstein
- __encoder__: RoBERTa-small
- __max_sentences__: 300
- __max_tokens__: 1.5e4
- __update_freq__: 1
- __num gpus__: 3 Titan Xp
- __num workers__: 1
