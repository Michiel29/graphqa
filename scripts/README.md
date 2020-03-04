# Wikipedia Corpus

### Latest dataset location: fantasia.usc.edu:/data2/urikz/nki/bin-v2

1. Create output directory
```console
$ mkdir /data/urikz/nki/bin-v2-threshold20/
```
2. Build dictionary of all entities
```console
$ python scripts/prepare_wiki.py --roberta /data/urikz/nki/roberta --tmp /tmp --entity-vocab /data/urikz/nki/bin-v3-threshold20/entity.dict.txt --limit-set-of-entities --append-eos --nworkers 60 --data "/data/urikz/nki/wiki/annotated_el_unquote_expand2_v2/\*/wiki\*" --entity-count-threshold 20
-- Found 42562 files
-- Build entity vocab mode: ON
Processing Wiki: 100%|█| 42562/42562 [02:15<00:00 , s=3.04e+7, ann=9.29e+7, f_ed=4.1e+7, f_h_overlap=3.03e+7, f_self_overlap=112435, f_cross_s_bd=32841
-- Successfully saved 514978 entities (out of 6220659) to /data/urikz/nki/bin-v3-threshold20/entity.dict.txt
```
3. Construct validation dataset
```console
$ python scripts/prepare_wiki.py --roberta /data/urikz/nki/roberta --tmp /tmp --output /data/urikz/nki/bin-v3-threshold20/valid --entity-vocab /data/urikz/nki/bin-v2-threshold20/entity.dict.txt --limit-set-of-entities --append-eos --nworkers 60 --data "/data/urikz/nki/wiki/annotated_el_unquote_expand2_v2/*/wiki99*"
-- Found 425 files
-- Build entity vocab mode: OFF
Processing Wiki: 100%|█| 425/425 [00:09<00:00 , s=237287, ann=691434, f_ed=407930, f_h_overlap=307946, f_self_overlap=1159, f_cross_s_bd=156, f_solo_s=
```
4. Construct training dataset
```console
$ python scripts/prepare_wiki.py --roberta /data/urikz/nki/roberta --tmp /tmp --output /data/urikz/nki/bin-v3-threshold20/train --entity-vocab /data/urikz/nki/bin-v3-threshold20/entity.dict.txt --limit-set-of-entities --append-eos --nworkers 60 --data "/data/urikz/nki/wiki/annotated_el_unquote_expand2_v2/*/wiki[0-8]*,/data/urikz/nki/wiki/annotated_el_unquote_expand2_v2/*/wiki9[0-8]*"
-- Found 42137 files
-- Build entity vocab mode: OFF
Processing Wiki: 100%|█| 42137/42137 [08:38<00:00 , s=2.35e+7, ann=6.87e+7, f_ed=4.06e+7, f_h_overlap=3e+7, f_self_overlap=111276, f_cross_s_bd=16101,
```

# FewRel Corpus

[FewRel v1 Official Webiste ](https://thunlp.github.io/1/fewrel1.html)

### Latest dataset location: fantasia.usc.edu:/data2/urikz/fewrel/bin

1. Download training and validation data
> $ wget https://github.com/thunlp/FewRel/raw/master/data/train_wiki.json

> $ wget https://github.com/thunlp/FewRel/raw/master/data/val_wiki.json

2. Preprocess the data

> $ python scripts/prepare_fewrel.py --roberta /data2/urikz/nki/roberta --data /data2/urikz/fewrel/val_wiki.json --output /data2/urikz/fewrel/bin/valid --append-eos

> $ python scripts/prepare_fewrel.py --roberta /data2/urikz/nki/roberta --data /data2/urikz/fewrel/train_wiki.json --output /data2/urikz/fewrel/bin/train --append-eos


# MTB Triplet Datasets

## What is an MTB triplet?
An MTB triplet is a directed triplet of the form (sentence_id, entity_1, entity_2), which has been verified to satisfy case0 (i.e, there exists at least one other sentence containing entity_1 and entity_2) and case1 (i.e., there exists at least one other sentence containing entity_1 but not entity_2). 

Thus, an MTB triplet can be used as the first sentence in a positive sentence pair (case0) or a strong negative sentence pair (case1), with respect to entity_1 and entity_2. Recall that weak negative pairs are those in which the two constituent sentences share no entities (case2).

## How to use the MTB triplet dataset
First, we assume here that the dataset is `bin-v3-threshold20`, but we also generate MTB triplets for `bin-v3-threshold20-small`.

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

__TODO: Compress Step 2 into a single script and fewer data files__

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
- __size of data__: 3.9GB
- __number of triplets__: 173,948,278
- __number of sentences__: 23,520,420
- __triplet dataset generation time__: 53m (6m for ent_pair_counts; 49m for mtb_triplets)
- __epoch runtime__: ~36h
#### valid
- __data path__: `waldstein.usc.edu:/data1/aarchan/self_inference/data/nki/bin-v3-threshold20/mtb_triplets_valid.npy`
- __size of data__: 29MB
- __number of triplets__: 1,238,670 
- __number of sentences__: 237,147
- __triplet dataset generation time__: 6.5m (6m for ent_pair_counts; 29s for mtb_triplets)
- __epoch runtime__: ~11.5m

## bin-v3-threshold20-small
#### train
- __data path__: `waldstein.usc.edu:/data1/aarchan/self_inference/data/nki/bin-v3-threshold20-small/mtb_triplets_train.npy`
- __size of data__: 356MB
- __number of triplets__: 15,527,749
- __number of sentences__: 2,138,107
- __triplet dataset generation time__: 4m (31s for ent_pair_counts; 3.5m for mtb_triplets)
- __epoch runtime__: ~3.5h
#### valid
- __data path__: `waldstein.usc.edu:/data1/aarchan/self_inference/data/nki/bin-v3-threshold20-small/mtb_triplets_valid.npy`
- __size of data__: 18MB
- __number of triplets__: 746,529
- __number of sentences__: 237,147
- __triplet dataset generation time__: 50s (31s for ent_pair_counts; 19s for mtb_triplets)
- __epoch runtime__: ~4.5m

## Note
Epoch runtime estimates computed using the following settings: 
- __machine__: waldstein
- __encoder__: RoBERTa-small
- __max_sentences__: 300
- __max_tokens__: 1.5e4
- __update_freq__: 1
- __num gpus__: 3 Titan Xp
- __num workers__: 1
