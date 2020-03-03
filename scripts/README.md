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

## bin-v3-threshold20
#### train
- data path: `waldstein.usc.edu:/data1/aarchan/self_inference/data/nki/bin-v3-threshold20/mtb_triplets_train.npy` 
- number of triplets: 174,109,469
- number of sentences: 23,520,420
- triplet dataset generation time: 17.5 mins
- epoch runtime: ~40 hrs
#### valid
- data path: `waldstein.usc.edu:/data1/aarchan/self_inference/data/nki/bin-v3-threshold20/mtb_triplets_valid.npy`
- number of triplets: 1,470,813 
- number of sentences: 237,147
- triplet dataset generation time: 16 secs
- epoch runtime: ~14 mins

## bin-v3-threshold20-small
#### train
- data path: `waldstein.usc.edu:/data1/aarchan/self_inference/data/nki/bin-v3-threshold20-small/mtb_triplets_train.npy`
- number of triplets: 15,717,417
- number of sentences: 2,138,107
- triplet dataset generation time: 88 secs 
- epoch runtime: ~3.5 hrs
#### valid
- data path: `waldstein.usc.edu:/data1/aarchan/self_inference/data/nki/bin-v3-threshold20-small/mtb_triplets_valid.npy`
- number of triplets: 1,470,813
- number of sentences: 237,147
- triplet dataset generation time: 8 secs
- epoch runtime: ~10 mins

## Note
- Epoch runtime estimates computed using 3 Titan Xp GPUs and 1 worker.
