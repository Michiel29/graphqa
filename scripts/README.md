# Wikipedia Corpus

### Latest dataset location: fantasia.usc.edu:/data2/urikz/nki/bin-v2

1. Create output directory
> $ mkdir /data/urikz/nki/bin-v2/
2. Build dictionary of all entities
> $ python scripts/prepare_wiki.py --roberta /data/urikz/nki/roberta --tmp /tmp --entity-vocab /data/urikz/nki/bin-v2/entity.dict.txt --limit-set-of-entities --append-eos --nworkers 30 --data "/data/urikz/nki/wiki/annotated_el_unquote_expand2_v2/\*/wiki\*"
3. Construct validation dataset
> $ python prepare_wiki.py --roberta /data/urikz/nki/roberta --tmp /tmp --output /data/urikz/nki/bin-v2/valid --entity-vocab /data/urikz/nki/bin-v2/entity.dict.txt --limit-set-of-entities --append-eos --nworkers 30 --data "/data/urikz/nki/wiki/annotated_el_unquote_expand2_v2/\*/wiki99\*"

> Processing Wiki: 100%|█| 425/425 [00:39<00:00 , s=309681, ann=937759, f_ed=407930, f_h_overlap=307946, f_self_overlap=1159, f_cross_s_bd=384, f_solo_s=8886, f_xao=1057

4. Construct training dataset
> $ python prepare_wiki.py --roberta /data/urikz/nki/roberta --tmp /tmp --output /data/urikz/nki/bin-v2/train --entity-vocab /data/urikz/nki/bin-v2/entity.dict.txt --limit-set-of-entities --append-eos --nworkers 30 --data "/data/urikz/nki/wiki/annotated_el_unquote_expand2_v2/\*/wiki[0-8]\*,/data/urikz/nki/wiki/annotated_el_unquote_expand2_v2/\*/wiki9[0-8]\*"

> ...


# FewRel Corpus

[FewRel v1 Official Webiste ](https://thunlp.github.io/1/fewrel1.html)

### Latest dataset location: fantasia.usc.edu:/data2/urikz/fewrel/bin

1. Download training and validation data
> $ wget https://github.com/thunlp/FewRel/raw/master/data/train_wiki.json

> $ wget https://github.com/thunlp/FewRel/raw/master/data/val_wiki.json

2. Preprocess the data

> $ python prepare_fewrel.py --roberta /data2/urikz/nki/roberta --data /data2/urikz/fewrel/val_wiki.json --output /data2/urikz/fewrel/bin/valid --append-eos

> $ python prepare_fewrel.py --roberta /data2/urikz/nki/roberta --data /data2/urikz/fewrel/train_wiki.json --output /data2/urikz/fewrel/bin/train --append-eos