# Wikipedia Corpus

1. Create output directory
> mkdir /data/urikz/nki/bin-v2/
2. Build dictionary of all entities
> python prepare_wiki.py --roberta /data/urikz/nki/roberta --tmp /tmp --entity-vocab /data/urikz/nki/bin-v2/entity.dict.txt --limit-set-of-entities --append-eos --nworkers 30 --data "/data/urikz/nki/wiki/annotated_el_unquote_expand2_v2/*/wiki*"
3. Construct validation dataset
> python prepare_wiki.py --roberta /data/urikz/nki/roberta --tmp /tmp --output /data/urikz/nki/bin-v2/valid --entity-vocab /data/urikz/nki/bin-v2/entity.dict.txt --limit-set-of-entities --append-eos --nworkers 30 --data "/data/urikz/nki/wiki/annotated_el_unquote_expand2_v2/*/wiki[99]*"
4. Construct training dataset
> python prepare_wiki.py --roberta /data/urikz/nki/roberta --tmp /tmp --output /data/urikz/nki/bin-v2/train --entity-vocab /data/urikz/nki/bin-v2/entity.dict.txt --limit-set-of-entities --append-eos --nworkers 30 --data "/data/urikz/nki/wiki/annotated_el_unquote_expand2_v2/*/wiki[!99]*"


# FewRel Corpus

[FewRel v1 Official Webiste ](https://thunlp.github.io/1/fewrel1.html)

1. Download training and validation data
> $ wget https://github.com/thunlp/FewRel/raw/master/data/train_wiki.json

> $ wget https://github.com/thunlp/FewRel/raw/master/data/val_wiki.json

2. Preprocess the data

> python prepare_fewrel.py --roberta /data2/urikz/nki/roberta --data /data2/urikz/fewrel/val_wiki.json --output /data2/urikz/fewrel/bin/valid --append-eos

> python prepare_fewrel.py --roberta /data2/urikz/nki/roberta --data /data2/urikz/fewrel/val_wiki.json --output /data2/urikz/fewrel/bin/valid --append-eos