# FewRel Dataset

[FewRel v1 Official Webiste ](https://thunlp.github.io/1/fewrel1.html)

1. Download training and validation data
> $ wget https://github.com/thunlp/FewRel/raw/master/data/train_wiki.json

> $ wget https://github.com/thunlp/FewRel/raw/master/data/val_wiki.json

2. Preprocess the data

> python prepare_fewrel.py --roberta /data2/urikz/nki/roberta --data /data2/urikz/fewrel/val_wiki.json --output /data2/urikz/fewrel/bin/valid --append-eos

> python prepare_fewrel.py --roberta /data2/urikz/nki/roberta --data /data2/urikz/fewrel/val_wiki.json --output /data2/urikz/fewrel/bin/valid --append-eos