# requires aws cli

pre_small_path="nki/bin-v5-threshold20-small"
pre_large_path="nki/bin-v5-threshold20"
semeval="SemEval2010_task8_all_data"
kbp37="kbp37"
fewrel="fewrel"
roberta="roberta"
tacred="tacred"

cd ../../data
zip -r data.zip $pre_small_path $pre_large_path $semeval $kbp37 $fewrel $roberta $tacred

aws s3 cp data.zip s3://selfinference/

