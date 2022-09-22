#!/bin/bash

data_dir=$1
track_year=$2
#process 1(train), 2(test), and 3(predict)
features=$3
process=$4
pretrained=$5
model_state=$6 
dist=$7
agg_post=$8
agg_chunk=$9
early=${10}
cumulative=${11}

echo "launching ..."
python3 app/bert_depression.py -data_dir $data_dir -p $process -ty $track_year -f $features -e $early -pt $pretrained -ms $model_state -d $dist -ap $agg_post -ac $agg_chunk -c $cumulative
echo "done."

