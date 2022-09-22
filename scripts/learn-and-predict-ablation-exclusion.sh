#!/bin/bash

data_dir=$1
track_year=$2
#process 1(train), 2(test), and 3(predict)
process=$3
features_type=$4

echo "launching ..."
python3 app/ablation_exclusion_depression.py -data_dir $data_dir -p $process -ty $track_year -f $features_type
echo "done."

