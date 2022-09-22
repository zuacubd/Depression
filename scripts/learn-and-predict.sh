#!/bin/bash

data_dir=$1
track_year=$2
#process 1(train), 2(test), and 3(predict)
features=$3
process=$4
early=$5

echo "launching ..."
python3 app/main_depression.py -data_dir $data_dir -p $process -ty $track_year -f $features -e $early
echo "done."

