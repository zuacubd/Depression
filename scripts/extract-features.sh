#!/bin/bash

data_dir=$1
track_year=$2
#process 1(train), 2(test), and 3(predict)
features_type=$3

echo "launching ..."
python3 app/feature_extraction/$features_type/feature_extraction.py -data_dir $data_dir -ty $track_year -f $features_type
echo "done."

