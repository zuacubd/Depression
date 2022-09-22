#!/bin/bash
#
# Usage:
# ./main_evaluator.sh train   # train the models
# ./main_evaluator.sh test    # test the models
# ./main_evaluator.sh detect    # detect the depression

data_dir="/projets/sig/mullah/nlp/depression/"
track_year=2017
#track_year=2018
task="erisk"
#features="bow"
#features="doc2vec"
#features_type="word2vec"
features_type="bert"
#train (1)
#test (2)
#detect (3)

sbatch -c 4 -p 48CPUNodes --mem-per-cpu 7168M -o logs/$task.$track_year.$features_type.extract.out -e logs/$task.$track_year.$features_type.extract.err -J $task-$track_year-$features_type-extract scripts/extract-features.sh $data_dir $track_year $features_type
