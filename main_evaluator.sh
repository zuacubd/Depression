#!/bin/bash
#
# Usage:
# ./main_evaluator.sh train   # train the models
# ./main_evaluator.sh test    # test the models
# ./main_evaluator.sh detect    # detect the depression

data_dir="/projets/sig/mullah/nlp/depression/"
#track_year=2017
track_year=2018
task="erisk"
#features="bow"
#features="doc2vec"
features="word2vec"
#features="ch2_fs"
#train (1)
#test (2)
#detect (3)
early=0
#early=1


if [[ $1 = "train" ]]
then
	sbatch -c 4 -p 48CPUNodes --mem-per-cpu 7168M -o logs/$task.$track_year.$features.$1.out -e logs/$task.$track_year.$features.$1.err -J $task-$track_year-$features-$1 scripts/learn-and-predict.sh $data_dir $track_year $features 1 $early

elif [[ $1 = "test" ]]
then
	sbatch -c 4 -p 48CPUNodes --mem-per-cpu 7168M -o logs/$task.$track_year.$features.$1.out -e logs/$task.$track_year.$features.$1.err -J $task-$track_year-$features-$1 scripts/learn-and-predict.sh $data_dir $track_year $features 2 $early

elif [[ $1 = "detect" ]]
then
	sbatch -c 4 -p 48CPUNodes --mem-per-cpu 7168M -o logs/$task.$track_year.$features.$1.$early.out -e logs/$task.$track_year.$features.$1.$early.err -J $task-$track_year-$features-$1-$early scripts/learn-and-predict.sh $data_dir $track_year $features 3 $early

fi
