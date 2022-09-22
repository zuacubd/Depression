#!/bin/bash
#
# Usage:
# ./bert_evaluator.sh train   # train the models
# ./bert_evaluator.sh test    # test the models
# ./bert_evaluator.sh detect    # detect the depression

data_dir="/projets/sig/mullah/nlp/depression/"
#track_year=2017
track_year=$2
task="erisk"
#features="bow"
#features="doc2vec"
#features="word2vec"
#features="ch2_fs"
features="bert"
pretrained="base_uncased"
model_state="finetuned" #pretrained
dist=$3 #balancednaturalbalanced
agg_post=$4
agg_chunk=$5 #"mean/max
cumulative=0 #1 (0: probability combine, 1: test chunk already combined)
#train (1)
#test (2)
#detect (3)
early=$6
#early=1

if [[ $1 = "train" ]]
then
	sbatch -c 4 -p 48CPUNodes --mem-per-cpu 7168M -o logs/$task.$track_year.$features.$1.out -e logs/$task.$track_year.$features.$1.err -J $task-$track_year-$features-$1 scripts/learn-and-predict-bert.sh $data_dir $track_year $features 1 $early

elif [[ $1 = "test" ]]
then
	sbatch -c 4 -p 48CPUNodes --mem-per-cpu 7168M -o logs/$task.$track_year.$features.$1.out -e logs/$task.$track_year.$features.$1.err -J $task-$track_year-$features-$1 scripts/learn-and-predict-bert.sh $data_dir $track_year $features 2 $early

elif [[ $1 = "detect" ]]
then
	sbatch -c 4 -p 48CPUNodes --mem-per-cpu 7168M -o logs/$task.$track_year.$features.$pretrained.$model_state.$dist.$agg_post.$agg_chunk.$1.$early.out -e logs/$task.$track_year.$features.$pretrained.$model_state.$dist.$agg_post.$agg_chunk.$1.$early.err -J $task-$track_year-$features-$pretrained-$model_state-$dist-$agg_post-$agg_chunk-$1-$early scripts/learn-and-predict-bert.sh $data_dir $track_year $features 3 $pretrained $model_state $dist $agg_post $agg_chunk $early $cumulative

fi
