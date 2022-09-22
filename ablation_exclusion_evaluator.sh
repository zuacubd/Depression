#!/bin/bash
#
# Usage:
# ./main_evaluator.sh train   # train the models
# ./main_evaluator.sh test    # test the models
# ./main_evaluator.sh detect    # detect the depression

data_dir="/projets/sig/mullah/nlp/depression/"
#track_year=2017
track_year=2018
task="depression"
features_type="bow"
#train (1)
#test (2)
#detect (3)

if [[ $1 = "train" ]]
then
	sbatch -c 4 -p 48CPUNodes --mem-per-cpu 7168M -o logs/$task.$track_year.$features_type.$1.ablation.exclusion.out -e logs/$task.$track_year.$features_type.$1.ablation.exclusion.err -J $task-$track_year-$features_type-$1-ablation-exclusion scripts/learn-and-predict-ablation-exclusion.sh $data_dir $track_year 1 $features_type

elif [[ $1 = "test" ]]
then
	sbatch -c 4 -p 48CPUNodes --mem-per-cpu 7168M -o logs/$task.$track_year.$features_type.$1.ablation.exclusion.out -e logs/$task.$track_year.$features_type.$1.ablation.exclusion.err -J $task-$track_year-$features_type-$1-ablation-exclusion scripts/learn-and-predict-ablation-exclusion.sh $data_dir $track_year 2 $features_type

elif [[ $1 = "detect" ]]
then
	sbatch -c 4 -p 48CPUNodes --mem-per-cpu 7168M -o logs/$task.$track_year.$features_type.$1.ablation.exclusion.out -e logs/$task.$track_year.$features_type.$1.ablation.exclusion.err -J $task-$track_year-$features_type-$1-ablation-exclusion scripts/learn-and-predict-ablation-exclusion.sh $data_dir $track_year 3 $features_type

fi
