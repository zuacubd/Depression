#!/bin/bash

#SBATCH --job-name=Depression_test
#SBATCH --output=logs/depression_test.log
#SBATCH --error=logs/depression_test.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=24CPUNodes
#SBATCH --mem-per-cpu=8000M


#source scripts/aven.sh

data_dir="/projets/sig/mullah/nlp/depression/"
track_year=2017
#process 1(train), 2(test), and 3(predict)
process=2

echo "launching ..."
python3 app/main_depression.py -data_dir $data_dir -p $process -ty $track_year
echo "done."

