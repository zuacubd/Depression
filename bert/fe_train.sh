#!/bin/bash

#SBATCH --job-name=train_fe_bert-2018
#SBATCH --output=log/gpu_train_fe_bert-2018.out
#SBATCH --error=log/gpu_train_fe_bert-2018.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=GPUNodes #RTX6000Node #GPUNodes
#SBATCH --gres=gpu:2
#SBATCH --gres-flags=enforce-binding
#SBATCH --mem-per-cpu=7800M


#data_dir="/projets/sig/mullah/nlp/depression/"
#track_year=2017
track_year=$1
#task="erisk"
#features="bow"
#features="doc2vec"
#features_type="word2vec"
#features_type="bert"
#train (1)
#test (2)
#detect (3)
echo "starting ..."
srun singularity exec /logiciels/containerCollections/CUDA11/pytorch-NGC-20-06-py3.sif /users/sig/mullah/.conda/envs/e36t11/bin/python feature_extraction_train.py $track_year
