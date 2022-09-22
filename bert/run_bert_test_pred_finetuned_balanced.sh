#!/bin/bash

#SBATCH --job-name=Prediction_finetune_balanced_bert_2017
#SBATCH --output=log/gpu_prediction_finetune_balanced_bert_2017.out
#SBATCH --error=log/gpu_prediction_finetune_balanced_bert_2017.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=GPUNodes #RTX6000Node #GPUNodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --mem-per-cpu=7800M


track_year=$1
echo "starting ..."
srun singularity exec /logiciels/containerCollections/CUDA11/pytorch-NGC-20-06-py3.sif /users/sig/mullah/.conda/envs/e36t11/bin/python bert_test_pred_finetuning_balanced.py $track_year
echo "done."
