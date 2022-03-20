#!/bin/bash

#SBATCH --job-name=fs-cls
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=train-save-test.out
#SBATCH --error=train-save-test.err

singularity exec --nv --bind /scratch/$USER \
--overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
--overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
--overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
/scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif \
/bin/bash -c "
source /ext3/env.sh
cd /scratch/$USER/FSC-codebase
python train.py --dataset miniImagenet --model Conv4 --method baseline --train_aug
python save_features.py --dataset miniImagenet --model Conv4 --method baseline --train_aug
python test.py --dataset miniImagenet --model Conv4 --method baseline --train_aug
"