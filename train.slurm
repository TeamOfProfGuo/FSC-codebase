#!/bin/bash

#SBATCH --job-name=fs-cls
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1


dataset=$1
model=$2
exp_dir=exp/${dataset}/${model}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
# config=config/${dataset}/${dataset}_${exp_name}.yaml

mkdir -p ${exp_dir}


echo "start"
singularity exec --nv \
--overlay /scratch/lg154/python36/python36.ext3:ro \
/scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
/bin/bash -c "
source /ext3/env.sh;
python train.py --dataset ${dataset} --model ${model} --method baseline --train_aug > ${exp_dir}/train.log 2>&1
"
echo "finish"
