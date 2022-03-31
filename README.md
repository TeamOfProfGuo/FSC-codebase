# Few-Shot Classification Codebase

Adopted from **A Closer Look at Few-shot Classification** [repo](https://github.com/wyharveychen/CloserLookFewShot).

## Requisites
- Test Env: Python 3.9.7 (Singularity)
- Packages:
    - torch (1.10.2+cu113), torchvision (0.11.3+cu113)
    - numpy, scipy, pandas
    - h5py

## Clone codebase
```
cd /scratch/$USER
git clone https://github.com/TeamOfProfGuo/FSC-codebase
cd FSC-codebase
```

## Datasets
We use miniImagenet for testing, please [request your access to ImageNet on Greene](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/hpc-storage/datasets#h.8ez1zupnvbl3).
```
# prepare miniImagenet

# switch to a compute node
srun --nodes=1 --cpus-per-task=4 --mem=32GB --time=1:00:00 --pty /bin/bash

# activate Singularity (path can be different)
singularity exec --bind /scratch/$USER \
--overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
--overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro \
--overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
/scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash
source /ext3/env.sh

# fetch files
cd /scratch/$USER/FSC-codebase/filelists/miniImagenet
bash prepare.sh

# exit Singularity
exit
```

## Codebase Testing
**Note:** Modify the path in slurm scripts (as needed) before you start.
```
# switch to project root
cd /scratch/$USER/FSC-codebase

# train & save & test
sbatch train-save-test.slurm

# After the job completes:
cat record/results.txt
# Intended output:
# Time: 20220320-191117, Setting: miniImagenet-novel-Conv4-baseline-aug 5shot 5way_test, Acc: 600 Test Acc = 59.89% +- 0.74%
```
Alternatively, you can also run them one by one:
```
# switch to project root
cd /scratch/$USER/FSC-codebase

# train
sbatch train.slurm
# => log: ./train.log
# => save dir: ./checkpoints/

# save features
sbatch save_features.slurm
# => log: ./save_features.log
# => save dir: ./features/

# test
sbatch test.slurm
# => log: ./test.log
# => results: ./record/results.txt

# After the job completes:
cat record/results.txt
# Intended output:
# Time: 20220320-191117, Setting: miniImagenet-novel-Conv4-baseline-aug 5shot 5way_test, Acc: 600 Test Acc = 59.89% +- 0.74%
```