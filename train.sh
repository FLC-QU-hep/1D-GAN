#!/bin/bash
#SBATCH --time 1-12:05:00
#SBATCH --nodes 1
#SBATCH --partition maxgpu
#SBATCH --constraint="GPUx1&A100"
#SBATCH --job-name gan_training

bash
source ~/.bashrc
conda activate A100-torch
# conda activate py36

cd /beegfs/desy/user/akorol/projects/1D-GAN

# python overfitting.py
python train.py
# python test_dataloader.py

exit
