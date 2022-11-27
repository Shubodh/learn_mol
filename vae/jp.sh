#!/bin/bash
#SBATCH -n 30
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=72:00:00
#SBATCH --mail-user=shivansh.seth@research.iiit.ac.in

WANDB_API_KEY=eb83c8968b3f04f158b08d0a69160b06cb90c75e
module add cuda/10.2
module add cudnn/7.6.5-cuda-10.2
~/miniconda3/envs/brain/bin/python -u main_vgae_qm9.py> asdf.log 2>&1
