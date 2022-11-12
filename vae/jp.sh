#!/bin/bash
#SBATCH -n 30
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=72:00:00
#SBATCH --mail-user=shivansh.seth@research.iiit.ac.in

module add cuda/10.2
module add cudnn/7.6.5-cuda-10.2
~/miniconda3/envs/brain/bin/python -u train.py> asdf.log 2>&1
