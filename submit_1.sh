#!/bin/sh
#SBATCH --job-name=job1    # Job name
#SBATCH --ntasks=1                                 # Run on a single CPU
#SBATCH --time=24:00:00                            # Time limit hrs:min:sec
#SBATCH --output=test_mm_%j.out                    # Standard output and error log
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --partition=dgx

CUDA_HOME=/usr/local/cuda
CUDA_VISIBLE_DEVICES=1
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

nvcc --version
nvidia-smi
python kd.py