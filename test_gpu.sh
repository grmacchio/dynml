#!/bin/bash

#SBATCH --job-name=testgpu
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=gm0796@princeton.edu
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:20:00

flake8 dynml
flake8 test_dynml

mypy dynml
mypy test_dynml

python -m pytest test_dynml
