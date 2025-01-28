#!/bin/bash

#SBATCH -A p_datascale
#SBATCH --time=04:00:00
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8192M
#SBATCH --output=sh_info/Task%j_Output
#SBATCH --error=sh_info/Task%j_Error
#SBATCH --hint=nomultithread

module load release/23.04
module load Anaconda3/2022.05
source $EBROOTANACONDA3/etc/profile.d/conda.sh
conda activate main

if [ -z "$1" ]; then
    echo "Error: No dataset specified."
    echo "Usage: $0 <dataset_name>"
    exit 1
fi

conda run python runner.py --dataset "$1"