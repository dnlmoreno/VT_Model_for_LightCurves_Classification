#!/bin/bash

#SBATCH -p seas_gpu      # Partition name
#SBATCH -c 200
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:8  # Solicita 8 GPUs
#SBATCH -o ./outputs/output_%j.txt  # Archivo de salida con el ID del trabajo
#SBATCH -e ./errors/errors_%j.txt   # Archivo de error con el ID del trabajo
#SBATCH -t 7-00:00  # Límite de tiempo
#SBATCH --mem=800GB

# --- Set up software environment ---
module load python
module load cudnn/8.9.2.26_cuda12-fasrc01

source ~/.virtualenvs/Visual/bin/activate

CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib

# --- Set HYDRA variables ---
export HYDRA_CONFIG_PATH="../configs"
export HYDRA_CONFIG_NAME="run_config"

# --- Parallel Hyperparameter Search with Optuna ---
for i in {1..4}; do  # Cambia 4 por el número de GPUs que tienes disponibles
    srun -n 1 --gres=gpu:1 python -m scripts.run \
        ft_classification.model_name='swinv2' \
        ft_classification.pretrained_model.path="microsoft/swinv2-tiny-patch4-window16-256" \
        ft_classification.loader.path_data="data/images/alcock_multiband/all/minmax_by_obj_2.56_m1.0_l2.0_eFalse_2grid" \
        ft_classification.training.batch_size=128 &
done

wait 


