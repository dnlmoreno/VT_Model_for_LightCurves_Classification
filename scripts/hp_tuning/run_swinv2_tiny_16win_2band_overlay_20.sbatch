#!/bin/bash

#SBATCH -p seas_gpu      # Partition name
#SBATCH -c 32
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1  # Request to allocate one GPU for this job
#SBATCH -o ./outputs/output_%j.txt  # File to which STDOUT will be written, including job ID
#SBATCH -e ./errors/errors_%j.txt   # File to which STDERR will be written, including job ID
#SBATCH -t 7-00:00       # Time limit in the format hh:mm:ss
#SBATCH --mem=100GB

# --- Set up software environment ---
module load python
module load cudnn/8.9.2.26_cuda12-fasrc01

source ~/envs/VisionTrans/bin/activate

CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib

# --- Set HYDRA variables ---
export HYDRA_CONFIG_PATH="../configs/online"
export HYDRA_CONFIG_NAME="run_config"

# --- Define hyperparameter ranges ---
markersizes=(1 2 3 4 5)
linewidths=(0.5 1.0 1.5 2.0)
learning_rates=(5.0e-5 5.0e-6)
use_errs=(false true)

# --- Loop through hyperparameter combinations ---
for markersize in "${markersizes[@]}"
do
  for linewidth in "${linewidths[@]}"
  do
    for lr in "${learning_rates[@]}"
    do
      for use_err in "${use_errs[@]}"
      do
        # --- Run the code with current hyperparameters ---
        srun -n 1 --gres=gpu:1 python -m scripts.run_online \
          ft_classification.model_name='swinv2' \
          ft_classification.pretrained_model.path="microsoft/swinv2-tiny-patch4-window16-256" \
          ft_classification.loader.name_dataset='macho_multiband' \
          ft_classification.loader.spc=20 \
          ft_classification.training.batch_size=64 \
          ft_classification.imgs_params.fig_params.markersize=$markersize \
          ft_classification.imgs_params.fig_params.linewidth=$linewidth \
          ft_classification.imgs_params.use_err=$use_err \
          ft_classification.imgs_params.fig_params.fmt='-o' \
          ft_classification.training.lr=$lr \
          ft_classification.imgs_params.input_type='overlay' \
          ft_classification.is_searching_hyperparameters=true
          
      done
    done
  done
done

