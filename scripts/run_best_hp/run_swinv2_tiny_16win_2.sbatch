#!/bin/bash



#SBATCH -p seas_gpu      # Partition name

#SBATCH -c 32
#SBATCH  --gres=gpu:nvidia_a100-sxm4-80gb:1          # Request to allocate one GPU for this job
#SBATCH -o ./outputs/output_%j.txt      # File to which STDOUT will be written, including job ID
#SBATCH -e ./errors/errors_%j.txt      # File to which STDERR will be written, including job ID
#SBATCH  -t 7-00:00       # Time limit in the format hh:mm:ss
#SBATCH --mem=150GB

# --- Set up software environment ---
module load python
module load cudnn/8.9.2.26_cuda12-fasrc01

source ~/envs/VisionTrans/bin/activate

CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib

# --- Set HYDRA variables ---
export HYDRA_CONFIG_PATH="../configs/online"
export HYDRA_CONFIG_NAME="run_config"

# --- Run the code ---
srun -n 1 --gres=gpu:1 python -m scripts.run_online \
        ft_classification.model_name='swinv2' \
        ft_classification.checkpoint.use=true \
        ft_classification.checkpoint.run_id='fd9d5c09b5d8475fb5c92353da16b362' \
        ft_classification.pretrained_model.path="microsoft/swinv2-tiny-patch4-window16-256" \
        ft_classification.loader.name_dataset='elasticc_1' \
        ft_classification.training.batch_size=64 \
        ft_classification.training.lr=5.0e-6 \
        ft_classification.imgs_params.input_type='6grid' \
        ft_classification.imgs_params.use_err=true \
        ft_classification.imgs_params.fig_params.markersize=5.0 \
        ft_classification.imgs_params.fig_params.linewidth=1.5 \
        ft_classification.imgs_params.fig_params.fmt='-o' \
        ft_classification.is_searching_hyperparameters=true \
        ft_classification.training.patience=5


