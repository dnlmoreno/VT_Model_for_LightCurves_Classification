#!/bin/bash



#SBATCH -p seas_gpu      # Partition name

#SBATCH -c 32
#SBATCH  --gres=gpu:1          # Request to allocate one GPU for this job
#SBATCH -o ./outputs/output_%j.txt      # File to which STDOUT will be written, including job ID
#SBATCH -e ./errors/errors_%j.txt      # File to which STDERR will be written, including job ID
#SBATCH  -t 1-00:00       # Time limit in the format hh:mm:ss
#SBATCH --mem=100GB

# --- Set up software environment ---
module load python
module load cudnn/8.9.2.26_cuda12-fasrc01

source ~/.virtualenvs/Visual/bin/activate

CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib

# --- Run the code ---
srun -n 1 --gres=gpu:1 python -m scripts.a
