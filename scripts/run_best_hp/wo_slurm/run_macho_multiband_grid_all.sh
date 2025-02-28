#!/bin/bash

# --- Set HYDRA variables ---
export HYDRA_CONFIG_PATH="../configs/online"
export HYDRA_CONFIG_NAME="run_config"

# --- Run the code ---
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_online \
        ft_classification.model_name='swinv2' \
        ft_classification.pretrained_model.path="microsoft/swinv2-tiny-patch4-window16-256" \
        ft_classification.loader.name_dataset='macho_multiband' \
        ft_classification.loader.spc='all' \
        ft_classification.training.batch_size=64 \
        ft_classification.training.lr=5.0e-6 \
        ft_classification.imgs_params.input_type='2grid' \
        ft_classification.imgs_params.use_err=true \
        ft_classification.imgs_params.fig_params.markersize=3.0 \
        ft_classification.imgs_params.fig_params.linewidth=1.5 \
        ft_classification.imgs_params.fig_params.fmt='-o' \
        ft_classification.is_searching_hyperparameters=false \
        ft_classification.training.patience=10 \
        ft_classification.list_folds="[0, 1, 2]"