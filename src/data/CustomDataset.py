import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import logging
import psutil
import glob
import torch
import sys
import io
import os

from PIL import Image
from joblib import load

from src.data.utils import get_normalization, create_2grid_images, create_6grid_images, create_overlay_images

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, partition, dataset_config, name_dataset, config):
        self.config = config
        self.name_dataset = name_dataset

        self.dataset = dataset
        self.partition = partition
        self.dataset_config = dataset_config
        self.dict_cols = dataset_config['dict_columns']

        self.dataset = self.filter_and_normalize_data(dataset)
    
        self.memory_alert_sent = False
        self.cache_enabled = self.config['training'].get('cache_enabled', False)
        if self.cache_enabled:
            self.image_cache = {}
            total_memory = self.get_total_memory()

            memory_usage_percentage = 0.90

            self.memory_threshold = total_memory * memory_usage_percentage
            logging.info(
                f'🗃️ Cache enabled | Total memory: {total_memory / (1024**3):.2f} GB | '
                f'Memory threshold: {self.memory_threshold / (1024**3):.2f} GB '
                f'({memory_usage_percentage * 100:.0f}% of total)'
            )
            
    def get_total_memory(self):
        """Get the total memory available, considering SLURM allocation if applicable."""
        slurm_mem = os.getenv('SLURM_MEM_PER_NODE')
        if slurm_mem:
            total_memory = int(slurm_mem) * 1024 * 1024  # Convert MB to bytes
        else:
            total_memory = psutil.virtual_memory().total  # Fallback to total system memory
        return total_memory

    def check_memory_usage(self):
        """Check if memory usage has exceeded the threshold and log a warning the first time."""
        mem = psutil.virtual_memory()
        if not self.memory_alert_sent and mem.used >= self.memory_threshold:
            logging.warning(f'⚠️ Memory usage has exceeded {self.memory_threshold / (1024**3):.2f} GB!')
            self.memory_alert_sent = True
            self.log_image_cache_memory()

    def log_image_cache_memory(self):
        """Log the current memory usage of the image cache."""
        total_size = self.get_size_of_image_cache()
        logging.info(f'📦 The image cache is using {total_size / (1024**3):.2f} GB of memory.')

    def get_size_of_image_cache(self):
        """Recursively calculate the memory size of the image cache."""
        print(f'Largo: {len(self.image_cache)}')
        total_size = sys.getsizeof(self.image_cache) 
        for key, value in self.image_cache.items():
            total_size += sys.getsizeof(key) 
            if isinstance(value, torch.Tensor):
                total_size += value.element_size() * value.numel()
            else:
                total_size += sys.getsizeof(value) 
        return total_size

    def __getitem__(self, idx):
        obj_series = self.partition.iloc[idx]
        snid = obj_series[self.dict_cols['snid']]

        # Check memory usage before processing
        if self.cache_enabled:
            self.check_memory_usage()

        if self.cache_enabled and snid in self.image_cache:
            image = self.image_cache[snid]
        else:
            obj_lc_df = self.dataset[self.dataset[self.dict_cols['snid']] == snid]

            if self.config['imgs_params']['input_type'] == '2grid': 
                image = create_2grid_images(obj_lc_df, self.config, self.dataset_config)
            elif self.config['imgs_params']['input_type'] == '6grid': 
                image = create_6grid_images(obj_lc_df, self.config, self.dataset_config)
            elif self.config['imgs_params']['input_type'] == 'overlay':
                image = create_overlay_images(obj_lc_df, self.config, self.dataset_config, self.name_dataset)
            else: 
                raise f"The input_type called: {self.config['imgs_params']['input_type']} is not implemented"

            image = torch.tensor(np.array(image))

            if self.cache_enabled:
                if psutil.virtual_memory().used < self.memory_threshold:
                    self.image_cache[snid] = image

        sample = {
            "id": snid,
            "y_true": obj_series[self.dict_cols['label']],
            "pixel_values": image
        }

        return sample

    def __len__(self):
        return self.dataset[self.dict_cols['snid']].nunique()

    def filter_and_normalize_data(self, dataset):
        snid_name = self.dict_cols['snid']
        lcids = set(self.partition[snid_name].values) 

        if isinstance(dataset, list):
            dataset = pd.concat([df[df[snid_name].isin(lcids)] for df in dataset], ignore_index=True)
        elif isinstance(dataset, pd.DataFrame):
            dataset = dataset[dataset[snid_name].isin(lcids)]
        
        dataset = get_normalization(dataset, self.config['imgs_params']['norm_name'], self.dict_cols)
        return dataset
