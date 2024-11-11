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

#import matplotlib
#matplotlib.use('Agg') 

from PIL import Image
from joblib import load
from torchvision.transforms import v2

from src.data.utils import get_normalization
from src.data.processing.create_images import create_2grid_images, create_6grid_images, create_overlay_images

#from multiprocessing import Manager
import torch.multiprocessing as mp
#mp.set_start_method('spawn', force=True)

class CustomDataset_test(torch.utils.data.Dataset):
    def __init__(self, dataset, partition, dataset_config, name_dataset, config):
        self.config = config
        self.name_dataset = name_dataset

        self.dataset_config = dataset_config
        self.dict_cols = dataset_config['dict_columns']
        self.use_png = self.config['loader']['use_png']
        if self.use_png:
            self.transform = v2.Compose([
                v2.ToImage(), 
                v2.ToDtype(torch.int8, scale=True)
                ])

        self.partition = partition
        self.dataset = self.filter_and_normalize_data(dataset).set_index(self.dict_cols['snid'])

        self.cache_enabled = self.config['training'].get('cache_enabled', False)
        self.first_epoch = True
        if self.cache_enabled:
            manager = mp.Manager()
            #manager = Manager()
            self.image_cache = manager.dict()

    def __getitem__(self, idx):
        obj_series = self.partition.iloc[idx]
        snid = obj_series[self.dict_cols['snid']]

        obj_lc_df = self.dataset.loc[snid]

        if self.first_epoch and self.cache_enabled:
            image = self.create_image(obj_lc_df)
            self.image_cache[snid] = image
        else:
            #image = self.image_cache.loc[snid, 'image']
            image = self.image_cache[snid]

        if self.use_png:
            image = self.transform(image)

        sample = {
            "id": snid,
            "y_true": obj_series[self.dict_cols['label']],
            "pixel_values": image
        }

        return sample
        
    def __len__(self):
        return self.dataset.index.nunique()

    def filter_and_normalize_data(self, dataset):
        snid_name = self.dict_cols['snid']
        lcids = set(self.partition[snid_name].values)

        if isinstance(dataset, list):
            dataset = pd.concat([df[df[snid_name].isin(lcids)] for df in dataset], ignore_index=True)
        elif isinstance(dataset, pd.DataFrame):
            dataset = dataset[dataset[snid_name].isin(lcids)]
        
        self.partition = self.partition[self.partition[snid_name].isin(dataset[snid_name].unique())]
        dataset = [
            get_normalization(group, self.config['imgs_params']['norm_name'], self.dict_cols)
            for _, group in dataset.groupby(snid_name)
        ]
        dataset = pd.concat(dataset, ignore_index=True)
        return dataset

    def create_image(self, group):
        if self.config['imgs_params']['input_type'] == '2grid': 
            image = create_2grid_images(group, self.config, self.dataset_config)
        elif self.config['imgs_params']['input_type'] == '6grid': 
            image = create_6grid_images(group, self.config, self.dataset_config)
        elif self.config['imgs_params']['input_type'] == 'overlay':
            image = create_overlay_images(group, self.config, self.dataset_config, self.name_dataset)
        else: 
            raise f"The input_type called: {self.config['imgs_params']['input_type']} is not implemented"

        return image if self.use_png else torch.from_numpy(np.array(image))


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