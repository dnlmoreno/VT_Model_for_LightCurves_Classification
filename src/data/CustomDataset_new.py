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

from multiprocessing import Pool, cpu_count
from tqdm import tqdm
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
        self.num_workers = self.config['loader']['num_workers']

        self.dataset = self.create_imgs_dataset(dataset)
            
    def __getitem__(self, idx):
        obj_series = self.partition.iloc[idx]
        snid = obj_series[self.dict_cols['snid']]
        sample = {
            "id": snid,
            "y_true": obj_series[self.dict_cols['label']],
            "pixel_values": self.dataset.loc[snid, 'image']
        }
        return sample

    def __len__(self):
        return self.dataset[self.dict_cols['snid']].nunique()

    def create_imgs_dataset(self, dataset):
        logging.info(" -→ 🔄 Image generation...")
        snid_name = self.dict_cols['snid']
        lcids = set(self.partition[snid_name].values) 

        if isinstance(dataset, list):
            dataset = pd.concat([df[df[snid_name].isin(lcids)] for df in dataset], ignore_index=True)
        elif isinstance(dataset, pd.DataFrame):
            dataset = dataset[dataset[snid_name].isin(lcids)]
        
        groups = [group for _, group in dataset.groupby(self.dict_cols['snid'])]

        with Pool(processes=self.num_workers) as pool:
            dataset = list(tqdm(pool.imap(self.normalize_and_create_image, groups), total=len(groups), desc="Processing groups"))

        dataset = pd.DataFrame(dataset, columns=['oid', 'image']).set_index('oid')
        logging.info(' -→ ✅ Image generation setup completed.')
    
        return dataset

    def normalize_and_create_image(self, group):
        # Normalización Min-Max
        for col in [self.dict_cols['flux'], self.dict_cols['flux_err'], self.dict_cols['mjd']]:
            group[col] = (group[col] - group[col].min()) / (group[col].max() - group[col].min())

        oid = group[self.dict_cols['snid']].iloc[0]
        image = self.create_imgs_dataset(group)
        return oid, image

    def create_imgs_dataset(self, group):
        if self.config['imgs_params']['input_type'] == '2grid': 
            image = create_2grid_images(group, self.config, self.dataset_config)
        elif self.config['imgs_params']['input_type'] == '6grid': 
            image = create_6grid_images(group, self.config, self.dataset_config)
        elif self.config['imgs_params']['input_type'] == 'overlay':
            image = create_overlay_images(group, self.config, self.dataset_config, self.name_dataset)
        else: 
            raise f"The input_type called: {self.config['imgs_params']['input_type']} is not implemented"
        return torch.tensor(np.array(image))
