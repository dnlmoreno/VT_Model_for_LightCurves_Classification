import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import logging
import glob
import torch
import io

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
    
    def __getitem__(self, idx):
        obj_series = self.partition.iloc[idx]

        snid = obj_series[self.dict_cols['snid']]
        sample = {
            "id": snid,
            "y_true": obj_series[self.dict_cols['label']]
            }

        obj_lc_df = self.dataset[self.dataset[self.dict_cols['snid']] == snid]

        # Generar la imagen
        if self.config['imgs_params']['input_type'] == '2grid': 
            image = create_2grid_images(obj_lc_df, self.config, self.dataset_config)
        elif self.config['imgs_params']['input_type'] == '6grid': 
            image = create_6grid_images(obj_lc_df, self.config, self.dataset_config)
        elif self.config['imgs_params']['input_type'] == 'overlay':
            image = create_overlay_images(obj_lc_df, self.config, self.dataset_config)
        else: 
            print(self.config)
            print(self.config['input_type'])
            raise f"The input_type called: {self.config['input_type']} is not implemented"
        
        sample['pixel_values'] = torch.tensor(np.array(image))

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

