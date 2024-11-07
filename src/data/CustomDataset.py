
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import logging
import psutil
import glob
import torch
import time
import sys
import io
import os

#import matplotlib
#matplotlib.use('Agg') 

from torchvision.transforms import v2
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from PIL import Image
from joblib import load

from src.data.processing.create_images import create_2grid_images, create_6grid_images, create_overlay_images
from src.data.utils import get_normalization

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, partition, dataset_config, name_dataset, config):
        self.config = config
        self.name_dataset = name_dataset

        self.dataset_config = dataset_config
        self.dict_cols = dataset_config['dict_columns']
        self.num_workers = self.config['loader']['num_workers']
        self.use_png = self.config['loader']['use_png']
        if self.use_png:
            self.transform = v2.Compose([
                v2.ToImage(), 
                v2.ToDtype(torch.int8, scale=True)
                ])

        self.partition = partition.set_index(self.dict_cols['snid'])
        self.dataset = self.generate_image_dataset(dataset)
            
    def __getitem__(self, idx):
        obj_series = self.dataset.iloc[idx]

        image = obj_series['image']
        if self.use_png:
            image = self.transform(image)

        sample = {
            "id": obj_series[self.dict_cols['snid']],
            "y_true": obj_series[self.dict_cols['label']],
            "pixel_values": image 
        }
        return sample

    def __len__(self):
        return self.dataset.shape[0]

    def generate_image_dataset(self, dataset):
        logging.info(" -→ Image generation...")
        snid_name = self.dict_cols['snid']
        lcids = set(self.partition.index.values)

        if isinstance(dataset, list):
            dataset = pd.concat([df[df[snid_name].isin(lcids)] for df in dataset], ignore_index=True)
        elif isinstance(dataset, pd.DataFrame):
            dataset = dataset[dataset[snid_name].isin(lcids)]

        groups = [group for _, group in dataset.groupby(snid_name)]
        
        #dataset = []
        #for group in tqdm(groups, desc="Processing groups", mininterval=10, file=sys.stdout):
        #    result = self.normalize_and_create_image(group)
        #    dataset.append(result)

        with multiprocessing.Pool(processes=self.num_workers) as pool:
            dataset = list(tqdm(pool.imap(self.normalize_and_create_image, groups), total=len(groups), desc="Processing groups"))

        dataset = pd.DataFrame(dataset, columns=[snid_name, 'image', self.dict_cols['label']])
        logging.info(' -→ Image generation completed.')

        return dataset

    def normalize_and_create_image(self, group):
        for col in [self.dict_cols['flux'], self.dict_cols['flux_err'], self.dict_cols['mjd']]:
            group[col] = (group[col] - group[col].min()) / (group[col].max() - group[col].min())

        snid = group[self.dict_cols['snid']].iloc[0]
        image = self.create_image(group)
        label = self.partition.loc[snid, self.dict_cols['label']]
        return snid, image, label

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
