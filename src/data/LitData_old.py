import multiprocessing as mp
import webdataset as wds
import lightning as L
import pandas as pd
import logging
import time
import torch
import glob
import gc
import os
import io

from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader

from src.data.CustomDataset import CustomDataset
from scripts.utils import load_yaml
from src.data.utils import get_dataset

class LitData(L.LightningDataModule):
    def __init__(self, name_dataset, **kwargs):
        super().__init__()

        self.config = kwargs
        self.name_dataset = name_dataset
        self.path_data = self.config['loader']['path_data']
        self.spc = self.config['loader'].get('spc', None)
        self.num_workers = self.config['loader']['num_workers']

        self.batch_size = self.config['training']['batch_size']
        self.fold = self.config['loader']['fold']
        self.use_weighted_sampling = self.config['training']['use_weighted_sampling']

        self.dataset_config = load_yaml('configs/datasets_config.yaml')[self.name_dataset]

    def prepare_data(self):
        if self.name_dataset == 'elasticc_1':
            self.partitions = pd.read_parquet(f'{self.path_data}/ATAT_partition/partitions_v1.parquet')
            self.dataset = get_dataset(self.path_data, self.dataset_config, self.name_dataset)

        elif self.name_dataset in ['alcock', 'alcock_multiband']:
            self.partitions = pd.read_parquet(f'{self.path_data}/ASTROMER_partition/partitions_v1.parquet')
            self.dataset = get_dataset(self.path_data, self.dataset_config, self.name_dataset)

        else:
            raise f"We don't have the implementation for the dataset called {self.name_dataset}"

        self.inv_mapping_classes = {value: key for key, value in self.dataset_config['dict_mapping_classes'].items()} 

    def setup(self, stage=None):
        logging.info('⚙️ Setting up the training dataset.')
        self.train_dataset = CustomDataset(
            dataset=self.dataset,
            partition=self.get_df_partition('train', self.fold),
            dataset_config=self.dataset_config,
            name_dataset=self.name_dataset,
            config=self.config,
        )
        logging.info('✅ Training dataset setup completed.')

        logging.info('⚙️ Setting up the validation dataset.')
        self.val_dataset = CustomDataset(
            dataset=self.dataset,
            partition=self.get_df_partition('val', self.fold),
            dataset_config=self.dataset_config,
            name_dataset=self.name_dataset,
            config=self.config,
        )
        logging.info('✅ Validation dataset setup completed.')

        logging.info('⚙️ Setting up the test dataset.')
        self.test_dataset = CustomDataset(
            dataset=self.dataset,
            partition=self.get_df_partition('test', self.fold),
            dataset_config=self.dataset_config,
            name_dataset=self.name_dataset,
            config=self.config,
        )
        logging.info('✅ Test dataset setup completed.')

        logging.info('🧹 Cleaning up the dataset from memory.')
        del self.dataset
        gc.collect()
        logging.info('🗑️ Dataset cleanup completed.')

    def create_weighted_sampler(self, dataset):
        y_true = [sample['y_true'] for sample in dataset]
        class_counts = torch.bincount(torch.tensor(y_true))
        class_weights = 1. / class_counts
        sample_weights = class_weights[torch.tensor(y_true)]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        return sampler

    def train_dataloader(self):
        sampler = self.create_weighted_sampler(self.train_dataset) if self.use_weighted_sampling else None
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          sampler=sampler,
                          shuffle=sampler is None)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size,
                          num_workers=self.num_workers) 

    def get_df_partition(self, subset_name, fold):
        condition = (self.partitions['subset'] == subset_name)
        if subset_name != 'test' or self.name_dataset in ['alcock', 'alcock_multiband']:
            condition &= (self.partitions['fold'] == fold)
        if self.spc is not None and self.name_dataset in ['alcock', 'alcock_multiband']:
            condition &= (self.partitions['spc'] == str(self.spc))
        return self.partitions[condition]
