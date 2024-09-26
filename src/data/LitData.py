import multiprocessing as mp
import webdataset as wds
import lightning as L
import logging
import time
import torch
import os
import io

from torchvision import transforms
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader


class LitData(L.LightningDataModule):
    def __init__(self, data_info, **kwargs):
        super().__init__()

        self.name_dataset = data_info['name_dataset']
        self.num_workers = kwargs['loader']['num_workers']
        self.batch_size = kwargs['training']['batch_size']
        self.path_data = kwargs['loader']['path_data']
        self.fold = kwargs['loader']['fold']

        self.inv_mapping_classes = {value: key for key, value in data_info['dict_mapping_classes'].items()} 

        self.transform = None
        self.buffer_size = 10000
        self.use_weighted_sampling = kwargs['training']['use_weighted_sampling']

    def prepare_data(self):
        self.dataset_url = {
            'train': self._get_url_data(mode='train'),
            'val': self._get_url_data(mode='val'),
            }
        try:
            self.dataset_url['test'] = self._get_url_data(mode='test')
        except:
            print("Test set won't be loaded, so the code won't calculate the metrics in the test set.")

        return self.dataset_url

    def _get_url_data(self, mode):
        path_subset = '{}/{}'.format(self.path_data, mode)
        
        if self.name_dataset in ['macho', 'alcock', 'atlas', 'ogle']:
            path_subset += '/fold_{}'.format(self.fold)
        else:
            if mode != 'test':
                path_subset += '/fold_{}'.format(self.fold)
            
        shard_files = [f for f in os.listdir(path_subset) if f.endswith(".gz")]
        if shard_files:
            min_index = min(int(f.split('-')[1].split('.')[0]) for f in shard_files)
            max_index = max(int(f.split('-')[1].split('.')[0]) for f in shard_files)
            url = f"{path_subset}/imgs_lc-{{{min_index:06d}..{max_index:06d}}}.tar.gz"
            return url
        else:
            raise FileNotFoundError("No shard files found in the directory.")

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            if self.use_weighted_sampling:
                self.train_dataset = self.load_data_to_memory(self.dataset_url['train'])
            else:
                self.train_dataset = (
                    wds.WebDataset(self.dataset_url['train'])
                    .map(lambda sample: self.get_input_model(sample))
                    .shuffle(self.buffer_size)
                )

            self.val_dataset = ( 
                wds.WebDataset(self.dataset_url['val'])
                .map(lambda sample: self.get_input_model(sample))
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = ( 
                wds.WebDataset(self.dataset_url['test'])
                .map(lambda sample: self.get_input_model(sample))
            )

    def create_weighted_sampler(self, dataset):
        y_true = [sample['y_true'] for sample in dataset]
        class_counts = torch.bincount(torch.tensor(y_true))
        class_weights = 1. / class_counts
        sample_weights = class_weights[torch.tensor(y_true)]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        return sampler

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers,
                          sampler=self.create_weighted_sampler(self.train_dataset) if self.use_weighted_sampling else None)

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

    def get_input_model(self, sample):
        sample = self.fn_decode(sample)

        sample_mod = sample['pixel_values.pth']
        sample_mod = sample_mod.permute(1, 2, 0, 3)
        sample_mod = sample_mod.reshape(sample_mod.size(0), sample_mod.size(1), -1)
        sample['pixel_values.pth'] = sample_mod
            
        if self.transform is not None:
            sample['pixel_values.pth'] = self.transform(sample['pixel_values.pth'])

        input_dict = {
            'id': sample['id.txt'],
            'pixel_values': sample['pixel_values.pth'],
            'y_true': sample['label.cls'],
        }
        return input_dict

    def fn_decode(self, sample):
        for key, value in sample.items():
            if key.endswith(".pth"):
                sample[key] = torch.load(io.BytesIO(value), weights_only=True)
            elif key.endswith(".txt"):
                sample[key] = value.decode("utf-8") 
            elif key.endswith(".cls"):
                sample[key] = int(value) 
        return sample

    def load_data_to_memory(self, url):
        logging.info('We are loading the train data in memory...')
        start_time = time.time()  # Start the timer
        
        dataset = wds.WebDataset(url)
        data_in_memory = []
        for sample in dataset:
            processed_sample = self.get_input_model(sample)
            data_in_memory.append(processed_sample)
        
        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time  # Calculate the elapsed time
        logging.info(f'Train data was loaded in {elapsed_time:.2f} seconds...')
        
        return data_in_memory


    #def load_data_to_memory(self, url):
    #    logging.info('We are loading the training data into memory...')
    #    
    #    def process_sample(sample):
    #        return self.get_input_model(sample)
    #    
    #    # Dataset decoding and preparation
    #    dataset = wds.WebDataset(url).decode()
    #    
    #    # Shared list to store data in memory
    #    manager = mp.Manager()
    #    data_in_memory = manager.list()
#
    #    def add_sample_to_memory(sample):
    #        processed_sample = process_sample(sample)
    #        data_in_memory.append(processed_sample)
#
    #    # Create a pool of workers
    #    with mp.Pool(self.num_workers) as pool:
    #        pool.map(add_sample_to_memory, dataset)
#
    #    logging.info('Training data was loaded...')
    #    return list(data_in_memory)