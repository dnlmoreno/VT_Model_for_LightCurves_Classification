import webdataset as wds
import pandas as pd
import numpy as np
import time

import pickle
import torch
import glob
import os

from src.data.create_images import create_single_png_image
from src.data.normalizations import min_max_normalize
from src.utils import save_yaml

import warnings
warnings.filterwarnings('ignore')
    

dict_mapping_classes = {
    'alcock': {
        'Cep_0': 0,  # Cepheid Type I
        'Cep_1': 1,  # Cepheid Type II
        'EC': 2,     # Eclipsing Binary
        'LPV': 3,    # Long Period Variable
        'RRab': 4,   # RR Lyrae Type AB
        'RRc': 5,    # RR Lyrae Type C
        },
    'ogle': {
        'EC': 0,      # Eclipsing Binary
        'ED': 1,      # Detached Binary (often used synonymously with EC in some contexts)
        'ESD': 2,     # Semi-Detached Binary
        'Mira': 3,    # Mira Variable
        'OSARG': 4,   # Small Amplitude Red Giants (often treated as semi-regular or less variable than Mira)
        'RRab': 5,    # RR Lyrae Type AB
        'RRc': 6,     # RR Lyrae Type C
        'SRV': 7,     # Semi-Regular Variable
        'cep': 8,     # Cepheid Variables
        'dsct': 9,    # Delta Scuti
        },
    'atlas': {
        'CB': 0,      # Close Binaries
        'DB': 1,      # Detached Binary
        'Mira': 2,    # Mira Variable
        'Other': 3,   # Other categories not specified here
        'Pulse': 4,   # Pulsating stars (General category)
    },
}


def read_lc_files(data_dir, list_spc, data_info, normalization, fig_params, path_save): 
    path_files_by_folds = glob.glob('{}/*'.format(data_dir))

    # Fold_0
    for path_files_by_fold in path_files_by_folds:
        print("*"*30 + f" Fold: {path_files_by_fold} " + "*"*30)
        fold_name = path_files_by_fold.split('/')[-1]

        # Alcock_20
        path_shots = glob.glob('{}/*'.format(path_files_by_fold))
        for path_shot in path_shots:
            spc = path_shot.split('/')[-1].split('_')

            if len(spc) > 1:
                spc = spc[-1]
            else:
                spc = 'all'

            if spc in list_spc:
                # Train
                path_subsets = glob.glob('{}/*'.format(path_shot))
                for path_subset in path_subsets:
                    subset = path_subset.split('/')[-1]
                    if path_subset.split('/')[-1] == 'objects.csv':
                        continue
 
                    print(f'- subset: {subset}')
                    entire_dataset = []

                    # Cep_0
                    path_objects = glob.glob('{}/*'.format(path_subset))
                    for path_object in path_objects:
                        path_files = glob.glob('{}/*'.format(path_object))

                        # Chunks.pickles
                        for path_file in path_files:
                            entire_dataset.append(pd.read_pickle(path_file))

                        if subset == 'test':
                            break
                        
                    # Entire dataset
                    entire_dataset = pd.concat(entire_dataset)
                    entire_dataset = entire_dataset.explode('lc_data')
                    entire_dataset[['mjd', 'flux', 'flux_err']] = pd.DataFrame(entire_dataset['lc_data'].tolist(), 
                                                                               index=entire_dataset.index)
                    
                    # I removed the negative error magnitudes because it doesn't make sense
                    entire_dataset = entire_dataset[entire_dataset['flux_err'] >= 0]                  

                    # Normalization
                    if normalization['norm_name'] == 'minmax_by_obj':
                        entire_dataset = entire_dataset.groupby('lcid').apply(lambda x: min_max_normalize(x, normalization)).reset_index(drop=True)
                    else:
                        raise 'No hay otra normalizacion implementada. Usa minmax_by_obj'
                    
                    # Create Shards
                    samples_per_shard = 10000
                    path_save_data = path_save + f'/{spc}/{subset}/{fold_name}'
                    output_pattern = f"{path_save_data}/imgs_lc-%06d.tar.gz"
                    os.makedirs(path_save_data, exist_ok=True)

                    entire_dataset.drop(['index'], axis=1).to_parquet(f'{path_save_data}/lc_data.parquet')

                    # Proceso de cada objeto y guardado en shard
                    with wds.ShardWriter(output_pattern, maxcount=samples_per_shard) as sink:
                        for i, snid in enumerate(entire_dataset['lcid'].unique()):
                            #print(f'Processed {i} objects')
                            obj_df = entire_dataset[entire_dataset['lcid'] == snid]

                            sample = {"__key__": "lc%06d" % i}
                            sample["ID.txt"] = str(snid)                

                            label = obj_df['label'].iloc[0]    
                            sample["label.cls"] = label

                            # PLOT
                            image = create_single_png_image(data=obj_df, 
                                                            normalization=normalization,
                                                            fig_params=fig_params)
                            
                            sample['pixel_values.pth'] = torch.from_numpy(np.array([image]))
                            
                            sink.write(sample)          

                            if i % 200 == 0:
                                print(f'Processed {i} objects')

                    print("Sharding complete.\n")

                save_yaml(data_info, f'{path_save}/{spc}/data_info.yaml')



if __name__ == "__main__":

    name_dataset = 'ogle'
    data_dir = 'data/lightcurves/astromer/{}'.format(name_dataset)
    list_spc = ['100']  #['20', '50', '100', '500'] #['20'] #, 50, 100, 500]

    num_classes = len(dict_mapping_classes[name_dataset])
    data_info = {
        'dict_mapping_classes': dict_mapping_classes[name_dataset],
        'name_dataset': name_dataset,
        'num_classes': num_classes,
        'num_channels': 3,
    }

    normalization = {
        'norm_name': 'minmax_by_obj',
        'norm_err': False,
    }

    fig_params = {
        'use_err': True,
        'figsize': (2.24, 2.24),
        'fmt': 'o', #-o
        'markersize': 2.5,
        'color': 'tab:blue',
        'alpha': 0.5,
        'ylim': (-0.05, 1.05),
    }

    path_save = 'data/images/astromer/{}/minmax_by_obj_224'.format(name_dataset) 

    start = time.time()
    read_lc_files(data_dir, list_spc, data_info, normalization, fig_params, path_save)
    end = time.time()

    print(f"Tiempo de extracción {end-start}")


