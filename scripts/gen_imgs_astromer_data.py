
import os
import io
import glob
import hydra
import shutil
import numpy as np
import pandas as pd
import webdataset as wds
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from omegaconf import DictConfig
from multiprocessing import Pool, Manager
from tqdm import tqdm 

from scripts.utils import *

import warnings
warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def get_labeled_datasets(config, dataset_config, fold, subset, spc):
    snid_name = dataset_config['dict_columns']['snid']
    target_name = dataset_config['dict_columns']['label']
    data_name = config['data_name']
    if spc != 'all':
        data_name += f'_{spc}'

    inv_dict_classes = {v: k for k, v in dataset_config['dict_mapping_classes'].items()}

    df_lc = []
    df_obj_label = []
    path_objects = glob.glob(f"{dataset_config['path_data']}/{config['data_name']}/fold_{fold}/{data_name}/{subset}/*")
    for path_object in path_objects:
        path_files = glob.glob('{}/*'.format(path_object))
        for path_file in path_files:
            df = pd.read_pickle(path_file)
            df_lc.append(df)
            df_obj_label.append(pd.DataFrame({
                snid_name: df[snid_name],
                target_name: df[target_name].replace(inv_dict_classes),
                'label_int': df[target_name],
            }))

        if subset == 'test':
            break

    df_obj_label = pd.concat(df_obj_label).reset_index(drop=True)
    df_lc = pd.concat(df_lc).reset_index(drop=True)
    df_lc = df_lc.explode('lc_data')
    df_lc[['mjd', 'flux', 'flux_err']] = pd.DataFrame(df_lc['lc_data'].tolist(), 
                                                      index=df_lc.index)
    
    # I removed the negative error magnitudes because it doesn't make sense
    df_lc = df_lc[df_lc['flux_err'] >= 0]     
    return df_lc, df_obj_label 


def get_dataset(config, dataset_config, fold, subset, spc):
    print('Data Loading...')
    df_lc, df_obj_label = get_labeled_datasets(config, dataset_config, fold, subset, spc)
    return df_lc, df_obj_label 

def min_max_normalize_lc(group, dict_columns):
    for col in [dict_columns['flux'], dict_columns['flux_err'], dict_columns['mjd']]:
        group[col] = (group[col] - group[col].min()) / (group[col].max() - group[col].min())
    return group

def get_normalization(df, norm_name, dict_columns):
    if norm_name == 'minmax_by_obj':
        return df.groupby([dict_columns['snid']]).apply(min_max_normalize_lc, dict_columns=dict_columns).reset_index(drop=True)
    else:
        raise 'The selected normalization has not been implemented...'

def process_shard(args):
    index, df, counter, lock, df_obj_label, config, dataset_config, saved_parent_dir = args
    dict_columns = dataset_config['dict_columns']
    df = get_normalization(df, config['norm_name'], dict_columns)
    
    path_save_tmp = saved_parent_dir + f'/temp_shards/shard_{index}'
    os.makedirs(path_save_tmp, exist_ok=True)
    output_pattern = f"{path_save_tmp}/imgs_lc-%06d.tar.gz"
    
    with wds.ShardWriter(output_pattern, maxcount=config['samples_per_shard']) as sink:
        for i, obj_id in enumerate(df[dict_columns['snid']].unique()):
            obj_df = df[df[dict_columns['snid']] == obj_id]
            sample_key = f"shard{index}_lc{i:06d}"
            sample = {
                "__key__": sample_key, 
                "ID.txt": str(obj_id)
                }
            label = df_obj_label[df_obj_label[dict_columns['snid']] == obj_id]['label_int'].iloc[0] 
            sample["label.cls"] = label
            
            # Generar la imagen
            fig_params = config['fig_params']
            plt.figure(figsize=fig_params['figsize'])
            plt.errorbar(obj_df[dict_columns['mjd']], 
                         obj_df[dict_columns['flux']],
                         yerr=obj_df[dict_columns['flux_err']] if config['use_err'] else None,
                         color=list(fig_params['colors'].values())[-1], 
                         fmt=fig_params['fmt'], 
                         alpha=fig_params['alpha'], 
                         markersize=fig_params['markersize'],
                         linewidth=fig_params['linewidth'])
            plt.axis('off')
            plt.ylim(fig_params['ylim'])
            buf = io.BytesIO()
            plt.savefig(buf, format='png', pad_inches=0)
            plt.close()
            buf.seek(0)
            image = Image.open(buf).convert('RGB')
            image_np = np.array(image)
            sample["pixel_values.pth"] = torch.tensor(np.array([image_np]))

            # Guardar el sample en el shard
            sink.write(sample)
            
            # Actualizar la barra de progreso
            with lock:
                counter.value += 1
    
    print(f"Shard {index} processed and saved in {path_save_tmp}")

@hydra.main(config_path=os.getenv("HYDRA_CONFIG_PATH", "../configs"),
            config_name=os.getenv("HYDRA_CONFIG_NAME", "create_imgs_config"), 
            version_base=None)

def main(config: DictConfig) -> None:
    dataset_config = load_yaml('configs/datasets_config.yaml')[config['data_name']]
    saved_parent_dir = f"data/images/{config['data_name']}"

    dataset_config.update({
        'name_dataset': config['data_name'],
        'num_classes': len(dataset_config['dict_mapping_classes']),
        'num_channels': 3,
    })

    params = config['fig_params']
    final_folder_name = f"{config['norm_name']}_{params['figsize'][0]}_m{params['markersize']}_l{params['linewidth']}"

    for spc in dataset_config['list_spc']:
        for subset in config['subsets']:
            for fold in config['list_folds']:
                df_lc, df_obj_label = get_dataset(config, 
                                                  dataset_config,
                                                  fold=fold,
                                                  subset=subset,
                                                  spc=spc)
                
                # Generate chunk of the dataset to multiprocess it
                print('Preparing multiprocessing...')
                dict_columns = dataset_config['dict_columns']
                ids = df_obj_label[dict_columns['snid']].unique()
                num_ids = len(ids)
                chunks_df = []
                for start in range(0, num_ids, config['samples_per_shard']):
                    end = start + config['samples_per_shard']
                    chunk_ids = ids[start:end]
                    chunk_df = df_lc[df_lc[dict_columns['snid']].isin(chunk_ids)]
                    chunks_df.append(chunk_df)
                print('We are ready...')

                # Generate the images and save in Shards
                manager = Manager()
                counter = manager.Value('i', 0)
                lock = manager.Lock()

                with Pool(processes=config['num_cores']) as pool:
                    args = [
                        (i, df, counter, lock, df_obj_label, config, dataset_config, saved_parent_dir) 
                        for i, df in enumerate(chunks_df)]

                    with tqdm(total=num_ids, desc="Shards processing...") as pbar:
                        for _ in pool.imap_unordered(process_shard, args):
                            pbar.update(counter.value - pbar.n)

                pool.close()
                pool.join()

                path_save_tmp = f"{saved_parent_dir}/temp_shards"
                path_save = f"{saved_parent_dir}/{spc}/{final_folder_name}"

                try:
                    os.makedirs(path_save, exist_ok=True)
                    save_yaml(config, path=saved_parent_dir)
                except:
                    pass

                path_save += f"/{subset}/fold_{fold}"
                join_shards(source_folder=path_save_tmp,
                            destination_folder=path_save)
                
        save_yaml(dataset_config, f"{saved_parent_dir}/{spc}/{final_folder_name}/data_info.yaml")

if __name__ == '__main__':

    main()
    

