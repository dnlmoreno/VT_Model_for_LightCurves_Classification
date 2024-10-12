
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

def create_overlay_images(obj_df, config, dataset_config):
    dict_columns = dataset_config['dict_columns']
    fig_params = config['fig_params']

    fig = plt.figure(figsize=(fig_params['figsize']))
    ax = fig.add_subplot(1, 1, 1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    for j in range(len(dataset_config['all_bands'])):
        band_data = obj_df[obj_df[dict_columns['band']] == j]

        if band_data.empty:
            ax.add_patch(patches.Rectangle((0, 0), 1, 1, color='white', transform=ax.transAxes))
        else:
            ax.errorbar(band_data[dict_columns['mjd']], 
                        band_data[dict_columns['flux']], 
                        yerr=band_data[dict_columns['flux_err']] if config['use_err'] else None,
                        color=fig_params['colors'][j+2],
                        fmt=fig_params['fmt'], 
                        alpha=fig_params['alpha'], 
                        markersize=fig_params['markersize'], 
                        linewidth=fig_params['linewidth'])

        ax.set_ylim(fig_params['ylim'])
        ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).convert('RGB')
    return image

def create_2grid_images(obj_df, config, dataset_config):
    dict_columns = dataset_config['dict_columns']
    fig_params = config['fig_params']

    fig, axs = plt.subplots(2, 1, figsize=(2.24, 2.24))  # Dos filas y tres columnas
    for band_key, j in dataset_config['all_bands'].items():
        #row, col = divmod(j, 2)
        row = j
        band_data = obj_df[obj_df[dict_columns['band']] == j]

        if band_data.empty:
            axs[row].add_patch(patches.Rectangle((0, 0), 1, 1, color='white', transform=axs[row].transAxes))
        else:
            axs[row].errorbar(band_data[dict_columns['mjd']], 
                              band_data[dict_columns['flux']], 
                              yerr=band_data[dict_columns['flux_err']] if config['use_err'] else None,
                              color=fig_params['colors'][j+2],
                              fmt=fig_params['fmt'], 
                              alpha=fig_params['alpha'], 
                              markersize=fig_params['markersize'], 
                              linewidth=fig_params['linewidth'])

        axs[row].set_ylim(fig_params['ylim'])
        axs[row].axis('off')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    # Cuadrado grande (borde exterior)
    rect = patches.Rectangle((0, 0), 1, 1, linewidth=1.5, edgecolor='black', facecolor='none', transform=fig.transFigure)
    fig.add_artist(rect)

    # Línea entre las filas
    rect = patches.Rectangle((0, 0.5), 1, 0, linewidth=0.3, edgecolor='black', facecolor='none', transform=fig.transFigure)
    fig.add_artist(rect)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).convert('RGB')
    return image


def get_alcock_multiband(config, dataset_config, fold, subset, spc):
    path_data = dataset_config['path_data']
    path_partition = dataset_config['path_partition']
    snid_name = dataset_config['dict_columns']['snid']
    target_name = dataset_config['dict_columns']['label']
    band_name = dataset_config['dict_columns']['band']

    partitions = pd.read_parquet(f'{path_partition}/partitions.parquet')
    partition = partitions[
        (partitions.subset == subset) &
        (partitions.fold == fold) &
        (partitions.spc == str(spc))
        ]
    id_lcs = partition[snid_name].values

    inv_dict_classes = {v: k for k, v in dataset_config['dict_mapping_classes'].items()}

    df_lc = []
    df_obj_label = []
    for lcid in id_lcs:
        df_B = pd.read_csv(f'{path_data}/B/{lcid}.dat')
        df_B['lcid'] = lcid
        df_B['band'] = 0
        df_lc.append(df_B)

        df_R = pd.read_csv(f'{path_data}/R/{lcid}.dat')
        df_R['lcid'] = lcid
        df_R['band'] = 1
        df_lc.append(df_R)

        label_int = partition[partition[snid_name] == lcid][target_name].iloc[0]
        df_obj_label.append(pd.DataFrame({
            snid_name: [lcid],
            target_name: [inv_dict_classes[label_int]],
            'label_int': [label_int],
        }))

    df_lc = pd.concat(df_lc)
    df_lc = df_lc[df_lc['err'] >= 0] 
    df_obj_label = pd.concat(df_obj_label)

    return df_lc, df_obj_label 


def get_labeled_astromer_datasets(config, dataset_config, fold, subset, spc):
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
    if config['data_name'] in ['alcock', 'atlas', 'ogle']: 
        df_lc, df_obj_label = get_labeled_astromer_datasets(config, dataset_config, fold, subset, spc)
    elif config['data_name'] == 'alcock_multiband':
        df_lc, df_obj_label = get_alcock_multiband(config, dataset_config, fold, subset, spc)
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
            if config['data_name'] in ['alcock', 'atlas', 'ogle']: 
                fig_params = config['fig_params']
                plt.figure(figsize=fig_params['figsize'])
                plt.errorbar(obj_df[dict_columns['mjd']], 
                            obj_df[dict_columns['flux']],
                            yerr=obj_df[dict_columns['flux_err']] if config['use_err'] else None,
                            color=fig_params['colors'][3], 
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
            else:
                if config['input_type'] == '2grid': 
                    image = create_2grid_images(obj_df, config, dataset_config)
                elif config['input_type'] == 'overlay':
                    image = create_overlay_images(obj_df, config, dataset_config)
                else: 
                    raise f"The input_type called: {config['input_type']} is not implemented"

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

    #for err in [False, True]:
    for markersize in [1.0, 2.0, 3.0]:
        for linewidth in [1.0, 2.0]:
            for input_name in ['overlay', '2grid']: # '2grid',
                params['markersize'] = markersize
                params['linewidth'] = linewidth
                config['input_type'] = input_name
                #config['use_err'] = err

                final_folder_name = f"{config['norm_name']}_{params['figsize'][0]}_m{params['markersize']}_l{params['linewidth']}_e{config['use_err']}_{config['input_type']}"

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
    

