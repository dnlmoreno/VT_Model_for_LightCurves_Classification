import os
import io
import glob
import hydra
import shutil
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

def get_plasticc_from_test(dataset_config, fold, subset):
    path_data = dataset_config['path_data']
    path_partition = dataset_config['path_partition']
    snid_name = dataset_config['dict_columns']['snid']
    target_name = dataset_config['dict_columns']['label']

    target_name = 'true_' + target_name
    df_metadata_test = pd.read_csv(f'{path_data}/unblinded_test_set_metadata.csv')
    df_metadata_test = df_metadata_test[~df_metadata_test[target_name].isin([991, 992, 993, 994])]
    df_obj_label = df_metadata_test.copy()[[snid_name, target_name]]
    df_metadata_test = df_metadata_test.drop([target_name], axis=1).set_index(snid_name)
    df_obj_label[target_name] = df_obj_label[target_name].replace(dataset_config['dict_mapping_real_classes'])
    df_obj_label['label_int'] = df_obj_label[target_name].map(dataset_config['dict_mapping_classes'])

    df_lc = []
    for i, path in enumerate(glob.glob(f'{path_data}/test_set_batch*')):
        df = pd.read_parquet(path)
        df = df[df[snid_name].isin(df_obj_label[snid_name])]
        df_lc.append(df)

    df_partitions = pd.read_parquet(path_partition)
    if 'test' == subset:
        partition = subset
    else:
        partition = f'{subset}_{fold}'
    ids_partition = df_partitions[df_partitions['subset'] == partition][snid_name].values

    df_lc = df_lc[df_lc[snid_name].isin(ids_partition)]
    df_obj_label = df_obj_label[df_obj_label[snid_name].isin(ids_partition)]

    return df_lc, df_obj_label


def get_plasticc(dataset_config, fold, subset):
    path_data = dataset_config['path_data']
    path_partition = dataset_config['path_partition']
    snid_name = dataset_config['dict_columns']['snid']
    target_name = dataset_config['dict_columns']['label']

    # Data and partition
    if 'train' == subset or 'val' == subset:
        df_metadata_training = pd.read_csv(f'{path_data}/training_set_metadata.csv')
        df_obj_label = df_metadata_training.copy()[[snid_name, target_name]]
        df_metadata_training = df_metadata_training.drop([target_name], axis=1).set_index(snid_name)
        df_obj_label[target_name] = df_obj_label[target_name].replace(dataset_config['dict_mapping_real_classes'])
        df_obj_label['label_int'] = df_obj_label[target_name].map(dataset_config['dict_mapping_classes'])

        df_partitions = pd.read_parquet(f'{path_partition}/train_val/stratified_5fold_splits.parquet')
        ids_partition = df_partitions[df_partitions['subset'] == f'{subset}_{fold}'][snid_name].values
        df_lc = pd.read_csv(f'{path_data}/training_set.csv')
        df_lc = df_lc[df_lc[snid_name].isin(ids_partition)]

    if 'test' == subset:
        target_name = 'true_' + target_name
        df_metadata_test = pd.read_csv(f'{path_data}/unblinded_test_set_metadata.csv')
        df_metadata_test = df_metadata_test[~df_metadata_test[target_name].isin([991, 992, 993, 994])]
        df_obj_label = df_metadata_test.copy()[[snid_name, target_name]]
        df_metadata_test = df_metadata_test.drop([target_name], axis=1).set_index(snid_name)
        df_obj_label[target_name] = df_obj_label[target_name].replace(dataset_config['dict_mapping_real_classes'])
        df_obj_label['label_int'] = df_obj_label[target_name].map(dataset_config['dict_mapping_classes'])

        df_lc = []
        for i, path in enumerate(glob.glob(f'{path_data}/test_set_batch*')):
            df = pd.read_parquet(path)
            df = df[df[snid_name].isin(df_obj_label[snid_name])]
            df_lc.append(df)

    return df_lc, df_obj_label

def get_elasticc_1(dataset_config, fold, subset):
    path_data = dataset_config['path_data']
    path_partition = dataset_config['path_partition']
    snid_name = dataset_config['dict_columns']['snid']
    target_name = dataset_config['dict_columns']['label']
    band_name = dataset_config['dict_columns']['band']

    ATAT_partition = pd.read_parquet(f'{path_partition}/partitions.parquet')
    if subset == 'test': 
        subset_name = f'{subset}'
    else:
        subset_name = f'{subset}_{fold}'
    ids_partition = ATAT_partition[ATAT_partition['subset'] == subset_name][snid_name].values

    df_lc = []
    df_obj_label = []
    for i, path in enumerate(glob.glob(f'{path_data}/lc_*')):
        class_name = path.split('/')[-1].split('.')[0].split('lc_')[-1]
        class_name = dataset_config['dict_mapping_real_classes'][class_name]
        df = pd.read_parquet(path)
        df_lc.append(df)
        unique_objs = df[snid_name].unique()
        df_obj_label.append(pd.DataFrame({
            snid_name: unique_objs,
            target_name: [class_name] * len(unique_objs),
            'label_int': [dataset_config['dict_mapping_classes'][class_name]] * len(unique_objs),
        }))

    df_lc = pd.concat(df_lc)
    df_lc = df_lc[df_lc[snid_name].isin(ids_partition)]
    df_lc[band_name] = df_lc[band_name].replace(dataset_config['all_bands'])
    df_obj_label = pd.concat(df_obj_label)
    df_obj_label = df_obj_label[df_obj_label[snid_name].isin(ids_partition)]

    return df_lc, df_obj_label

def get_dataset(config, dataset_config, fold, subset):
    print('Data Loading...')
    if config['data_name'] == 'plasticc':
        if config['extra_plasticc']['only_test']['use']:
            dataset_config['path_partition'] = config['extra_plasticc']['path_partition']
            df_lc, df_obj_label = get_plasticc_from_test(dataset_config, fold, subset)
        else:
            df_lc, df_obj_label = get_plasticc(dataset_config, fold, subset)
    elif config['data_name'] == 'elasticc_1':
        df_lc, df_obj_label = get_elasticc_1(dataset_config, fold, subset)
    else:
        raise f"We don't have the implementation for the dataset called {config['data_name']}"
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
                                color=fig_params['colors'][j],
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

    test_fold = 0
    for subset in config['subsets']:
        folds_to_use = [test_fold] if subset == 'test' else config['list_folds']

        for fold in folds_to_use:
            df_lc, df_obj_label = get_dataset(config, 
                                              dataset_config,
                                              fold=fold,
                                              subset=subset)
            
            # Generate chunk of the dataset to multiprocess it
            dict_columns = dataset_config['dict_columns']
            ids = df_obj_label[dict_columns['snid']].unique()
            num_ids = len(ids)
            chunks_df = []
            for start in range(0, num_ids, config['samples_per_shard']):
                end = start + config['samples_per_shard']
                chunk_ids = ids[start:end]
                chunk_df = df_lc[df_lc[dict_columns['snid']].isin(chunk_ids)]
                chunks_df.append(chunk_df)

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
            path_save = f"{saved_parent_dir}/{config['final_folder_name']}"

            try:
                os.makedirs(path_save, exist_ok=True)
                save_yaml(config, path=saved_parent_dir)
            except:
                pass

            path_save += f"/{subset}"
            if subset != 'test':
                path_save += f"/fold_{fold}"
            join_shards(source_folder=path_save_tmp,
                        destination_folder=path_save)

if __name__ == '__main__':

    main()
    

