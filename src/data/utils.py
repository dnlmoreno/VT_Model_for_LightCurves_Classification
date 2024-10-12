import pandas as pd
import glob
import os
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io

from PIL import Image

def get_first_last_detection(group):
    detection_mjd = group[group['PHOTFLAG'].isin([4096, 6144])]['MJD']
    if not detection_mjd.empty:
        return pd.Series({
            'first_detection_mjd': detection_mjd.min(),
            'last_detection_mjd': detection_mjd.max()
        })
    else:
        return pd.Series({
            'first_detection_mjd': float('nan'),
            'last_detection_mjd': float('nan')
        })

def get_elasticc_1(path_data, dataset_config):
    snid_name = dataset_config['dict_columns']['snid']
    mjd_name = dataset_config['dict_columns']['mjd']
    band_name = dataset_config['dict_columns']['band']
    label_name = dataset_config['dict_columns']['label']
    filtered_values = [value for key, value in dataset_config['dict_columns'].items() if key != label_name]

    df_lc = []
    path_chunks = glob.glob(f'{path_data}/raw/lc_*')
    for i, path in enumerate(path_chunks):
        logging.info(f' -→ Opening chunk {i + 1:02}/{len(path_chunks)}')
        class_name = path.split('/')[-1].split('.')[0].split('lc_')[-1]
        class_name = dataset_config['dict_mapping_real_classes'][class_name]
        df = pd.read_parquet(path)

        detection_mjd = df.groupby(snid_name).apply(get_first_last_detection)
        df = df.merge(detection_mjd, left_on=snid_name, right_index=True)
        df = df[(df[mjd_name] >= df['first_detection_mjd'] - 30) & 
                (df[mjd_name] <= df['last_detection_mjd'])]
        df = df.drop(columns=['first_detection_mjd', 'last_detection_mjd'])
        df = df[df['PHOTFLAG'] != 1024]

        df[band_name] = df[band_name].replace(dataset_config['all_bands'])
        df_lc.append(df[filtered_values])

        if i == 1:
            break

    df_lc = pd.concat(df_lc)
    return df_lc


def get_alcock_multiband(path_data, dataset_config):
    df_lc = []
    lcids_B = set([os.path.splitext(os.path.basename(file))[0] for file in glob.glob(f'{path_data}/raw/B/*')])
    lcids_R = set([os.path.splitext(os.path.basename(file))[0] for file in glob.glob(f'{path_data}/raw/R/*')])
    lcids = list(lcids_B.union(lcids_R))
    for i, lcid in enumerate(lcids, start=1):
        path_B = f'{path_data}/raw/B/{lcid}.dat'
        if os.path.exists(path_B) and os.path.getsize(path_B) > 0:
            df_B = pd.read_csv(path_B)
            df_B['lcid'] = lcid
            df_B['band'] = 0
            df_lc.append(df_B)

        # Verifica si el archivo para la banda R existe antes de cargarlo
        path_R = f'{path_data}/raw/R/{lcid}.dat'
        if os.path.exists(path_R) and os.path.getsize(path_R) > 0:
            df_R = pd.read_csv(path_R)
            df_R['lcid'] = lcid
            df_R['band'] = 1
            df_lc.append(df_R)

        if i % 2000 == 0:
            logging.info(f' -→ Opening chunk {i}/{len(lcids)}')
    logging.info(f' -→ Opening chunk {i}/{len(lcids)}')

    df_lc = pd.concat(df_lc)
    df_lc = df_lc[df_lc['err'] >= 0] 
    return df_lc.reset_index(drop=True)


def get_dataset(path_data, dataset_config, name_dataset):
    logging.info('🔄 Data Loading...')
    if name_dataset == 'elasticc_1': 
        df_lc = get_elasticc_1(path_data, dataset_config)
    elif name_dataset == 'alcock_multiband':
        df_lc = get_alcock_multiband(path_data, dataset_config)
    return df_lc


def min_max_normalize_lc(group, dict_columns):
    for col in [dict_columns['flux'], dict_columns['flux_err'], dict_columns['mjd']]:
        group[col] = (group[col] - group[col].min()) / (group[col].max() - group[col].min())
    return group

def get_normalization(df, norm_name, dict_columns):
    if norm_name == 'minmax_by_obj':
        return df.groupby([dict_columns['snid']]).apply(min_max_normalize_lc, dict_columns=dict_columns).reset_index(drop=True)
    else:
        raise 'The selected normalization has not been implemented...'


def create_2grid_images(obj_df, config, dataset_config):
    dict_columns = dataset_config['dict_columns']
    fig_params = config['imgs_params']['fig_params']

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
                              yerr=band_data[dict_columns['flux_err']] if config['imgs_params']['use_err'] else None,
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

def create_overlay_images(obj_df, config, dataset_config):
    dict_columns = dataset_config['dict_columns']
    fig_params = config['imgs_params']['fig_params']

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
                        yerr=band_data[dict_columns['flux_err']] if config['imgs_params']['use_err'] else None,
                        color=fig_params['colors'][j], #[j+2],
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

def create_6grid_images(obj_df, config, dataset_config):
    dict_columns = dataset_config['dict_columns']
    fig_params = config['imgs_params']['fig_params']

    fig, axs = plt.subplots(2, 3, figsize=(fig_params['figsize']))  # Dos filas y tres columnas
    for j in range(len(dataset_config['all_bands'])):
        row, col = divmod(j, 3)
        band_data = obj_df[obj_df[dict_columns['band']] == j]

        if band_data.empty:
            axs[row, col].add_patch(patches.Rectangle((0, 0), 1, 1, color='white', transform=axs[row, col].transAxes))
        else:
            axs[row, col].errorbar(band_data[dict_columns['mjd']], 
                                   band_data[dict_columns['flux']], 
                                   yerr=band_data[dict_columns['flux_err']] if config['imgs_params']['use_err'] else None,
                                   color=fig_params['colors'][j],
                                   fmt=fig_params['fmt'], 
                                   alpha=fig_params['alpha'], 
                                   markersize=fig_params['markersize'], 
                                   linewidth=fig_params['linewidth'])

        axs[row, col].set_ylim(fig_params['ylim'])
        axs[row, col].axis('off')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    # Agregar rectángulos para las columnas
    for col in range(3):
        rect = patches.Rectangle((col/3, 0), 1/3, 1, linewidth=0.3, edgecolor='black', facecolor='none', transform=fig.transFigure)
        fig.add_artist(rect)

    # Agregar rectángulos para las filas
    for row in range(2):
        rect = patches.Rectangle((0, row/2), 1, 0.5, linewidth=0.3, edgecolor='black', facecolor='none', transform=fig.transFigure)
        fig.add_artist(rect)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).convert('RGB')
    return image