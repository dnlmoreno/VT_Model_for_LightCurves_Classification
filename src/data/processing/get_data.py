import pandas as pd
import glob
import os
import logging

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

def get_elasticc_1(dataset_config, debug):
    data_dir = dataset_config['data_dir']
    snid_name = dataset_config['dict_columns']['snid']
    mjd_name = dataset_config['dict_columns']['mjd']
    band_name = dataset_config['dict_columns']['band']
    label_name = dataset_config['dict_columns']['label']
    filtered_values = [value for key, value in dataset_config['dict_columns'].items() if key != label_name]

    df_lc = []
    path_chunks = glob.glob(f'{data_dir}/lc_*')
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

        if debug:
            if i == 2: break

    return df_lc


def get_macho(dataset_config, multiband, debug):
    data_dir = dataset_config['data_dir']
    df_lc = []
    lcids_B = set([os.path.splitext(os.path.basename(file))[0] for file in glob.glob(f'{data_dir}/B/*')])
    lcids_R = set([os.path.splitext(os.path.basename(file))[0] for file in glob.glob(f'{data_dir}/R/*')])
    lcids = list(lcids_B.union(lcids_R))
    for i, lcid in enumerate(lcids, start=1):
        path_R = f'{data_dir}/R/{lcid}.dat'
        if os.path.exists(path_R) and os.path.getsize(path_R) > 0:
            df_R = pd.read_csv(path_R)
            df_R['lcid'] = lcid
            df_R['band'] = dataset_config['all_bands']['R']
            df_lc.append(df_R)

        if multiband:
            path_B = f'{data_dir}/B/{lcid}.dat'
            if os.path.exists(path_B) and os.path.getsize(path_B) > 0:
                df_B = pd.read_csv(path_B)
                df_B['lcid'] = lcid
                df_B['band'] = dataset_config['all_bands']['B']
                df_lc.append(df_B)

        if debug:
            if i == 2000: break

        if i % 2000 == 0:
            logging.info(f' -→ Opening chunk {i}/{len(lcids)}')
    logging.info(f' -→ Opening chunk {i}/{len(lcids)}')

    df_lc = pd.concat(df_lc)
    df_lc = df_lc[df_lc['err'] >= 0] 
    return df_lc.reset_index(drop=True)

def get_dataset(dataset_config, name_dataset, debug):
    logging.info('🔄 Data Loading...')
    if name_dataset == 'elasticc_1': 
        df_lc = get_elasticc_1(dataset_config, debug=debug)
    elif name_dataset == 'macho_multiband':
        df_lc = get_macho(dataset_config, multiband=True, debug=debug)
    elif name_dataset == 'macho':
        df_lc = get_macho(dataset_config, multiband=False, debug=debug)
    return df_lc