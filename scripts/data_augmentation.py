import os
import pandas as pd
import numpy as np
import random
from multiprocessing import Pool

# Define the function to generate all augmented data for a single astronomical object
def augment_object(args):
    df, i = args
    augmented_data = []
    for index, row in df.iterrows():
        sample = np.random.normal(loc=row['flux'], scale=row['flux_err'], size=1)
        augmented_data.append({
            'object_id': f"{int(row['object_id'])}_{i}", #
            'mjd': row['mjd'],
            'passband': int(row['passband']),
            'flux': sample[0],
            'flux_err': row['flux_err'],
            'detected': int(row['detected']),
        })
    return pd.DataFrame(augmented_data)

#for i, sample in enumerate(samples, 1):  # Start enumeration at 1 for augmented sample ID

if __name__ == '__main__':

    aug_name = '30000'
    folder_save = f'oversampling_{aug_name}_class'
    path_partition = 'data/lightcurves/plasticc/partitions'
    df_partitions = pd.read_parquet(f'{path_partition}/train_val/stratified_5fold_splits.parquet')

    df_train_0 = df_partitions[df_partitions.subset == 'train_0']
    aug_name = df_train_0.groupby('target').count()['object_id'].max() if aug_name == 'max' else int(aug_name),
    num_aug_by_class = aug_name - df_train_0.groupby('target').count()['object_id']

    os.makedirs(f'data/lightcurves/plasticc/augmented_data/{folder_save}/fold_0', exist_ok=True)

    path_data = 'data/lightcurves/plasticc/raw'
    df_lc = pd.read_csv(f'{path_data}/training_set.csv')

    for label, num_aug in num_aug_by_class.items():
        print(f'We are augmentating the {label} class:')
        if num_aug > 0:
            object_ids = list(df_train_0[df_train_0.target == label].object_id.values)
            #num_total = len(object_ids) + num_aug

            if num_aug <= len(object_ids):
                augmented_ids = object_ids[:num_aug]
            else:
                # Calculate how many times to repeat the original IDs
                num_repeats = num_aug // len(object_ids)
                remainder = num_aug % len(object_ids)
                
                # Repeat the list and add the remaining IDs
                augmented_ids = np.tile(object_ids, num_repeats).tolist()
                if remainder > 0:
                    random_remainder = random.sample(object_ids, remainder)
                    augmented_ids += random_remainder

            #print('object_ids:', len(object_ids))
            #print('augmented_ids:', len(augmented_ids))
            args_list = [(df_lc[df_lc['object_id'] == obj_id], i) for i, obj_id in enumerate(augmented_ids)]

            print(f'- Multiprocessing started...')
            with Pool(processes=20) as pool:
                results = pool.map(augment_object, args_list)

            df_augmented = pd.concat(results, ignore_index=True)
            df_augmented.to_parquet(f'data/lightcurves/plasticc/augmented_data/{folder_save}/fold_0/{label}_{num_aug}.parquet')

            print(f'- File was saved.')
        else:
            print(f'- There is not augmentations for {label}.')