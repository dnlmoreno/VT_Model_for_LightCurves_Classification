import pandas as pd
import numpy as np
import glob
import os

from sklearn.preprocessing import QuantileTransformer
from scripts.utils import *

data_name = 'elasticc_1'

dataset_config = load_yaml('configs/datasets_config.yaml')[data_name]
path_data = dataset_config['path_data']
path_partition = dataset_config['path_partition']
col_metadata = dataset_config['col_metadata']
col_snid = dataset_config['dict_columns']['snid']
path_save = 'data/metadata'

df_metadata = []
for i, path in enumerate(glob.glob(f'{path_data}/features_*')):
    df_temp = pd.read_parquet(path)
    missing_cols = set(col_metadata) - set(df_temp.columns)
    for col in missing_cols:
        df_temp[col] = -9999
    df_temp = df_temp[[col_snid] + col_metadata]
    df_metadata.append(df_temp)

df_metadata = pd.concat(df_metadata, ignore_index=True).set_index(col_snid)

df_partitions = pd.read_parquet(f'{path_partition}/partitions.parquet')
df_metadata_train = df_metadata[df_metadata.index.isin(df_partitions[df_partitions['subset'] == 'train_0'][col_snid])]
df_metadata_val = df_metadata[df_metadata.index.isin(df_partitions[df_partitions['subset'] == 'val_0'][col_snid])]

print(df_metadata_train.shape)
print(df_metadata_val.shape)

# Guardar los índices (que son los IDs) antes de transformar
train_index = df_metadata_train.index
val_index = df_metadata_val.index

# Aplicar QuantileTransformer
qt = QuantileTransformer(n_quantiles=1000, random_state=0, output_distribution='normal')
qt.fit(df_metadata_train)

transformed_train = qt.transform(df_metadata_train)
transformed_val = qt.transform(df_metadata_val)

# Crear DataFrames a partir de los datos transformados y restaurar los índices (IDs)
df_metadata_train = pd.DataFrame(transformed_train, columns=df_metadata_train.columns, index=train_index)
df_metadata_val = pd.DataFrame(transformed_val, columns=df_metadata_val.columns, index=val_index)

# Concatenar los DataFrames de entrenamiento y validación
df_metadata_processed = pd.concat([df_metadata_train, df_metadata_val])

os.makedirs(path_save)
df_metadata_processed.to_parquet(f'{path_save}/metadata_qt_{data_name}.parquet')









