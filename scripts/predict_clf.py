import glob
import torch
import pandas as pd
import numpy as np
import lightning as L

from typing import Optional
from lightning.pytorch import LightningDataModule, LightningModule
from sklearn.metrics import f1_score

from scripts.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(dataset: LightningDataModule, 
            loaded_model: LightningModule, 
            path_save_metrics: Optional[str] = None):

    trainer = L.Trainer(logger=None)
    batches_output = trainer.predict(loaded_model, dataloaders=dataset.predict_dataloader())

    # Handling output
    sort_name_classes = list(sort_dict_by_value(dataset.dict_mapping_classes).values())
    df_list = [batch_to_df(batch, sort_name_classes) for batch in batches_output]
    df_proba = pd.concat(df_list, ignore_index=True)

    # Probabilities by windows
    df_proba['y_pred'] = df_proba['y_pred'].replace(dataset.inv_mapping_classes)
    df_proba['y_true'] = df_proba['y_true'].replace(dataset.inv_mapping_classes)

    # Metrics
    dict_metrics = dict()
    dict_metrics = calculate_metrics(y_true=df_proba['y_true'],
                                     y_pred=df_proba['y_pred'])

    # Save metrics
    if path_save_metrics is not None:
        with open(f'{path_save_metrics}/classification_report.txt', 'w') as file:
            file.write(dict_metrics)

        # Save confusion matrix
        order_classes = sorted(dataset.dict_mapping_classes.values())

        single_confusion_matrix(y_true=df_proba['y_true'], 
                                y_pred=df_proba['y_pred'], 
                                order_classes=order_classes, 
                                path_save=f'{path_save_metrics}/confusion_matrix.png')

    return dict_metrics


if __name__ == "__main__":

    config = {
        'mlflow_dir': 'ml-runs',

        'checkpoint': {
            'exp_name': 'classification/macho/testing',
            'run_name': '2024-07-13_18-06-43',
        },

        'loader': {
            'fold': 2
            }
    }

    ckpt_dir = handle_ckpt_dir(config, fold=config['loader']['fold'])
    ckpt_model = sorted(glob.glob(ckpt_dir + "/*.ckpt"))[-1]

    # Data
    hparams = load_yaml(f'{ckpt_dir}/hparams.yaml')
    dict_info_ds = load_yaml(path='./{}/dict_info.yaml'.format(hparams['loader']['path_data']))
    dataset = LitDataCLF(dict_info_ds, **hparams)
    dataset.setup('test')

    # Model
    loaded_model = LitModelCLF.load_from_checkpoint(checkpoint_path=ckpt_model, 
                                                    map_location=device).eval()

    dict_metrics = predict(dataset, loaded_model)

    print(f"Windows:\n{dict_metrics['Windows']}")
    print(f"Avg windows:\n{dict_metrics['LCs']}")
