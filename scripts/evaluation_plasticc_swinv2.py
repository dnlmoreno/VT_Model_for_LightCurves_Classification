import glob
import torch
import pandas as pd
import numpy as np
import lightning as L
import importlib

from typing import Optional
from lightning.pytorch import LightningDataModule, LightningModule

from src.models.LitModels.swinv2 import LitModel
from src.data.LitData import LitData

import sys

from scripts.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dict_order_classes = {
    'plasticc': ['Single u-lens', 'TDE', 'Eclip. Binary', 'SNII', 'SNIax', 'Mira', 
                 'SNIbc', 'Kilonova', 'M-dwarf', 'SNIa-91bg', 'AGN', 'SNIa', 
                 'RR lyrae', 'SLSN-I'],
    'elasticc_1': ['CART', 'Iax', '91bg', 'Ia', 'Ib/c', 'II', 'SN-like/Other', 'SLSN', 
                   'PISN', 'TDE', 'ILOT', 'KN', 'M-dwarf Flare', 'uLens', 'Dwarf Novae', 
                   'AGN', 'Delta Scuti', 'RR Lyrae', 'Cepheid', 'EB']  
}

def load_model(data_info, config):
    model_name = config['model_name']
    LitModel_module = importlib.import_module(f"src.models.LitModels.{model_name}")
    model = getattr(LitModel_module, 'LitModel')(data_info, **config)
    return model

def plasticc_log_loss(y_true, y_pred_prob, relative_class_weights=None):
    """
    Implementation of weighted log loss used for the Kaggle challenge
    """

    # sanitize predictions
    epsilon = sys.float_info.epsilon # this is machine dependent but essentially prevents log(0)
    y_pred_prob = np.clip(y_pred_prob, epsilon, 1.0 - epsilon)
    y_pred_prob = y_pred_prob / np.sum(y_pred_prob, axis=1)[:, np.newaxis]

    y_pred_prob = np.log(y_pred_prob)
    # multiplying the arrays is equivalent to a truth mask as y_true only contains zeros and ones
    class_logloss = []
    for i in range(y_pred_prob.shape[1]):
        # average column wise log loss with truth mask applied
        result = np.average(y_pred_prob[:, i][y_true[:, i] == 1])
        class_logloss.append(result)

    return -1 * np.average(class_logloss, weights=relative_class_weights)

def predict(dataset: LightningDataModule, 
            loaded_model: LightningModule, 
            path_save_metrics: Optional[str] = None,
            weights = None):

    trainer = L.Trainer(logger=None)
    batches_output = trainer.predict(loaded_model, dataloaders=dataset.predict_dataloader())
   
    # Handling output
    sort_name_classes = list(sort_dict_by_value(dataset.inv_mapping_classes).values())
    df_list = [batch_to_df(batch, sort_name_classes) for batch in batches_output]
    df_windows_proba = pd.concat(df_list, ignore_index=True)

    #num_classes = len(dataset.inv_mapping_classes)
    #class_counts = class_counts['val']
    #loss = weigthed_multi_class_logs_loss(
    #    y_pred_prob = df_windows_proba['y_pred_prob'], 
    #    y_true = df_windows_proba['y_true'], , 
    #    num_classes = num_classes, 
    #    class_counts = class_counts, 
    #    weights = weights,
    #)

    weights = np.ones(df_windows_proba['y_true'].nunique())
    y_true_one_hot = pd.get_dummies(df_windows_proba['y_true'])
    loss = plasticc_log_loss(
        y_true=y_true_one_hot.values, 
        y_pred_prob=df_windows_proba.drop(['id', 'y_pred', 'y_true'], axis=1).values, 
        relative_class_weights=weights,
        )

    # Probabilities by windows
    df_windows_proba['y_pred'] = df_windows_proba['y_pred'].replace(dataset.inv_mapping_classes)
    df_windows_proba['y_true'] = df_windows_proba['y_true'].replace(dataset.inv_mapping_classes)

    # Metrics
    dict_metrics = dict()
    dict_metrics['LC'] = calculate_metrics(y_true=df_windows_proba['y_true'],
                                           y_pred=df_windows_proba['y_pred'])
    dict_metrics['plasticc_log_loss'] = loss

    #order_classes = sorted(dataset.inv_mapping_classes.values())
    order_classes = dict_order_classes[dataset.name_dataset]
    single_confusion_matrix(y_true=df_windows_proba['y_true'], 
                            y_pred=df_windows_proba['y_pred'], 
                            order_classes=order_classes,
                            path_save='elasticc_1_mal_pero_mejor.jpg')

    return dict_metrics


if __name__ == "__main__":

    config = {
        'mlflow_dir': 'results_harvard_2/ml-runs',

        'checkpoint': {
            'use': True,
            'exp_name': 'ft_classification/elasticc_1/testing',
            'run_name': '2024-09-30_12-39-43',
            'results_dir': 'results_harvard_2',
        },

        'loader': {
            'fold': 0
            }
    }

    ckpt_dir = handle_ckpt_dir(config, fold=config['loader']['fold'])
    ckpt_model = sorted(glob.glob(ckpt_dir + "/*.ckpt"))[-1]

    # Data
    hparams = load_yaml(f'{ckpt_dir}/hparams.yaml')
    data_info = load_yaml(path='./{}/data_info.yaml'.format(hparams['loader']['path_data']))
    print(hparams)

    hparams['training']['batch_size'] = 500
    hparams['checkpoint'] = {
        'use': True,
        'exp_name': 'ft_classification/elasticc_1/testing',
        'run_name': '2024-09-30_12-39-43',
        'results_dir': 'results_harvard_2',
    }

    hparams['pretrained_model'] = {
        'use': True,
        'path': "microsoft/swinv2-tiny-patch4-window16-256",
    }

    dataset = LitData(**hparams)
    dataset.prepare_data()
    dataset.setup('test')

    # Model
    #loaded_model = LitModel.load_from_checkpoint(checkpoint_path=ckpt_model, 
    #                                             map_location=device).eval()

    data_info = hparams.pop('data_info')
    loaded_model = load_model(data_info=data_info,
                              config=hparams)
    if os.path.exists(ckpt_model):
        loaded_model = load_checkpoint(loaded_model, ckpt_model)            
    else:
        raise FileNotFoundError(f"Checkpoint file not found at {ckpt_dir}")

    dict_metrics = predict(dataset, loaded_model)

    print(f"LC:\n{dict_metrics['LC']}\n")

    if dataset.name_dataset == 'plasticc_1':
        print(f"loss: {dict_metrics['plasticc_log_loss']}")

    #print(f"Avg windows:\n{dict_metrics['LCs']}")
