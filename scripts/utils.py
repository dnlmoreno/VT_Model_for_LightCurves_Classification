import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import yaml
import mlflow
import logging
import shutil
import os

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

def load_yaml(path):
    with open(path, 'r') as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
    return args

def save_yaml(args, path):
    with open(path, 'w') as file:
        yaml.dump(args, file, sort_keys=False)

def sort_dict_by_value(d, reverse=False):
    return dict(sorted(d.items(), key=lambda x: x[0], reverse=reverse))

def batch_to_df(batch, sorted_classes):
    df_dict = {
        'id': batch['id'],
        'y_pred': batch['y_pred'],
        'y_true': batch['y_true'],
    }
    
    # Add columns for each class probability
    for i, class_label in enumerate(sorted_classes):
        df_dict[class_label] = batch['y_pred_prob'][:, i]

    return pd.DataFrame(df_dict)

def get_experiment_id_mlflow(exp_name):
    experiment = mlflow.get_experiment_by_name(exp_name)
    if experiment:
        experiment_id = experiment.experiment_id
        logging.info(f"Experiment ID for '{exp_name}': {experiment_id}")
    else:
        logging.info(f"Experiment '{exp_name}' not found.")
    return experiment_id

def get_run_id_mlflow(experiment_id, run_name):
    filter_string = f"tags.mlflow.runName = '{run_name}'"
    runs = mlflow.search_runs(experiment_ids=[experiment_id], filter_string=filter_string)
    if not runs.empty:  # Check if the DataFrame is empty
        run_id = runs.iloc[0].run_id  # Get the run ID of the first row
        logging.info(f"Run ID for '{run_name}': {run_id}")
    else:
        logging.info(f"Run '{run_name}' not found in experiment '{experiment_id}'.")
    return run_id

def handle_ckpt_dir(config, fold=None):
    mlflow.set_tracking_uri(f"file:{config['checkpoint']['results_dir']}/ml-runs")
    exp_name = f"{config['checkpoint']['exp_name']}"
    exp_id = get_experiment_id_mlflow(exp_name)
    ckpt_dir = '{}/ml-runs/{}'.format(config['checkpoint']['results_dir'], exp_id)

    run_name = config['checkpoint']['run_name']
    if 'ft_classification' in exp_name:
        run_name = 'Fold_{}_{}'.format(fold, run_name)  
        run_id = get_run_id_mlflow(exp_id, run_name)
        ckpt_dir += '/{}'.format(run_id)
    else:
        run_id = get_run_id_mlflow(exp_id, run_name)
        ckpt_dir += '/{}'.format(run_id) 

    ckpt_dir += '/artifacts/model'
    return ckpt_dir

def load_checkpoint(model, ckpt_model):
    try:
        checkpoint = torch.load(ckpt_model)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        logging.info("Weights were loaded correctly.")
    except FileNotFoundError:
        logging.error(f"Checkpoint file not found at {ckpt_model}")
        return None
    except Exception as e:
        logging.error("Failed to load weights due to an error: %s", e)
        return None

    # Checking for missing keys
    model_dict = model.state_dict()
    missing_keys = set(model_dict.keys()) - set(checkpoint['state_dict'].keys())
    if missing_keys:
        logging.warning("Missing keys in state_dict: %s", missing_keys)
        
    return model

def join_shards(source_folder, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)
    counter = 0
    num_digits = 6
    for subdir, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith('.tar.gz'):
                original_path = os.path.join(subdir, file)
                new_filename = f"imgs_lc-{str(counter).zfill(num_digits)}.tar.gz"
                new_path = os.path.join(destination_folder, new_filename)
                shutil.move(original_path, new_path)
                counter += 1
    shutil.rmtree(source_folder)
    print("Files renamed and moved successfully.")

def calculate_metrics(y_true, y_pred):    
    return classification_report(y_true, y_pred, output_dict=False, digits=4)
    #{
    #    'count': len(y_true),
    #    'f1_score': f1_score(y_true, y_pred, average='macro'),
    #    'recall': recall_score(y_true, y_pred, average='macro'),
    #    'precision': precision_score(y_true, y_pred, average='macro'),
    #    'accuracy': accuracy_score(y_true, y_pred),
    ##    'cm': confusion_matrix(y_true, y_pred),
    #}

def single_confusion_matrix(y_true, y_pred, order_classes, path_save=None):
    figsize = max(len(order_classes)-2, 6)
    fs_increment = len(order_classes) * 0.5
    fs = 10 + fs_increment

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=order_classes, normalize='true')

    cmap = plt.cm.Blues
    fig, ax = plt.subplots(figsize=(figsize, figsize)) #, dpi=110)
    im = ax.imshow(np.around(cm, decimals=2), interpolation='nearest', cmap=cmap)

    # color map
    new_color = cmap(1.0) 

    # Añadiendo manualmente las anotaciones con la media y desviación estándar
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] >= 0.005:
                #print(cm[i, j])
                text = f'{np.around(cm[i, j], decimals=2)}'
                color = "white" if cm[i, j] > 0.5 else new_color  # Blanco para la diagonal, tono de azul para otras celdas
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=fs)
            else:
                text = f'{np.around(cm[i, j], decimals=2)}'
                color = "white" if cm[i, j] > 0.5 else new_color  # Blanco para la diagonal, tono de azul para otras celdas
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=fs)

    # Ajustes finales y mostrar la gráfica
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.set_xticks(np.arange(len(order_classes)))
    ax.set_yticks(np.arange(len(order_classes)))
    ax.set_xticklabels(order_classes)
    ax.set_yticklabels(order_classes)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    ax.set_xlabel('Predicted Label', fontsize=fs+2)
    ax.set_ylabel('True Label', fontsize=fs+2)

    ax.xaxis.label.set_size(fs+2)
    ax.yaxis.label.set_size(fs+2)
    ax.xaxis.labelpad = 13
    ax.yaxis.labelpad = 13

    plt.tight_layout()
    plt.savefig(path_save)
    plt.close(fig)