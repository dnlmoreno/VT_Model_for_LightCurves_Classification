import os
import hydra
import mlflow
import torch
import glob
import logging
import importlib
import lightning as L

from omegaconf import DictConfig, OmegaConf
from datetime import datetime

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, MLFlowLogger, CSVLogger
from lightning.pytorch.profilers import PyTorchProfiler

from src.training.callbacks.ModelSummary import ModelSummary
from scripts.utils import *

import warnings
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    logging.info(f"GPU name: {gpu_name}")
else:
    logging.info("No GPU available.")

def load_dataset(data_info, config):
    LitData_module = importlib.import_module(f"src.data.LitData")
    data_module = getattr(LitData_module, 'LitData')(data_info, **config)
    return data_module

def load_model(data_info, config):
    model_name = config['model_name']
    LitModel_module = importlib.import_module(f"src.models.LitModels.{model_name}")
    model = getattr(LitModel_module, 'LitModel')(data_info, **config)
    return model

def perform_ft_classification(run, config, dataset, experiment_name, data_info):

    experiment_id = run.info.experiment_id
    run_id = run.info.run_id
    EXPDIR = '{}/ml-runs/{}/{}/artifacts'.format(config['results_dir'], experiment_id, run_id)
    os.makedirs(EXPDIR, exist_ok=True)

    # Model
    #dataset.setup('fit')
    model = load_model(data_info=data_info,
                       config=config)

    # Save params:
    os.makedirs(f'{EXPDIR}/model', exist_ok=True)
    save_yaml(dict(model.hparams), path=f'{EXPDIR}/model/hparams.yaml')

    # Callbacks 
    monitor = config['training']['monitor']
    checkpoint = ModelCheckpoint(
        monitor=monitor,
        dirpath=f'{EXPDIR}/model',
        save_top_k=1,
        mode="min" if 'loss' in monitor else "max",  
        every_n_train_steps=1,
        filename="my_best_checkpoint-{step}",
    )
    early_stopping = EarlyStopping(
        monitor=monitor,
        min_delta=0.00,
        patience=config['training']['patience'],
        verbose=False,
        mode="min" if 'loss' in monitor else "max",
    )
    model_summary = ModelSummary(max_depth=10, output_dir=f'{EXPDIR}/model')
    all_callbacks = [checkpoint, early_stopping, model_summary]

    # Loggers
    mlflow_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=f"file:{config['results_dir']}/ml-runs",
        )
    mlflow_logger._run_id = run.info.run_id

    #tensorboard_logger = TensorBoardLogger(
    #    save_dir='{}/tensorboard_logs'.format(config['results_dir']), 
    #    name=f"ft_classification/{config['run_name']}", 
    #    version=f"Fold_{config['loader']['fold']}",
    #    default_hp_metric=False
    #    )
    
    csv_logger = CSVLogger(
        save_dir=EXPDIR, 
        name="logs",
        version='.',
        )
    
    all_loggers = [mlflow_logger, csv_logger] #, tensorboard_logger]
                
    # If debugging
    if config['debug']:
        max_epochs = 1
    else:
        max_epochs = config['training']['num_epochs']

    # Training
    trainer = L.Trainer(
        callbacks=all_callbacks,
        logger=all_loggers,
        val_check_interval=1.0,
        log_every_n_steps=100,
        accelerator="gpu",
        min_epochs=1,
        max_epochs=max_epochs,
        num_sanity_val_steps=-1,
    )

    trainer.fit(model, dataset)

    # Testing
    try:
        dataset.setup('test')
        trainer.test(dataloaders=dataset.test_dataloader(),
                     ckpt_path="best")
    except:
        print('We finish the training without evaluating on the Test Set, because It was not created.')
    
    #if any(isinstance(logger, CSVLogger) for logger in all_loggers):
    #    plot_joint_learning_curves()


@hydra.main(config_path=os.getenv("HYDRA_CONFIG_PATH", "../configs"),
            config_name=os.getenv("HYDRA_CONFIG_NAME", "run_config"), 
            version_base=None)

def run(config: DictConfig) -> None:
    config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    config = config['ft_classification']
    list_folds = config.pop('list_folds')

    # Dataset info
    data_info = load_yaml(path='{}/data_info.yaml'.format(config['loader']['path_data']))
    name_dataset = data_info['name_dataset']

    # Setup MLflow
    mlflow.set_tracking_uri(f"file:{config['results_dir']}/ml-runs")
    experiment_name = f"ft_classification/{name_dataset}/testing"
    mlflow.set_experiment(experiment_name)

    config['run_name'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config['gpu_name'] = gpu_name
    with mlflow.start_run(run_name=f"{config['run_name']}") as parent_run:

        for fold in list_folds:
            logging.info(f'We are starting the FT Classification in Fold {fold}.')
            config['loader']['fold'] = fold

            # Data
            dataset = load_dataset(data_info, config)

            with mlflow.start_run(run_name=f"Fold_{fold}_{config['run_name']}", nested=True) as child_run:
                perform_ft_classification(child_run, config, dataset, experiment_name, data_info)


if __name__ == "__main__":

    run()
