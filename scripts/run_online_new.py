import os
import hydra
import mlflow
import torch
import glob
import logging
import shutil
import subprocess
import importlib
import lightning as L

from omegaconf import DictConfig, OmegaConf
from datetime import datetime

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, MLFlowLogger, CSVLogger
from lightning.pytorch.profilers import PyTorchProfiler

#from src.training.callbacks.ModelSummary import ModelSummary
from scripts.predict_clf import predict
from scripts.utils import *

import warnings
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    logging.info(f'🖥️ GPU detected: {gpu_name}')
else:
    logging.info('⚠️ No GPU available.')

def load_dataset(name_dataset, config):
    LitData_module = importlib.import_module(f"src.data.LitData_new")
    data_module = getattr(LitData_module, 'LitData_new')(name_dataset, **config)
    return data_module

def load_model(dataset_config, config):
    model_name = config['model_name']
    LitModel_module = importlib.import_module(f"src.models.LitModels.{model_name}")
    model = getattr(LitModel_module, 'LitModel')(data_info=dataset_config, **config)
    return model

def perform_ft_classification(run, config, dataset, experiment_name):

    experiment_id = run.info.experiment_id
    run_id = run.info.run_id
    EXPDIR = '{}/ml-runs/{}/{}/artifacts'.format(config['results_dir'], experiment_id, run_id)
    os.makedirs(EXPDIR, exist_ok=True)
    logging.info(f'📁 Experiment directory created: {EXPDIR}')

    # Model
    logging.info('🗂️  Creating the model.')
    model = load_model(dataset_config=dataset.dataset_config.copy(), config=config)

    # Save params:
    if config['checkpoint']['use']:
        logging.info('🔄 Checkpoint loading is enabled.')
        ckpt_dir = handle_ckpt_dir(config, fold=config['loader']['fold'])
        ckpt_model = sorted(glob.glob(ckpt_dir + "/*.ckpt"))[-1]
        if os.path.exists(ckpt_model):
            logging.info(f'📦 Loading checkpoint from {ckpt_model}.')
            model = load_checkpoint(model, ckpt_model)            
        else:
            raise FileNotFoundError(f"Checkpoint file not found at {ckpt_dir}")

        loaded_config = load_yaml(path='{}/hparams.yaml'.format(ckpt_dir))
        config['model_name'] = loaded_config['model_name']
        logging.info(f'✅ Model parameters loaded from checkpoint: {config["model_name"]}')

    os.makedirs(f'{EXPDIR}/model', exist_ok=True)
    save_yaml(dict(model.hparams), path=f'{EXPDIR}/model/hparams.yaml')
    logging.info('💾 Model hyperparameters saved.')

    # Callbacks 
    logging.info('🔧 Setting up training callbacks.')
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
    #model_summary = ModelSummary(max_depth=10, output_dir=f'{EXPDIR}/model')
    all_callbacks = [checkpoint, early_stopping] #, model_summary]

    # Loggers
    logging.info('📝 Initializing loggers for MLflow and CSV logging.')
    mlflow_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=f"file:{config['results_dir']}/ml-runs",
        )
    mlflow_logger._run_id = run.info.run_id
    
    csv_logger = CSVLogger(
        save_dir=EXPDIR, 
        name="logs",
        version='.',
        )
    
    all_loggers = [mlflow_logger, csv_logger] 
                
    # If debugging
    if config['debug']:
        logging.warning('⚠️ Debug mode enabled: Running only one epoch.')
        max_epochs = 1
    else:
        max_epochs = config['training']['num_epochs']

    # Training
    logging.info('🏋️‍♂️ Starting model training.')
    trainer = L.Trainer(
        callbacks=all_callbacks,
        logger=all_loggers,
        val_check_interval=1.0,
        log_every_n_steps=100,
        accelerator="gpu",
        min_epochs=1,
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, dataset)
    logging.info('🎉 Training completed successfully.')

    # Testing
    try:
        logging.info('🧪 Starting test evaluation.')
        trainer.test(dataloaders=dataset.test_dataloader(), ckpt_path="best")
        logging.info('✅ Test evaluation completed successfully.')
    except Exception as e:
        logging.error('❌ Test set evaluation failed. The test set was not created.')
        logging.exception(e)

    path_save_metrics = f'{EXPDIR}/metrics'
    os.makedirs(path_save_metrics, exist_ok=True)

    #ckpt_dir = handle_ckpt_dir(config, fold=config['loader']['fold'])
    #ckpt_model = sorted(glob.glob(ckpt_dir + "/*.ckpt"))[-1]
    #best_model = load_checkpoint(model, ckpt_model)   
    #print(type(best_model))  
    ##loaded_model = model.load_from_checkpoint(checkpoint.best_model_path).eval()
    #_ = predict(dataset, best_model, path_save_metrics)
    
    # --- Transfer to remote server and delete local files ---
    #remote_user = "dmoreno2016"  # Replace with your username on the remote server
    #remote_server = "pececillo.inf.udec.cl"  # Replace with your remote server address
    #remote_path = "/home/shared/daniel_results_tmp/hp_tuning"  # Replace with the path to save on the remote server
    #RUN_DIR = f"{config['results_dir']}/ml-runs/{experiment_id}/{run_id}"
#
    #try:
    #    logging.info(f'🔄 Transferring experiment files to {remote_server}')
    #    scp_command = f"scp -r {RUN_DIR} {remote_user}@{remote_server}:{remote_path}"
    #    subprocess.run(scp_command, shell=True, check=True)
    #    
    #    logging.info(f'✅ Files successfully transferred to {remote_server}. Now deleting local files.')
    #    shutil.rmtree(RUN_DIR)  # Delete the local directory after successful transfer
    #    logging.info(f'🗑️ Local files deleted: {RUN_DIR}')
    #    
    #except Exception as e:
    #    logging.error(f'❌ Failed to transfer files to {remote_server}. Local files were not deleted.')
    #    logging.exception(e)

    # --- Eliminar el archivo de checkpoint ---
    try:
        checkpoint_file = sorted(glob.glob(f'{EXPDIR}/model/my_best_checkpoint-*.ckpt'))[-1]  # Buscar el último archivo de checkpoint
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            logging.info(f'🗑️ Checkpoint file deleted: {checkpoint_file}')
    except Exception as e:
        logging.error('❌ Failed to delete checkpoint file.')
        logging.exception(e)


@hydra.main(config_path=os.getenv("HYDRA_CONFIG_PATH", "../configs/online"),
            config_name=os.getenv("HYDRA_CONFIG_NAME", "run_config"), 
            version_base=None)

def run(config: DictConfig) -> None:
    config = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    config = config['ft_classification']
    list_folds = config.pop('list_folds')

    config['hp_tuning'] = f"{config['imgs_params']['norm_name']}_" \
                          f"{config['imgs_params']['fig_params']['figsize'][0]}_" \
                          f"m{config['imgs_params']['fig_params']['markersize']}_" \
                          f"l{config['imgs_params']['fig_params']['linewidth']}_" \
                          f"e{config['imgs_params']['use_err']}_" \
                          f"{config['imgs_params']['input_type']}"
    
    # Dataset info
    name_dataset = config['loader']['path_data'].split('/')[2]

    # Setup MLflow
    logging.info('⚙️ Setting up MLflow tracking URI and experiment configuration.')
    mlflow.set_tracking_uri(f"file:{config['results_dir']}/ml-runs")
    experiment_name = f"ft_classification/{name_dataset}/testing"
    mlflow.set_experiment(experiment_name)

    config['run_name'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config['gpu_name'] = gpu_name

    logging.info(f'🚩 Starting parent run with name: {config["run_name"]}. Experiment: {experiment_name}')
    with mlflow.start_run(run_name=f"{config['run_name']}") as parent_run:

        for fold in list_folds:
            logging.info(f'📂 Starting FT Classification for Fold {fold}.')
            config['loader']['fold'] = fold

            # Data
            dataset = load_dataset(name_dataset, config)

            with mlflow.start_run(run_name=f"Fold_{fold}_{config['run_name']}", nested=True) as child_run:
                perform_ft_classification(child_run, config, dataset, experiment_name)


if __name__ == "__main__":

    run()
