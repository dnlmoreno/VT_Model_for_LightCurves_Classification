from lightning.pytorch.callbacks.callback import Callback
import pandas as pd

class EpochUpdate(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch

        if current_epoch < 1:
            train_dataloader = trainer.datamodule.train_dataloader()
            train_dataset = train_dataloader.dataset
            train_dataset.first_epoch = False
            #train_dataset.image_cache = dict(train_dataset.image_cache)
            #train_dataset.image_cache = pd.DataFrame({
            #    "snid": train_dataset.image_cache.keys(),
            #    "image": train_dataset.image_cache.values()
            #}).set_index("snid")

            val_dataloader = trainer.datamodule.val_dataloader()
            val_dataset = val_dataloader.dataset
            val_dataset.first_epoch = False
            #val_dataset.image_cache = dict(val_dataset.image_cache)
            #val_dataset.image_cache = pd.DataFrame({
            #    "snid": val_dataset.image_cache.keys(),
            #    "image": val_dataset.image_cache.values()
            #}).set_index("snid")
