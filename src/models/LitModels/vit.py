from src.models.LitBaseModel import LitBaseModel
from transformers import ViTImageProcessor, ViTModel

class LitModel(LitBaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        use_ckpt = kwargs['checkpoint']['use']

        if use_ckpt:
            pretrained_model = kwargs['checkpoint']['pretrained_model']
            self.model = ViTModel.from_pretrained(pretrained_model).train()
            self.processor = ViTImageProcessor.from_pretrained(pretrained_model)

    def _get_pooled_output(self, outputs):
        return outputs.last_hidden_state[:, 0, :]