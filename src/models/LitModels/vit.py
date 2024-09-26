from src.models.LitBaseModel import LitBaseModel
from transformers import ViTImageProcessor, ViTModel

class LitModel(LitBaseModel):
    def __init__(self, data_info, **kwargs):
        super().__init__(data_info, **kwargs)
        use_pt_model = kwargs.get('pretrained_model', {}).get('use', False)

        if use_pt_model:
            path_pt_model = kwargs['pretrained_model']['path']
            self.model = ViTModel.from_pretrained(path_pt_model).train()
            self.processor = ViTImageProcessor.from_pretrained(path_pt_model)
        else:
            raise "You need to define a model"
        
        self.classifier = self.init_classifier()

    def _get_pooled_output(self, outputs):
        return outputs.last_hidden_state[:, 0, :]