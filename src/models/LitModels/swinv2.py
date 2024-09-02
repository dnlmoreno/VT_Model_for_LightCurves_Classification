from src.models.LitBaseModel import LitBaseModel
from transformers import AutoImageProcessor, Swinv2Model

class LitModel(LitBaseModel):
    def __init__(self, data_info, **kwargs):
        super().__init__(data_info, **kwargs)

        use_ckpt = kwargs.get('checkpoint', {}).get('use', False)

        if use_ckpt:
            pretrained_model = kwargs['checkpoint']['pretrained_model']
            self.model = Swinv2Model.from_pretrained(pretrained_model).train()
            self.processor = AutoImageProcessor.from_pretrained(pretrained_model)
        else:
            raise "You need to define a model"
        
        self.classifier = self.init_classifier()

    def _get_pooled_output(self, outputs):
        return outputs[1] 