import torch
from torch import nn
from transformers import AutoModel, AutoConfig, Wav2Vec2Model, Wav2Vec2Config


class MusicGenreClassificationModel(nn.Module):

    def __init__(self, model_name, freeze_part: str):
        super(MusicGenreClassificationModel, self).__init__()
        self.config = Wav2Vec2Config.from_pretrained(model_name)
        self.wav2vec2_model = Wav2Vec2Model.from_pretrained(model_name)
        self.init_freeze(freeze_part)

        # Classifier head
        self.projector = nn.Linear(in_features=self.config.hidden_size, out_features=int(self.config.hidden_size / 2))
        self.classifier = nn.Linear(in_features=int(self.config.hidden_size / 2), out_features=10)


    def init_freeze(self, freeze_part: str):
        """
        :param freeze_part: ['full' (entire model), 'feature_extractor', 'none']
        """
        if freeze_part is None:
            raise ValueError(
                "Need to select part of the pretrained model that will be freeze but got None. Choose from the following choices: full, feature_extractor, none")
        if freeze_part == "full":
            self.wav2vec2_model._freeze_parameters()
        elif freeze_part == "feature_extractor":
            self.wav2vec2_model.feature_extractor._freeze_parameters()



    def forward(self, inputs):
        # Freeze the model (use as feature extractor)
        with torch.no_grad():
            hidden_states = self.wav2vec2_model(inputs).last_hidden_state
            # Get hidden state of last token of the entire sequence
            last_token_hidden = hidden_states[:, -1, :]

        projector_outputs = self.projector(last_token_hidden)
        outputs = self.classifier(projector_outputs)

        return outputs
