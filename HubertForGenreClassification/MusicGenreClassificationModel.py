
import torch
from torch import nn
from transformers import AutoModel, AutoConfig, Wav2Vec2Model, Wav2Vec2Config


class MusicGenreClassificationModel(nn.Module):

    def __init__(self, model_name, freeze_part: str, process_last_hidden_state_method: str, dropout_rate=0.1):
        super(MusicGenreClassificationModel, self).__init__()
        self.config = Wav2Vec2Config.from_pretrained(model_name)
        self.wav2vec2_model = Wav2Vec2Model.from_pretrained(model_name)
        self.process_last_hidden_state_method = process_last_hidden_state_method    # average, sum, max

        self.init_freeze(freeze_part)

        # Classifier head
        # self.projector = nn.Linear(in_features=self.config.hidden_size, out_features=int(self.config.hidden_size / 2))
        # self.classifier = nn.Linear(in_features=int(self.config.hidden_size / 2), out_features=10)

        self.projector = nn.Linear(in_features=self.config.hidden_size, out_features=self.config.hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.tanh_f = nn.Tanh()
        self.classifier = nn.Linear(in_features=self.config.hidden_size, out_features=10)


    def init_freeze(self, freeze_part: str):
        """
        :param freeze_part: ['full' (entire model), 'feature_extractor', 'none', 'except_last_encoder']
        """
        if freeze_part is None:
            raise ValueError(
                "Need to select part of the pretrained model that will be freeze but got None. Choose from the following choices: full, feature_extractor, none")
        if freeze_part == "full":
            for p in self.wav2vec2_model.parameters():
                p.requires_grad = False
        elif freeze_part == "feature_extractor":
            self.wav2vec2_model.feature_extractor._freeze_parameters()
        elif freeze_part == "except_last_encoder":
            for p in self.wav2vec2_model.parameters():
                print(p)
                p.requires_grad = False
            raise ValueError("test")




    def process_last_hidden_state(self, hidden_states):
        # last, average, sum, max
        if self.process_last_hidden_state_method == "last":
            return hidden_states[:, -1, :]
        elif self.process_last_hidden_state_method == "average":
            return torch.mean(hidden_states, dim=1)
        elif self.process_last_hidden_state_method == "sum":
            return torch.sum(hidden_states, dim=1)
        elif self.process_last_hidden_state_method == "max":
            return torch.max(hidden_states, dim=1)[0]
        else:
            raise ValueError(
                f"Should pick one of the following method: last, average, sum, max (but got {self.process_last_hidden_state_method})")


    def forward(self, inputs):
        # Freeze the model (use as feature extractor)
        hidden_states = self.wav2vec2_model(inputs).last_hidden_state
        processed_hidden_states = self.process_last_hidden_state(hidden_states)

        # projector_outputs = self.projector(processed_hidden_states)
        # outputs = self.classifier(projector_outputs)

        projector_outputs = self.projector(processed_hidden_states)
        tanh_outputs = self.tanh_f(projector_outputs)
        dropout_outputs = self.dropout(tanh_outputs)
        outputs = self.classifier(dropout_outputs)

        return outputs
