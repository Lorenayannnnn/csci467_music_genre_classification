import torch
from torch import nn
from transformers import AutoModel, AutoConfig


class MusicGenreClassificationModel(nn.Module):

    def __init__(self, model_name):
        super(MusicGenreClassificationModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.projector = nn.Linear(in_features=self.config.hidden_size, out_features=int(self.config.hidden_size / 2))
        self.classifier = nn.Linear(in_features=int(self.config.hidden_size / 2), out_features=10)


    def forward(self, inputs):
        # Freeze the model (use as feature extractor)
        with torch.no_grad():
            hidden_states = self.hubert_model(inputs).last_hidden_state
            # Get hidden state of last token of the entire sequence
            last_token_hidden = hidden_states[:, -1, :]

        projector_outputs = self.projector(last_token_hidden)
        outputs = self.classifier(projector_outputs)

        return outputs
