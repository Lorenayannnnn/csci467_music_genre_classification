
from torch import nn
from transformers import AutoModelForAudioClassification

from AudioToGenreDataset import genre_2_index, index_2_genre


class HubertForGenreClassification(nn.Module):

    def __init__(self, model_name):
        super(HubertForGenreClassification, self).__init__()
        self.model = AutoModelForAudioClassification.from_pretrained(model_name, num_labels=10,
                                                                     label2id=genre_2_index, id2label=index_2_genre)

    def forward(self, inputs):
        return self.model(inputs)
