
import numpy as np
import torch
from scipy.interpolate import interpolate
from scipy.io import wavfile
from torch.utils.data.dataset import Dataset

genre_2_index = {
    "blues": 0,
    "classical": 1,
    "country": 2,
    "disco": 3,
    "hiphop": 4,
    "jazz": 5,
    "metal": 6,
    "pop": 7,
    "reggae": 8,
    "rock": 9
}
index_2_genre = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

class AudioToGenreDataset(Dataset):
    def __init__(self, root_dir, dataframe, feature_extractor, sample_rate):
        # dataframe = {"audio_filename": [], "label": []}
        self.data_dict = dataframe
        self.root_dir = root_dir
        # Hubert feature extractor
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # root: data/genres_original/
        audio_file_name = self.root_dir + self.data_dict.loc[idx, "audio_filename"]
        label = genre_2_index[self.data_dict.loc[idx, "label"]]

        # Convert audio to array of float
        processed_audio_data = convert_audio_to_float_input_values(audio_file_name, self.sample_rate)

        inputs = self.feature_extractor(processed_audio_data, sampling_rate=self.feature_extractor.sampling_rate,
                                        max_length=self.sample_rate, truncation=True)

        inputs["label"] = torch.tensor(label, dtype=torch.long)
        return inputs


# Reference: https://stackoverflow.com/questions/33682490/how-to-read-a-wav-file-using-scipy-at-a-different-sampling-rate
def convert_audio_to_float_input_values(audio_file_name: str, input_sample_rate: int):
    """
    Read input audio file and convert the file into the required sample rate
    :return audio: array of float
    """
    sample_rate, audio = wavfile.read(audio_file_name)

    if sample_rate != input_sample_rate:
        duration = audio.shape[0] / sample_rate
        time_old = np.linspace(0, duration, audio.shape[0])
        time_new = np.linspace(0, duration, int(audio.shape[0] * input_sample_rate / sample_rate))
        interpolator = interpolate.interp1d(time_old, audio.T)
        audio = interpolator(time_new).T

    return audio
