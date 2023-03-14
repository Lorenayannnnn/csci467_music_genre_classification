import pandas as pd
import torch

from AudioToGenreDataset import AudioToGenreDataset


def get_device(force_cpu, status=True):
    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        if status:
            print("Using CUDA")
    else:
        device = torch.device("cpu")
        if status:
            print("Using CPU")
    return device


def load_split_dataframe(data_split_filename: str):
    """
    Create dataframe for train, dev, test based on input data split file
    :param data_split_filename:
    :return: train_dataframe, dev_dataframe, test_dataframe
    """
    # Dataframe: {"audio_filename": [], "label": []}
    train_dataframe = {}
    dev_dataframe = {}
    test_dataframe = {}
    f = open(data_split_filename, "r")
    for i in range(3):
        split_name = f.readline().replace(":\n", "")
        split_data = f.readline().replace("['", "").replace("']", "").strip().split("', '")

        temp_audio_filenames = []
        temp_labels = []
        for filename in split_data:
            # filename: pop/pop00048.png
            filename_label_split = filename.split("/")      # Get genre
            genre = filename_label_split[0]

            filename_type_split = filename_label_split[1].split(".")    # For changing from png to wav
            filename_id_split = filename_type_split[0].split(genre)

            temp_audio_filenames.append(genre + "/" + genre + "." + filename_id_split[1] + ".wav")
            temp_labels.append(genre)

        if split_name == "train":
            train_dataframe = pd.DataFrame({"audio_filename": temp_audio_filenames, "label": temp_labels})
        elif split_name == "dev":
            dev_dataframe = pd.DataFrame({"audio_filename": temp_audio_filenames, "label": temp_labels})
        else:
            test_dataframe = pd.DataFrame({"audio_filename": temp_audio_filenames, "label": temp_labels})

    f.close()
    return train_dataframe, dev_dataframe, test_dataframe


def create_dataset_w_dataframe(dataframe, root_dir: str, feature_extractor, sample_rate: int):
    return AudioToGenreDataset(
        root_dir=root_dir,
        dataframe=dataframe,
        feature_extractor=feature_extractor,
        sample_rate=sample_rate
    )
