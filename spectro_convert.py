import os
import matplotlib.pyplot as plt

# for loading and visualizing audio files
import librosa
import librosa.display
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

rootdir = "../Data/genres_original/split"

folders = [x[0] for x in os.walk(rootdir)]
folders.pop(0)

for folder in folders:
    # # print(folder)
    audio_fpath = folder
    audio_clips = os.listdir(audio_fpath)
    print(folder[30:])

    if 'country' not in folder:
        continue

    savepath = "../Data/genres_original/spectro/" + folder[30:]

    Path(savepath).mkdir(parents=True, exist_ok=True)

    for audio in audio_clips:
        # x, sr = librosa.load(audio_fpath + '/' + audio, sr=44100)
        # # X = librosa.stft(x)
        # # Xdb = librosa.amplitude_to_db(abs(X))
        #
        # window_size = 1024
        # window = np.hanning(window_size)
        # stft = librosa.core.spectrum.stft(x, n_fft=window_size, hop_length=512, window=window)
        # out = 2 * np.abs(stft) / np.sum(window)
        #
        # fig = plt.figure(figsize=(3, 3))
        # canvas = FigureCanvas(fig)
        # ax = fig.add_subplot(111)
        # p = librosa.display.specshow(Xdb, ax=ax)
        # index = audio.find('.')

        y, sr = librosa.load(audio_fpath + '/' + audio, sr=44100)

        window_size = 1024
        window = np.hanning(window_size)
        stft = librosa.stft(y, n_fft=window_size, hop_length=512, window=window)
        out = 2 * np.abs(stft) / np.sum(window)

        # For plotting headlessly

        fig = plt.figure(figsize=(1, 1))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=ax)

        index = audio.find('.')
        filename = savepath + '/' + audio[:len(audio) - 4] + '.png'
        fig.savefig(filename, dpi=200)
        im = Image.open(filename)
        im1 = im.crop((37, 26, 171, 161))
        im1 = im1.save(filename)
        im.close()
        plt.close()


