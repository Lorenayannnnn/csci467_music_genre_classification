
import os
from pathlib import Path

from pydub import AudioSegment


"""
Usage:
audio_segments = split_audio_file(
        data_dir="../data/genres_original/",
        genre="pop",
        file_id="00048",
        original_filename="pop/pop.00048.wav"
    )
"""
def split_audio_file(data_dir: str, genre: str, file_id: str, original_filename: str):
    """
    Split input audio file into pieces of 3 seconds with 50%
    :param data_dir: ../data/genres_original/
    :param genre: pop
    :param file_id (00048)
    :param original_filename (pop/pop.00048.wav)

    :returns list of processed audio filenames (e.g.: split/pop/pop.00048.0.wav)
    """
    start = 0
    end = 3
    idx = 0

    # Read in input audio file
    original_audio = AudioSegment.from_wav(os.path.join(data_dir, original_filename))

    audio_segments = []
    # Each audio track is 30 seconds long
    while end < 30:
        segment = original_audio[(start * 1000):(end * 1000)]
        segment_filename = f"split/{genre}/{genre}.{file_id}.{idx}.wav"

        # Make dir if does not exist
        Path(os.path.join(data_dir, f"split/{genre}/")).mkdir(parents=True, exist_ok=True)

        # f = open(os.path.join(data_dir, segment_filename))
        segment.export(os.path.join(data_dir, segment_filename), format="wav")
        # f.close()

        audio_segments.append(segment_filename)
        start += 1.5
        end += 1.5
        idx += 1

    return audio_segments
