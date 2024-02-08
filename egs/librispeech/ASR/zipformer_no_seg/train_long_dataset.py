import os
from pathlib import Path
from typing import Tuple, Union
import glob

import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchaudio

from lhotse import CutSet

# References:
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# https://huggingface.co/blog/audio-datasets
# https://pytorch.org/audio/stable/transforms.html

class LongAudioDataset(Dataset):
    
    def __init__(
        self,
        root: Union[str, Path],
        audio_type: str = "mp3",
    ) -> None:
        self.root = [root]

        self.audio_files = glob.glob(f"{root}/**/*.{audio_type}", recursive=False)
        if len(self.audio_files) == 0:
            self.audio_files = glob.glob(f"{root}/*.{audio_type}", recursive=False)

    def __getitem__(self, n: int): #  -> Tuple[Tensor, int, str, int, int]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            Tensor:
                Waveform
            int:
                Sample rate
            str:
                Transcript
            int:
                Speaker ID
            int:
                Utterance ID
        """
        audio_path = self.audio_files[n]
        trans_path_func = lambda x: x.replace("/audio", "/trans")[:-4] + ".txt"
        trans_path = trans_path_func(audio_path)

        audio_id = Path(audio_path).stem
        speaker_id, segment_id = 0, 0

        with open(trans_path) as fin:
            text = fin.readlines()
        text = text[0].strip()

        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except:  # in case of broken file
            waveform, sample_rate = [], -1

        return waveform, sample_rate, text, speaker_id, audio_id, audio_path

    def __len__(self) -> int:
        return len(self.audio_files)


if __name__ == "__main__":
    long_dataset = LongAudioDataset(
        root = "/scratch4/skhudan1/rhuang25/data/seekingalpha/audio2019/",
    )

    waveform, sample_rate, text, speaker_id, audio_id, audio_path = \
        long_dataset[0]
    
    print(f"[{audio_id}] len(text) = {len(text)}")
