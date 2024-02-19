import os
from pathlib import Path
from typing import Tuple, Union
import glob

import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchaudio

from lhotse import CutSet
from data_libri_pre_filter import pre_filter


# References:
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# https://huggingface.co/blog/audio-datasets
# https://pytorch.org/audio/stable/transforms.html
# https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51

class EarningsCallLongAudioDataset(Dataset):
    
    def __init__(
        self,
        root: Union[str, Path],
        audio_type: str = "mp3",
    ) -> None:
        self.root = root

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


class LibrispeechLongAudioDataset(Dataset):
    # cd /exp/rhuang/meta/icefall/egs/librispeech/ASR/download 
    # mkdir -p LibriSpeechOriginal
    # cd LibriSpeechOriginal/
    # wget https://www.openslr.org/resources/12/original-books.tar.gz 
    # wget https://www.openslr.org/resources/12/raw-metadata.tar.gz
    # wget https://us.openslr.org/resources/12/original-mp3.tar.gz
    # tar -xzf raw-metadata.tar.gz
    # tar -xzf original-books.tar.gz
    # tar -xzf original-mp3.tar.gz
    
    def __init__(
        self,
        root: Union[str, Path],
        audio_type: str = "mp3",
    ) -> None:
        self.root = root

        manifest_file = f"{root}/LibriSpeechOriginal/chapter_manifest.txt"
        with open(manifest_file) as fin:
            self.manifest = [l.strip().split("\t") for l in fin.readlines() if l.strip() > 0]

    def text_normalize(self, text: str) -> str:
        # We preserve the word index in the transcript (i.e, we don't remove words)
        # E.g., "hello 这 world" -> "hello * world"

        # [Ref] https://github.com/kaldi-asr/kaldi/blob/master/egs/librispeech/s5/local/lm/normalize_text.sh
        text = text.split()

        def text_normalize0(text: str) -> str:
            # Remove all punctuation
            text = text.translate(str.maketrans("", "", string.punctuation))
            # Convert all upper case to lower case
            text = text.upper()
            if len(text) == 0:
                return "*"
            return text
        
        text = [text_normalize0(w) for w in text]
        text = " ".join(text)
        return text

    def load_book(self, filename):
        # TODO: this can be done off-line

        with open(filename, 'r') as fin:
            lines = [l.strip() for l in fin]
        
        lines = pre_filter(lines)
        lines = [l for l in lines if len(l) > 0]
        lines = [self.text_normalize(l) for l in lines]

        book = " ".join(lines)
        return book


    def __getitem__(self, n: int):  # -> Tuple[Tensor, int, str, int, int, dict]:
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
            dict:
                Other metadata
        """
        audio_path, text_path = self.manifest[n]

        chapter_id = Path(audio_path).parent.stem
        speaker_id = Path(audio_path).parent.parent.stem
        book_id = Path(text_path).parent.stem

        text = self.load_book(text_path)

        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except:  # in case of broken file
            waveform, sample_rate = [], -1

        return waveform, sample_rate, text, speaker_id, audio_id, audio_path

    def __len__(self) -> int:
        return len(self.manifest)


class TedliumLongAudioDataset(Dataset):
    pass


if __name__ == "__main__":
    long_dataset = LongAudioDataset(
        root = "/scratch4/skhudan1/rhuang25/data/seekingalpha/audio2019/",
    )

    waveform, sample_rate, text, speaker_id, audio_id, audio_path = \
        long_dataset[0]
    
    print(f"[{audio_id}] len(text) = {len(text)}")
