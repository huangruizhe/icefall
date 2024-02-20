import os
from pathlib import Path
from typing import Tuple, Union
import glob
import string

import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchaudio

from lhotse import CutSet
from data_libri_pre_filter import pre_filter

import codecs
from unidecode import unidecode


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
        
        meta_data = {
            "audio_path": audio_path,
            "trans_path": trans_path,
        }

        return waveform, sample_rate, text, speaker_id, audio_id, meta_data

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
        skip_loading_audio=False,
        skip_text_normalization=True,
        manifest_file=None,
    ) -> None:
        self.root = root

        self.skip_loading_audio = skip_loading_audio
        self.skip_text_normalization = skip_text_normalization

        if manifest_file is None:
            manifest_file = f"{root}/LibriSpeechAligned/chapter_manifest.txt"
            # manifest_file = f"{root}/LibriSpeechOriginal/chapter_manifest.txt"
        with open(manifest_file) as fin:
            self.manifest = [l.strip().split("\t") for l in fin.readlines() if len(l.strip()) > 0]

    def text_normalize(self, text: str) -> str:
        # We preserve the word index in the transcript (i.e, we don't remove words)
        # E.g., "hello 这 world" -> "hello * world"

        # [Ref] https://github.com/kaldi-asr/kaldi/blob/master/egs/librispeech/s5/local/lm/normalize_text.sh
        text = text.split()

        def text_normalize0(text: str) -> str:
            # Remove all punctuation
            # punctuation = string.punctuation
            punctuation = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'  # keep "'" as in Librispeech (how about "-"? -- ok, I don't find '-' in the per_utt file. So not allowed)
            text = text.translate(str.maketrans("", "", punctuation))
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

        try:
            with open(filename, 'r') as fin:
                lines = [l.strip() for l in fin]
        except UnicodeDecodeError:
            try:
                with open(filename, 'r', encoding='latin-1') as fin:
                    lines = [unidecode(l.strip()) for l in fin]
            except UnicodeDecodeError:
                with codecs.open(filename, 'r', encoding='utf-8') as fin:
                    lines = [unidecode(l.strip()) for l in fin.readlines()]

        if not self.skip_text_normalization:
            lines = pre_filter(lines)
            lines = [l for l in lines if len(l) > 0]
            lines = [self.text_normalize(l) for l in lines]

        book = " ".join(lines)
        book = book.strip()
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

        try:
            text = self.load_book(f"{self.root}/{text_path}")
        except:  # in case of broken file
            text = None

        if self.skip_loading_audio:
            waveform, sample_rate = [], -1
        else:
            try:
                waveform, sample_rate = torchaudio.load(f"{self.root}/{audio_path}")
            except:  # in case of broken file
                waveform, sample_rate = [], -1
        
        meta_data = {
            "book_id": book_id,
            "chapter_id": chapter_id,
            "speaker_id": speaker_id,
            "audio_path": audio_path,
            "text_path": text_path,
        }

        return waveform, sample_rate, text, speaker_id, chapter_id, meta_data 

    def __len__(self) -> int:
        return len(self.manifest)


class TedliumLongAudioDataset(Dataset):
    pass


if __name__ == "__main__":

    import logging
    logging.basicConfig(
        format = "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        level = 10
    )
    
    # long_dataset = EarningsCallLongAudioDataset(
    #     root = "/scratch4/skhudan1/rhuang25/data/seekingalpha/audio2019/",
    # )
    # waveform, sample_rate, text, speaker_id, audio_id, meta_data = \
    #     long_dataset[0]    
    # print(f"[{audio_id}] len(text) = {len(text)}")

    long_dataset = LibrispeechLongAudioDataset(
        root = "/exp/rhuang/meta/icefall/egs/librispeech/ASR/download/",
    )
    waveform, sample_rate, text, speaker_id, audio_id, meta_data = \
        long_dataset[0]    
    print(f"[{audio_id}] len(text) = {len(text)}")

