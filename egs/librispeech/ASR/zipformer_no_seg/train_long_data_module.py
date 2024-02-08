import os
import random
import math
from functools import partial

import torch
import torchaudio
import torchaudio.transforms as T

from train_long_dataset import LongAudioDataset

import sentencepiece as spm
from tqdm import tqdm


# _additive_noise_transform = None


class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform_fn):
        self.dataset = dataset
        self.transform_fn = transform_fn

    def __getitem__(self, idx):
        return self.transform_fn(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)


class LongAudioDataModule():
    def __init__(
        self,
        data_path,
        transform,
        max_tokens=700,
        batch_size=2,
        num_buckets=50,
        train_shuffle=True,
        num_workers=10,
        dataset_name="BUCKEYE",
    ):
        super().__init__()
        self.buckeye_path = buckeye_path
        self.dataset_lengths = None
        self.speakers = None
        self.transform = transform
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.num_buckets = num_buckets
        self.train_shuffle = train_shuffle
        self.num_workers = num_workers
        self.dataset_name = dataset_name

    def train_dataloader(self):
        if self.dataset_name == "BUCKEYE":
            datasets = [BUCKEYE(self.buckeye_path)]
        elif self.dataset_name == "TIMIT":
            datasets = [TIMIT(self.buckeye_path)]
        elif self.dataset_name == "librispeech" or self.dataset_name == "spgispeech":
            datasets = [LhotseDataset(self.buckeye_path)]
        else:
            raise NotImplementedError

        if not self.dataset_lengths:
            self.dataset_lengths = [get_sample_lengths(dataset) for dataset in datasets]

        dataset = torch.utils.data.ConcatDataset(
            [
                CustomBucketDataset(
                    dataset,
                    lengths,
                    self.max_tokens,
                    self.num_buckets,
                    batch_size=self.batch_size,
                    shuffle=False,
                )
                for dataset, lengths in zip(datasets, self.dataset_lengths)
            ]
        )
        dataset = TransformDataset(dataset, self.transform)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=None,
            shuffle=self.train_shuffle,
        )
        return dataloader

    def val_dataloader(self):
        return self.train_dataloader()
    
    def train_speaker_dataloader(self):
        datasets = [BUCKEYE(self.buckeye_path)]

        if not self.dataset_lengths:
            self.dataset_lengths = []
            self.speakers = []
            for dataset in datasets:
                sample_lengths, speakers = get_sample_lengths_and_speakers(dataset)
                self.dataset_lengths.append(sample_lengths)
                self.speakers.append(speakers)

        dataset = torch.utils.data.ConcatDataset(
            [
                CustomSpeakerDataset(
                    dataset,
                    lengths,
                    speakers,
                    self.max_tokens,
                    self.num_buckets,
                    batch_size=self.batch_size,
                )
                for dataset, lengths, speakers in zip(datasets, self.dataset_lengths, self.speakers)
            ]
        )
        dataset = TransformDataset(dataset, self.transform)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=None,
            shuffle=self.train_shuffle,
        )
        return dataloader

    # def val_dataloader(self):
    #     datasets = [
    #         self.librispeech_cls(self.librispeech_path, url="dev-clean"),
    #         self.librispeech_cls(self.librispeech_path, url="dev-other"),
    #     ]

    #     if not self.val_dataset_lengths:
    #         self.val_dataset_lengths = [get_sample_lengths(dataset) for dataset in datasets]

    #     dataset = torch.utils.data.ConcatDataset(
    #         [
    #             CustomBucketDataset(
    #                 dataset,
    #                 lengths,
    #                 self.max_tokens,
    #                 1,
    #                 batch_size=self.batch_size,
    #             )
    #             for dataset, lengths in zip(datasets, self.val_dataset_lengths)
    #         ]
    #     )
    #     dataset = TransformDataset(dataset, self.val_transform)
    #     dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=self.num_workers)
    #     return dataloader

    # def test_dataloader(self, test_part="test-other"):
    #     dataset = self.librispeech_cls(self.librispeech_path, url=test_part)
    #     dataset = TransformDataset(dataset, self.test_transform)
    #     dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)
    #     return dataloader
