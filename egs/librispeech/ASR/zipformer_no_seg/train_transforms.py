import random
import math
from functools import partial
from typing import List

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from train_musan_dataset import Musan

from lhotse import Fbank, FbankConfig


class AddNoise(torch.nn.Module):
    def __init__(
        self,
        noise_dataset,
        sampling_rate = 16000,
        snr = (10, 20),
        p = 0.5,
        seed: int = 42,
    ):
        super().__init__()

        self.noise_dataset = noise_dataset
        self.noise_count = len(noise_dataset)
        self.sampling_rate = sampling_rate
        self.snr = snr
        self.p = p
        self.seed = seed
        self.rng = random.Random(seed)

        self.noise_batch = None
        self.position = 0


    def fetch_noise_batch(self, total_length):
        # This will fetch noise until the number of sample points is equal or more than the speech samples

        wav_list = []
        while total_length > 0:
            idx = self.rng.randint(0, self.noise_count - 1)  # (both included)
            wav, sr, filename = self.noise_dataset[idx]
            assert sr == self.sampling_rate
            wav_list.append(wav)
            total_length -= wav.size(-1)
        self.noise_batch = torch.cat(wav_list, dim=-1)
        self.position = self.rng.randint(0, self.noise_batch.size(-1) - 1)
        return self.noise_batch


    def get_my_noise(self, length):
        noise_list = []
        while length > 0:
            noise = self.noise_batch[:, self.position:self.position+length]
            noise_list.append(noise)
            length -= noise.size(-1)
            self.position += length
            if self.position > self.noise_batch.size(-1):
                self.position = 0
        return torch.cat(noise_list, dim=-1)


    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        # # Resample the input
        # resampled = self.resample(sample)

        if sample.ndim == 1:
            sample = sample.unsqueeze(0)

        if self.rng.uniform(0.0, 1.0) > self.p:
            return sample[0], sample.size(-1)

        snr = self.rng.uniform(*self.snr) if isinstance(self.snr, (list, tuple)) else self.snr
        noise = self.get_my_noise(sample.size(-1))
        noisy_sample = F.add_noise(sample, noise, torch.tensor([snr]))
        return noisy_sample[0], noisy_sample.size(-1)


# https://github.com/lhotse-speech/lhotse/blob/master/docs/features.rst
# https://github.com/lhotse-speech/lhotse/blob/master/lhotse/features/fbank.py
# https://github.com/lhotse-speech/lhotse/blob/master/lhotse/cut/set.py#L2107
# https://github.com/lhotse-speech/lhotse/blob/master/lhotse/features/kaldi/extractors.py
# https://github.com/lhotse-speech/lhotse?tab=readme-ov-file#examples
# https://github.com/lhotse-speech/lhotse/blob/master/lhotse/cut/base.py#L289
# https://github.com/lhotse-speech/lhotse/blob/master/lhotse/cut/base.py#L95C13-L95C37
class LhotseFbank(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.sample_rate = 16000
        self.extractor = Fbank()  # It's 16kHz. https://github.com/lhotse-speech/lhotse/blob/master/lhotse/features/kaldi/extractors.py#L24
        # self.extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))  # https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/local/compute_fbank_librispeech.py#L119
    
    def forward(self, sample, sampling_rate=16000):
        # features = self.extractor.extract_batch(
        #     sample, sampling_rate=sampling_rate, lengths=wave_lens
        # )

        if sampling_rate != self.sample_rate:
            sample = F.resample(sample, sampling_rate, self.sample_rate)

        with torch.no_grad():
            features = self.extractor.extract(sample, sampling_rate=self.sample_rate)
        return features


class Resample(torch.nn.Module):
    def __init__(self, target_sample_rate=16000):
        super().__init__()
        self.target_sample_rate = target_sample_rate
    
    def forward(self, sample, sampling_rate=16000):
        return F.resample(sample, sampling_rate, self.target_sample_rate)



_decibel = 2 * 20 * math.log10(torch.iinfo(torch.int16).max)
_gain = pow(10, 0.05 * _decibel)

_spectrogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, n_mels=80, hop_length=160)
_fbank_transform = LhotseFbank()  # 16kHz
_resample_transform = Resample(target_sample_rate=16000)
_speed_perturb_transform = torchaudio.transforms.SpeedPerturbation(orig_freq=16000, factors=[0.9, 1.0, 1.1])

musan_path = "/home/rhuang25/work/icefall/egs/librispeech/ASR/download/musan/"
subsets = ["noise", "music"]  # ["music", "noise", "speech"]  # TODO: check what's done in lhotse
musan = Musan(musan_path, subsets)
_additive_noise_transform = AddNoise(musan, snr=(15, 25), p=0.5)


def _piecewise_linear_log(x):
    x = x * _gain
    x[x > math.e] = torch.log(x[x > math.e])
    x[x <= math.e] = x[x <= math.e] / math.e
    return x


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


# https://github.com/lhotse-speech/lhotse/blob/master/docs/features.rst#feature-normalization
# TODO: it seems icefall don't do CMVN
class GlobalStatsNormalization(torch.nn.Module):
    def __init__(self, global_stats_path):
        super().__init__()

        with open(global_stats_path) as f:
            blob = json.loads(f.read())

        self.mean = torch.tensor(blob["mean"])
        self.invstddev = torch.tensor(blob["invstddev"])

    def forward(self, input):
        return (input - self.mean) * self.invstddev


def _extract_labels(sp_model, samples: List):
    targets = [sp_model.encode(sample[2].lower()) for sample in samples]
    lengths = torch.tensor([len(elem) for elem in targets]).to(dtype=torch.int32)
    targets = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(elem) for elem in targets],
        batch_first=True,
        padding_value=1.0,
    ).to(dtype=torch.int32)
    return targets, lengths


def _extract_features(data_pipeline, samples: List, speed_perturbation=False, musan_noise=False):
    if speed_perturbation:
        samples = [_speed_perturb_transform(sample[0].squeeze()) for sample in samples]

    if musan_noise:
        total_length = sum([sample[0].size(-1) for sample in samples])
        _additive_noise_transform.fetch_noise_batch(total_length)
        samples = [_additive_noise_transform(sample[0].squeeze()) for sample in samples]

    # print(f"samples[0][0]={samples[0][0]}")
    # print(f"samples={samples}")
    mel_features = [_spectrogram_transform(sample[0].squeeze()).transpose(1, 0) for sample in samples]
    features = torch.nn.utils.rnn.pad_sequence(mel_features, batch_first=True)
    features = data_pipeline(features)
    lengths = torch.tensor([elem.shape[0] for elem in mel_features], dtype=torch.int32)
    return features, lengths


class TrainTransform:
    def __init__(self, global_stats_path: str, sp_model, config: dict):
        self.sp_model = sp_model

        self.config = config
        if config["specaug_conf"]["new_spec_aug_api"]:
            spec_aug_transform = T.SpecAugment(
                n_time_masks=config["specaug_conf"]["n_time_masks"],
                time_mask_param=config["specaug_conf"]["time_mask_param"],
                p=config["specaug_conf"]["p"],
                n_freq_masks=config["specaug_conf"]["n_freq_masks"],
                freq_mask_param=config["specaug_conf"]["freq_mask_param"],
                iid_masks=config["specaug_conf"]["iid_masks"],
                zero_masking=config["specaug_conf"]["zero_masking"],
            )
            self.train_data_pipeline = torch.nn.Sequential(
                FunctionalModule(_piecewise_linear_log),
                GlobalStatsNormalization(global_stats_path),
                FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)),
                spec_aug_transform,
                FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)),
            )
        else:
            layers = []
            layers.append(FunctionalModule(_piecewise_linear_log))
            layers.append(GlobalStatsNormalization(global_stats_path))
            layers.append(FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)))
            for _ in range(config["specaug_conf"]["n_freq_masks"]):
                layers.append(
                    torchaudio.transforms.FrequencyMasking(
                        config["specaug_conf"]["freq_mask_param"]
                    )
                )
            for _ in range(config["specaug_conf"]["n_time_masks"]):
                layers.append(
                    torchaudio.transforms.TimeMasking(
                        config["specaug_conf"]["time_mask_param"], 
                        p=config["specaug_conf"]["p"]
                    )
                )
            layers.append(FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)))
            self.train_data_pipeline = torch.nn.Sequential(
                *layers,
            )

    def __call__(self, samples: List):
        features, feature_lengths = _extract_features(
            self.train_data_pipeline, 
            samples,
            speed_perturbation=self.config["speed_perturbation"],
            musan_noise=self.config["musan_noise"],
        )

        targets, target_lengths = _extract_labels(self.sp_model, samples)
        return Batch(features, feature_lengths, targets, target_lengths, samples)


class ValTransform:
    def __init__(self, global_stats_path: str, sp_model):
        self.sp_model = sp_model
        self.valid_data_pipeline = torch.nn.Sequential(
            FunctionalModule(_piecewise_linear_log),
            GlobalStatsNormalization(global_stats_path),
        )

    def __call__(self, samples: List):
        features, feature_lengths = _extract_features(self.valid_data_pipeline, samples)
        targets, target_lengths = _extract_labels(self.sp_model, samples)
        return Batch(features, feature_lengths, targets, target_lengths, samples)


class TestTransform:
    def __init__(self, global_stats_path: str, sp_model):
        self.val_transforms = ValTransform(global_stats_path, sp_model)

    def __call__(self, sample):
        # return self.val_transforms([sample]), [sample]
        # return self.val_transforms(sample), sample
        return self.val_transforms(sample)