import random
import math
import json
from functools import partial
from typing import List, Optional, Dict, Any

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from data_musan_dataset import Musan

from lhotse import Fbank, FbankConfig
from lhotse.dataset import SpecAugment


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

        self.target_sample_rate = 16000
        self.extractor = Fbank()  # It's 16kHz. https://github.com/lhotse-speech/lhotse/blob/master/lhotse/features/kaldi/extractors.py#L24
        # self.extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))  # https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/local/compute_fbank_librispeech.py#L119
    
    def forward(self, sample, sampling_rate=16000):
        # features = self.extractor.extract_batch(
        #     sample, sampling_rate=sampling_rate, lengths=wave_lens
        # )

        if sampling_rate != self.target_sample_rate:
            sample = F.resample(sample, sampling_rate, self.target_sample_rate)

        with torch.no_grad():
            features = self.extractor.extract(sample, sampling_rate=self.target_sample_rate)
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

_spec_aug_transform = SpecAugment(
    time_warp_factor=80,
    num_frame_masks=10,
    features_mask_size=27,
    num_feature_masks=2,
    frames_mask_size=100,
)

# What transforms are there in icefall?
# - Add noise          [mine]
# - Speed perturbation [torchaudio]
# - SpecAugment        [lhotse/icefall] 
# In torchaudio, there's another:
# - _piecewise_linear_log
# - GlobalStatsNormalization

# check the transforms here:
# https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/tdnn_lstm_ctc/asr_datamodule.py#L218
# https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/speech_recognition.py
# and here:
# https://github.com/pytorch/audio/blob/main/examples/asr/librispeech_conformer_rnnt/transforms.py
# https://pytorch.org/audio/stable/transforms.html


# Dataloader:
#   - load n long audios + alignments (each audio 200MB)
#   - downsample each long audio to 16kHz
#   - get n segments of roughly t seconds according to the alignments


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
    # sample[0] is the waveform
    # sample[1] is sample rate
    # sample[2] is transcript
    # sample[3] is audio_id
    # sample[4] is start frame in the audio

    if speed_perturbation:
        samples = [_speed_perturb_transform(sample[0].squeeze()) for sample in samples]

    if musan_noise:  # TODO: this add noise process may be changed to the same as in lhotse in the future
        total_length = sum([sample[0].size(-1) for sample in samples])
        _additive_noise_transform.fetch_noise_batch(total_length)
        samples = [_additive_noise_transform(sample[0].squeeze()) for sample in samples]

    # print(f"samples[0][0]={samples[0][0]}")
    # print(f"samples={samples}")
    
    sample_rate = samples[0][1]  # all samples has been converted to the same sampling rate before this
    features = [_fbank_transform(sample[0].squeeze(), sampling_rate=sample_rate) for sample in samples]
    lengths = torch.tensor([elem.shape[0] for elem in features], dtype=torch.int32)
    features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    features = data_pipeline(features)
    return features, lengths


class TrainTransform:
    def __init__(self, sp_model, config: dict):
        self.sp_model = sp_model
        self.config = config
        
        if True:
            self.train_data_pipeline = torch.nn.Sequential(
                _spec_aug_transform,
            )

    def __call__(self, samples: List):
        features, feature_lengths = _extract_features(
            self.train_data_pipeline, 
            samples,
            # speed_perturbation=self.config["speed_perturbation"],
            # musan_noise=self.config["musan_noise"],
            speed_perturbation=True,
            musan_noise=True,
        )

        targets, target_lengths = _extract_labels(self.sp_model, samples)
        return features, feature_lengths, targets, target_lengths, samples


class ValTransform:
    def __init__(self, sp_model):
        self.sp_model = sp_model
        self.valid_data_pipeline = torch.nn.Sequential()
        # self.valid_data_pipeline = torch.nn.Sequential(
        #     FunctionalModule(_piecewise_linear_log),
        #     GlobalStatsNormalization(global_stats_path),
        # )

    def __call__(self, samples: List):
        features, feature_lengths = _extract_features(self.valid_data_pipeline, samples)
        targets, target_lengths = _extract_labels(self.sp_model, samples)
        return features, feature_lengths, targets, target_lengths, samples


class TestTransform:
    def __init__(self, sp_model):
        self.val_transforms = ValTransform(sp_model)

    def __call__(self, sample: List):
        # return self.val_transforms([sample]), [sample]
        # return self.val_transforms(sample), sample
        return self.val_transforms(sample)