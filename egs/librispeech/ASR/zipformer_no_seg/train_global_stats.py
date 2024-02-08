#!/usr/bin/env python3
"""Generate feature statistics for training set.

Example:
python /home/rhuang25/work/icefall/egs/librispeech/ASR/zipformer_no_seg/train_global_stats.py --dataset-path "/home/rhuang25/work/icefall/egs/librispeech/ASR/download/"
python /home/rhuang25/work/icefall/egs/librispeech/ASR/zipformer_no_seg/train_global_stats.py --dataset-path "/scratch4/skhudan1/rhuang25/data/seekingalpha/audio2019/" --output-path "/scratch4/skhudan1/rhuang25/data/seekingalpha/icefall_data/global_stats.json" --nj 45
python /home/rhuang25/work/icefall/egs/librispeech/ASR/zipformer_no_seg/train_global_stats.py --dataset-path "/scratch4/skhudan1/rhuang25/data/seekingalpha/audio2019/" --output-path "/scratch4/skhudan1/rhuang25/data/seekingalpha/icefall_data/global_stats.part1-1000.json" --nj 45 --part 1/1000
"""

import json
import logging
import pathlib
from argparse import ArgumentParser, RawTextHelpFormatter

import torch
import torchaudio

from tqdm import tqdm
from train_long_dataset import LongAudioDataset
from train_transforms import LhotseFbank


logger = logging.getLogger()


def parse_args():
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--dataset-path",
        required=True,
        type=pathlib.Path,
        help="Path to dataset. "
        "For LibriSpeech, all of 'train-clean-360', 'train-clean-100', and 'train-other-500' must exist.",
    )
    parser.add_argument(
        "--output-path",
        default=pathlib.Path("global_stats.json"),
        type=pathlib.Path,
        help="File to save feature statistics to. (Default: './global_stats.json')",
    )
    parser.add_argument('--nj', type=int, default=3, help='')
    parser.add_argument('--part', type=str, default=None, help='e.g., 1/4')
    return parser.parse_args()


def generate_statistics(samples, len_dataset=None, multiple_parts=False):
    E_x = 0
    E_x_2 = 0
    N = 0

    fbank_transform = LhotseFbank()

    for idx, sample in tqdm(enumerate(samples), total=len_dataset):
        if sample[1] == -1:
            print(f"audio_id={sample[-2]} cannot be loaded.")
            continue

        fbank = fbank_transform(sample[0].squeeze(), sampling_rate=sample[1])

        ####### Just verified that the feature extraction mechanism is the same as in icefall recipes #######
        ####### However, there's some minor differences still -- due to LilcomChunkyWriter? #######
        # from lhotse import CutSet
        # import torch
        # cs = CutSet.from_file("/home/rhuang25/work/icefall/egs/librispeech/ASR/data/fbank/librispeech_cuts_train-clean-100.jsonl.gz").to_eager()
        # c=cs['103-1240-0000-2545']
        # ft = c.load_features()
        # A=fbank
        # B=torch.Tensor(ft)
        # percentage_difference = torch.abs(A - B) / ((torch.abs(A) + torch.abs(B)) / 2) * 100
        # percentage_difference.min()
        # percentage_difference.max()
        # percentage_difference
        # torch.sum(percentage_difference > 1).item()
        # torch.sum(percentage_difference > 10).item()
        # cut_set = CutSet.from_cuts([c])
        # cut_no_feat = c.drop_features()
        # cut_no_feat
        # c
        # from lhotse import Fbank
        # ft2 = cut_no_feat.compute_features(extractor=Fbank())
        # num_mel_bins=80
        # from lhotse import FbankConfig
        # extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))
        # ft2 = cut_no_feat.compute_features(extractor=extractor)
        # from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter
        # cut_set = CutSet.from_cuts([cut_no_feat])
        # cut_set1 = cut_set.compute_and_store_features(extractor=extractor,storage_path='feats',storage_type=LilcomChunkyWriter)
        # cut_set1[0]
        # cut_set1[0].load_features()
        # torch.Tensor(cut_set1[0].load_features()) - ft

        sum = fbank.sum(0)
        sq_sum = fbank.pow(2).sum(0)
        M = fbank.size(0)

        E_x = E_x * (N / (N + M)) + sum / (N + M)
        E_x_2 = E_x_2 * (N / (N + M)) + sq_sum / (N + M)
        N += M

        if idx % 100 == 0:
            logger.info(f"Processed {idx}")

    if multiple_parts:
        return E_x, E_x_2, N
    else:
        return E_x, (E_x_2 - E_x**2) ** 0.5


def get_dataset(args):
    # if args.model_type == MODEL_TYPE_LIBRISPEECH:
    #     return torch.utils.data.ConcatDataset(
    #         [
    #             torchaudio.datasets.LIBRISPEECH(args.dataset_path, url="train-clean-360"),
    #             torchaudio.datasets.LIBRISPEECH(args.dataset_path, url="train-clean-100"),
    #             torchaudio.datasets.LIBRISPEECH(args.dataset_path, url="train-other-500"),
    #         ]
    #     )
    # elif args.model_type == MODEL_TYPE_TEDLIUM3:
    #     return torchaudio.datasets.TEDLIUM(args.dataset_path, release="release3", subset="train")
    # elif args.model_type == MODEL_TYPE_MUSTC:
    #     return MUSTC(args.dataset_path, subset="train")
    # else:
    #     raise ValueError(f"Encountered unsupported model type {args.model_type}.")

    return LongAudioDataset(
        root = args.dataset_path, # "/scratch4/skhudan1/rhuang25/data/seekingalpha/audio2019/",
    )

    # return torch.utils.data.ConcatDataset(
    #     [
    #         # torchaudio.datasets.LIBRISPEECH(args.dataset_path, url="train-clean-360"),
    #         torchaudio.datasets.LIBRISPEECH(args.dataset_path, url="train-clean-100"),
    #         # torchaudio.datasets.LIBRISPEECH(args.dataset_path, url="train-other-500"),
    #     ]
    # )


def post_process_parts():
    from glob import glob

    E_x = 0
    E_x_2 = 0
    N = 0

    files = glob("/scratch4/skhudan1/rhuang25/data/seekingalpha/icefall_data/global_stats.part*-4.json")
    print(f"len(files) = {len(files)}")
    for global_stats_path in files:
        with open(global_stats_path) as f:
            blob = json.loads(f.read())

            _E_x = torch.tensor(blob["E_x"])
            _E_x_2 = torch.tensor(blob["E_x_2"])
            _N = torch.tensor(blob["N"])

            E_x = E_x * (N / (N + _N)) + _E_x * (_N / (N + _N))
            E_x_2 = E_x_2 * (N / (N + _N)) + _E_x_2 * (_N / (N + _N))
            N += _N
    return E_x, (E_x_2 - E_x**2) ** 0.5


def cli_main():
    args = parse_args()
    dataset = get_dataset(args)
    dataset.audio_files.sort()
    
    if args.part is None:
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=args.nj)  # https://github.com/lhotse-speech/lhotse/discussions/440
        mean, stddev = generate_statistics(iter(dataloader), len_dataset=len(dataset))
        print(f"mean.shape: {mean.shape}")
        print(f"stddev.shape: {stddev.shape}")
        json_str = json.dumps({"mean": mean.tolist(), "invstddev": (1 / stddev).tolist()}, indent=2)

        with open(args.output_path, "w") as f:
            f.write(json_str)
        print(f"Saved to: {args.output_path}")
    else:
        i_chunk, n_chunks = args.part.split("/")
        i_chunk, n_chunks = int(i_chunk), int(n_chunks)
        i_chunk = i_chunk % n_chunks
        chunk_size = int(len(dataset.audio_files) / n_chunks) + 1
        i_start = chunk_size * i_chunk

        print(f"[{args.part}] i_start: {i_start}, chunk_size {chunk_size}, i_end: {i_start + chunk_size}")
        dataset.audio_files = dataset.audio_files[i_start: i_start + chunk_size]
        print(f"len(dataset.audio_files) = {len(dataset.audio_files)}")
        
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=args.nj, shuffle=True)  # https://github.com/lhotse-speech/lhotse/discussions/440
        E_x, E_x_2, N = generate_statistics(iter(dataloader), len_dataset=len(dataset), multiple_parts=True)
        json_str = json.dumps({"E_x": E_x.tolist(), "E_x_2": E_x_2.tolist(), "N": N}, indent=2)

        with open(args.output_path, "w") as f:
            f.write(json_str)
        print(f"Saved to: {args.output_path}")


if __name__ == "__main__":
    cli_main()