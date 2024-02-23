# Usage:
# python /exp/rhuang/meta/icefall/egs/librispeech/ASR/conformer_ctc2_noseg/scripts/get_audacy_labels.py /exp/rhuang/meta/icefall/egs/librispeech/ASR/conformer_ctc2_noseg/exp/exp_seed_small_model/ali_-18.0_segment15_overlap2/4757/1811/1811.pt

import sys
sys.path.append("/exp/rhuang/meta/icefall/egs/librispeech/ASR/conformer_ctc2_noseg/")

import logging
import argparse
import torch
from pathlib import Path
from data_long_dataset import *
from alignment import align_long_text, handle_failed_groups, to_audacity_label_format


logging.basicConfig(
    format = "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
    level = 10
)

def parse_opts():
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('pt', type=str, default=None, help='')

    opts = parser.parse_args()
    logging.info(f"Parameters: {vars(opts)}")
    return opts


def main(opts):
    long_dataset = LibrispeechLongAudioDataset(
        root = "/exp/rhuang/meta/icefall/egs/librispeech/ASR/download/",
        # skip_loading_audio = False,
        # skip_text_normalization = True,
        # manifest_file = None,
    )

    long_dataset_index = {Path(long_dataset.manifest[i][0]).parent.stem: i for i in range(len(long_dataset))}  # chapter id => index

    chapter_id = Path(opts.pt).parent.stem
    waveform, sample_rate, text, speaker_id, chapter_id, meta_data  = long_dataset[long_dataset_index[chapter_id]]
    # waveform, sample_rate, text, speaker_id, chapter_id, meta_data  = long_dataset[long_dataset_index[Path(meta_data["audio_path"]).parent.stem]]

    rs = torch.load(opts.pt)
    alignment_results = rs["alignment_results"]

    class Params:
        def __init__(self):
            self.subsampling_factor = 4

    frame_rate = 0.01
    audacity_labels_str = to_audacity_label_format(Params(), frame_rate, alignment_results, text)
    audacity_path = str(opts.pt)[:-3] + "_audacity.txt"
    with open(audacity_path, "w") as fout:
        print(audacity_labels_str, file=fout)
    
    print((Path(long_dataset.root) / meta_data["audio_path"]).absolute())
    print(Path(audacity_path).absolute())


if __name__ == '__main__':
    opts = parse_opts()

    main(opts)