#!/usr/bin/env python3
# Copyright      2021-2022  Xiaomi Corp.   (authors: Fangjun Kuang,
#                                                    Zengwei)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This script loads a checkpoint and uses it to decode waves.
You can generate the checkpoint with the following command:

./zipformer_mmi/export.py \
  --exp-dir ./zipformer_mmi/exp \
  --tokens data/lang_bpe_500/tokens.txt \
  --epoch 20 \
  --avg 10

Usage of this script:

(1) 1best
./zipformer_mmi/pretrained.py \
    --checkpoint ./zipformer_mmi/exp/pretrained.pt \
    --tokens data/lang_bpe_500/tokens.txt \
    --method 1best \
    /path/to/foo.wav \
    /path/to/bar.wav
(2) nbest
./zipformer_mmi/pretrained.py \
    --checkpoint ./zipformer_mmi/exp/pretrained.pt \
    --tokens data/lang_bpe_500/tokens.txt \
    --nbest-scale 1.2 \
    --method nbest \
    /path/to/foo.wav \
    /path/to/bar.wav
(3) nbest-rescoring-LG
./zipformer_mmi/pretrained.py \
    --checkpoint ./zipformer_mmi/exp/pretrained.pt \
    --tokens data/lang_bpe_500/tokens.txt \
    --nbest-scale 1.2 \
    --method nbest-rescoring-LG \
    /path/to/foo.wav \
    /path/to/bar.wav
(4) nbest-rescoring-3-gram
./zipformer_mmi/pretrained.py \
    --checkpoint ./zipformer_mmi/exp/pretrained.pt \
    --tokens data/lang_bpe_500/tokens.txt \
    --nbest-scale 1.2 \
    --method nbest-rescoring-3-gram \
    /path/to/foo.wav \
    /path/to/bar.wav
(5) nbest-rescoring-4-gram
./zipformer_mmi/pretrained.py \
    --checkpoint ./zipformer_mmi/exp/pretrained.pt \
    --tokens data/lang_bpe_500/tokens.txt \
    --nbest-scale 1.2 \
    --method nbest-rescoring-4-gram \
    /path/to/foo.wav \
    /path/to/bar.wav


You can also use `./zipformer_mmi/exp/epoch-xx.pt`.

Note: ./zipformer_mmi/exp/pretrained.pt is generated by
./zipformer_mmi/export.py
"""


import argparse
import logging
import math
from pathlib import Path
from typing import List

import k2
import kaldifeat
import torch
import torchaudio
from decode import get_decoding_params
from torch.nn.utils.rnn import pad_sequence
from train import add_model_arguments, get_ctc_model, get_params

from icefall.decode import (
    get_lattice,
    nbest_decoding,
    nbest_rescore_with_LM,
    one_best_decoding,
)
from icefall.mmi_graph_compiler import MmiTrainingGraphCompiler
from icefall.utils import get_texts, num_tokens


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint. "
        "The checkpoint is assumed to be saved by "
        "icefall.checkpoint.save_checkpoint().",
    )

    parser.add_argument(
        "--tokens",
        type=str,
        help="""Path to tokens.txt.""",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="1best",
        help="""Decoding method. Use HP as decoding graph, where H is
        ctc_topo and P is token-level bi-gram lm.
        Supported values are:
        - (1) 1best. Extract the best path from the decoding lattice as the
          decoding result.
        - (2) nbest. Extract n paths from the decoding lattice; the path
          with the highest score is the decoding result.
        - (4) nbest-rescoring-LG. Extract n paths from the decoding lattice,
          rescore them with an word-level 3-gram LM, the path with the
          highest score is the decoding result.
        - (5) nbest-rescoring-3-gram. Extract n paths from the decoding
          lattice, rescore them with an token-level 3-gram LM, the path with
          the highest score is the decoding result.
        - (6) nbest-rescoring-4-gram. Extract n paths from the decoding
          lattice, rescore them with an token-level 4-gram LM, the path with
          the highest score is the decoding result.
        """,
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="The sample rate of the input sound file",
    )

    parser.add_argument(
        "--lang-dir",
        type=Path,
        default="data/lang_bpe_500",
        help="The lang dir containing word table and LG graph",
    )

    parser.add_argument(
        "--num-paths",
        type=int,
        default=100,
        help="""Number of paths for n-best based decoding method.
        Used only when "method" is one of the following values:
        nbest, nbest-rescoring, and nbest-oracle
        """,
    )

    parser.add_argument(
        "--nbest-scale",
        type=float,
        default=1.2,
        help="""The scale to be applied to `lattice.scores`.
        It's needed if you use any kinds of n-best based rescoring.
        Used only when "method" is one of the following values:
        nbest, nbest-rescoring, and nbest-oracle
        A smaller value results in more unique paths.
        """,
    )

    parser.add_argument(
        "--ngram-lm-scale",
        type=float,
        default=0.1,
        help="""
        Used when method is nbest-rescoring-LG, nbest-rescoring-3-gram,
        and nbest-rescoring-4-gram.
        It specifies the scale for n-gram LM scores.
        (Note: You need to tune it on a dataset.)
        """,
    )

    parser.add_argument(
        "--hp-scale",
        type=float,
        default=1.0,
        help="""The scale to be applied to `ctc_topo_P.scores`.
        """,
    )

    parser.add_argument(
        "sound_files",
        type=str,
        nargs="+",
        help="The input sound file(s) to transcribe. "
        "Supported formats are those supported by torchaudio.load(). "
        "For example, wav and flac are supported. "
        "The sample rate has to be 16kHz.",
    )

    add_model_arguments(parser)

    return parser


def read_sound_files(
    filenames: List[str], expected_sample_rate: float
) -> List[torch.Tensor]:
    """Read a list of sound files into a list 1-D float32 torch tensors.
    Args:
      filenames:
        A list of sound filenames.
      expected_sample_rate:
        The expected sample rate of the sound files.
    Returns:
      Return a list of 1-D float32 torch tensors.
    """
    ans = []
    for f in filenames:
        wave, sample_rate = torchaudio.load(f)
        assert (
            sample_rate == expected_sample_rate
        ), f"expected sample rate: {expected_sample_rate}. Given: {sample_rate}"
        # We use only the first channel
        ans.append(wave[0])
    return ans


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()

    params = get_params()
    # add decoding params
    params.update(get_decoding_params())
    params.update(vars(args))

    # Load tokens.txt here
    token_table = k2.SymbolTable.from_file(params.tokens)

    # Load id of the <blk> token and the vocab size
    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = token_table["<blk>"]
    params.unk_id = token_table["<unk>"]
    params.vocab_size = num_tokens(token_table) + 1  # +1 for <blk>

    logging.info(f"{params}")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"device: {device}")

    logging.info("Creating model")
    model = get_ctc_model(params)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    model.eval()
    model.device = device

    logging.info("Constructing Fbank computer")
    opts = kaldifeat.FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = params.sample_rate
    opts.mel_opts.num_bins = params.feature_dim
    opts.mel_opts.high_freq = -400

    fbank = kaldifeat.Fbank(opts)

    logging.info(f"Reading sound files: {params.sound_files}")
    waves = read_sound_files(
        filenames=params.sound_files, expected_sample_rate=params.sample_rate
    )
    waves = [w.to(device) for w in waves]

    logging.info("Decoding started")
    features = fbank(waves)
    feature_lengths = [f.size(0) for f in features]

    features = pad_sequence(features, batch_first=True, padding_value=math.log(1e-10))
    feature_lengths = torch.tensor(feature_lengths, device=device)

    mmi_graph_compiler = MmiTrainingGraphCompiler(
        params.lang_dir,
        uniq_filename="lexicon.txt",
        device=device,
        oov="<UNK>",
        sos_id=1,
        eos_id=1,
    )
    HP = mmi_graph_compiler.ctc_topo_P
    HP.scores *= params.hp_scale
    if not hasattr(HP, "lm_scores"):
        HP.lm_scores = HP.scores.clone()

    def token_ids_to_words(token_ids: List[int]) -> str:
        text = ""
        for i in token_ids:
            text += token_table[i]
        return text.replace("▁", " ").strip()

    method = params.method
    assert method in (
        "1best",
        "nbest",
        "nbest-rescoring-LG",  # word-level 3-gram lm
        "nbest-rescoring-3-gram",  # token-level 3-gram lm
        "nbest-rescoring-4-gram",  # token-level 4-gram lm
    )
    # loading language model for rescoring
    LM = None
    if method == "nbest-rescoring-LG":
        lg_filename = params.lang_dir / "LG.pt"
        logging.info(f"Loading {lg_filename}")
        LG = k2.Fsa.from_dict(torch.load(lg_filename, map_location=device))
        LG = k2.Fsa.from_fsas([LG]).to(device)
        LG.lm_scores = LG.scores.clone()
        LM = LG
    elif method in ["nbest-rescoring-3-gram", "nbest-rescoring-4-gram"]:
        order = method[-6]
        assert order in ("3", "4")
        order = int(order)
        logging.info(f"Loading pre-compiled {order}gram.pt")
        d = torch.load(params.lang_dir / f"{order}gram.pt", map_location=device)
        G = k2.Fsa.from_dict(d)
        G.lm_scores = G.scores.clone()
        LM = G

    # Encoder forward
    nnet_output, encoder_out_lens = model(x=features, x_lens=feature_lengths)

    batch_size = nnet_output.shape[0]
    supervision_segments = torch.tensor(
        [
            [i, 0, feature_lengths[i] // params.subsampling_factor]
            for i in range(batch_size)
        ],
        dtype=torch.int32,
    )

    lattice = get_lattice(
        nnet_output=nnet_output,
        decoding_graph=HP,
        supervision_segments=supervision_segments,
        search_beam=params.search_beam,
        output_beam=params.output_beam,
        min_active_states=params.min_active_states,
        max_active_states=params.max_active_states,
        subsampling_factor=params.subsampling_factor,
    )

    if method in ["1best", "nbest"]:
        if method == "1best":
            best_path = one_best_decoding(
                lattice=lattice, use_double_scores=params.use_double_scores
            )
        else:
            best_path = nbest_decoding(
                lattice=lattice,
                num_paths=params.num_paths,
                use_double_scores=params.use_double_scores,
                nbest_scale=params.nbest_scale,
            )
    else:
        best_path_dict = nbest_rescore_with_LM(
            lattice=lattice,
            LM=LM,
            num_paths=params.num_paths,
            lm_scale_list=[params.ngram_lm_scale],
            nbest_scale=params.nbest_scale,
        )
        best_path = next(iter(best_path_dict.values()))

    # Note: `best_path.aux_labels` contains token IDs, not word IDs
    # since we are using HP, not HLG here.
    #
    # token_ids is a lit-of-list of IDs
    token_ids = get_texts(best_path)
    hyps = [token_ids_to_words(ids) for ids in token_ids]

    s = "\n"
    for filename, hyp in zip(params.sound_files, hyps):
        s += f"{filename}:\n{hyp}\n\n"
    logging.info(s)

    logging.info("Decoding Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
