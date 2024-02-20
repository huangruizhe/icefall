#!/usr/bin/env python3
# Copyright 2021 Xiaomi Corporation (Author: Liyong Guo,
#                                            Fangjun Kuang,
#                                            Quandong Wang)
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


import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from asr_datamodule import LibriSpeechAsrDataModule
from conformer import Conformer

from icefall.bpe_graph_compiler import BpeCtcTrainingGraphCompiler
from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.decode import (
    get_lattice,
    nbest_decoding,
    nbest_oracle,
    one_best_decoding,
    rescore_with_attention_decoder,
    rescore_with_n_best_list,
    rescore_with_rnn_lm,
    rescore_with_whole_lattice,
)
from icefall.env import get_env_info
from icefall.lexicon import Lexicon
from icefall.rnn_lm.model import RnnLmModel
from icefall.utils import (
    AttributeDict,
    get_texts,
    load_averaged_model,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
)

from data_long_dataset import *
from data_transforms import *
from factor_transducer import *
from no_seg_utils import *
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from lhotse.utils import fix_random_seed
from icefall.dist import cleanup_dist, setup_dist


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=77,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=15,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="alignment",
        help="""Decoding method.
        Supported values are:
            - (0) ctc-decoding. Use CTC decoding. It uses a sentence piece
              model, i.e., lang_dir/bpe.model, to convert word pieces to words.
              It needs neither a lexicon nor an n-gram LM.
            - (1) ctc-greedy-search. It only use CTC output and a sentence piece
              model for decoding. It produces the same results with ctc-decoding.
            - (2) 1best. Extract the best path from the decoding lattice as the
              decoding result.
            - (3) nbest. Extract n paths from the decoding lattice; the path
              with the highest score is the decoding result.
            - (4) nbest-rescoring. Extract n paths from the decoding lattice,
              rescore them with an n-gram LM (e.g., a 4-gram LM), the path with
              the highest score is the decoding result.
            - (5) whole-lattice-rescoring. Rescore the decoding lattice with an
              n-gram LM (e.g., a 4-gram LM), the best path of rescored lattice
              is the decoding result.
            - (6) attention-decoder. Extract n paths from the LM rescored
              lattice, the path with the highest score is the decoding result.
            - (7) rnn-lm. Rescoring with attention-decoder and RNN LM. We assume
              you have trained an RNN LM using ./rnn_lm/train.py
            - (8) nbest-oracle. Its WER is the lower bound of any n-best
              rescoring method can achieve. Useful for debugging n-best
              rescoring method.
        """,
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--num-decoder-layers",
        type=int,
        default=6,
        help="""Number of decoder layer of transformer decoder.
        Setting this to 0 will not create the decoder at all (pure CTC model)
        """,
    )

    parser.add_argument(
        "--num-paths",
        type=int,
        default=100,
        help="""Number of paths for n-best based decoding method.
        Used only when "method" is one of the following values:
        nbest, nbest-rescoring, attention-decoder, rnn-lm, and nbest-oracle
        """,
    )

    parser.add_argument(
        "--nbest-scale",
        type=float,
        default=0.5,
        help="""The scale to be applied to `lattice.scores`.
        It's needed if you use any kinds of n-best based rescoring.
        Used only when "method" is one of the following values:
        nbest, nbest-rescoring, attention-decoder, rnn-lm, and nbest-oracle
        A smaller value results in more unique paths.
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="conformer_ctc2/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--lang-dir",
        type=str,
        default="data/lang_bpe_500",
        help="The lang dir",
    )

    parser.add_argument(
        "--lm-dir",
        type=str,
        default="data/lm",
        help="""The n-gram LM dir.
        It should contain either G_4_gram.pt or G_4_gram.fst.txt
        """,
    )

    parser.add_argument(
        "--rnn-lm-exp-dir",
        type=str,
        default="rnn_lm/exp",
        help="""Used only when --method is rnn-lm.
        It specifies the path to RNN LM exp dir.
        """,
    )

    parser.add_argument(
        "--rnn-lm-epoch",
        type=int,
        default=7,
        help="""Used only when --method is rnn-lm.
        It specifies the checkpoint to use.
        """,
    )

    parser.add_argument(
        "--rnn-lm-avg",
        type=int,
        default=2,
        help="""Used only when --method is rnn-lm.
        It specifies the number of checkpoints to average.
        """,
    )

    parser.add_argument(
        "--rnn-lm-embedding-dim",
        type=int,
        default=2048,
        help="Embedding dim of the model",
    )

    parser.add_argument(
        "--rnn-lm-hidden-dim",
        type=int,
        default=2048,
        help="Hidden dim of the model",
    )

    parser.add_argument(
        "--rnn-lm-num-layers",
        type=int,
        default=4,
        help="Number of RNN layers the model",
    )
    parser.add_argument(
        "--rnn-lm-tie-weights",
        type=str2bool,
        default=False,
        help="""True to share the weights between the input embedding layer and the
        last output linear layer
        """,
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            # parameters for conformer
            "feature_dim": 80,
            "subsampling_factor": 4,
            "encoder_dim": 128,
            "nhead": 8,
            "dim_feedforward": 512,
            "num_encoder_layers": 8,
            # parameters for decoding
            "search_beam": 20,
            "output_beam": 8,
            "min_active_states": 30,
            "max_active_states": 10000,
            "use_double_scores": True,
            "env_info": get_env_info(),
            # new parameters
            "blank_id": 0,
            "vocab_size": 500,
            "my_args": None,
            "search_beam": 20,
            "output_beam": 8,
            "min_active_states": 30,
            "max_active_states": 10000,
        }
    )
    return params


def ctc_greedy_search(
    nnet_output: torch.Tensor,
    memory: torch.Tensor,
    memory_key_padding_mask: torch.Tensor,
) -> List[List[int]]:
    """Apply CTC greedy search

     Args:
         speech (torch.Tensor): (batch, max_len, feat_dim)
         speech_length (torch.Tensor): (batch, )
    Returns:
         List[List[int]]: best path result
    """
    batch_size = memory.shape[1]
    # Let's assume B = batch_size
    encoder_out = memory
    encoder_mask = memory_key_padding_mask
    maxlen = encoder_out.size(0)

    ctc_probs = nnet_output  # (B, maxlen, vocab_size)
    topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
    topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
    topk_index = topk_index.masked_fill_(encoder_mask, 0)  # (B, maxlen)
    hyps = [hyp.tolist() for hyp in topk_index]
    scores = topk_prob.max(1)
    hyps = [remove_duplicates_and_blank(hyp) for hyp in hyps]
    return hyps, scores


def remove_duplicates_and_blank(hyp: List[int]) -> List[int]:
    # from https://github.com/wenet-e2e/wenet/blob/main/wenet/utils/common.py
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != 0:
            new_hyp.append(hyp[cur])
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return new_hyp


def get_str_by_range(book, rgs):
    tlong = book.split()
    # Note: we need to shift the index by one
    texts = [" ".join(tlong[rg[0] - 1: rg[-1] - 1]) for rg in rgs]
    return texts


def align_one_batch(
    batch: dict,
    params: AttributeDict,
    model: nn.Module,
    sp: Optional[spm.SentencePieceProcessor],
):  # -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    
    feature = batch["inputs"]
    assert feature.ndim == 3
    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]

    ctc_output, memory, memory_key_padding_mask = model(feature, supervisions)
    # nnet_output is (N, T, C)

    # hyps, _ = ctc_greedy_search(
    #     nnet_output,
    #     memory,
    #     memory_key_padding_mask,
    # )
    # hyps = sp.decode(hyps)

    y_long = [batch["y_long"]]
    book = supervisions["text"]

    if False:
        batch_wer = get_batch_wer(params, ctc_output, batch, sp, decoding_graph=y_long)
    else:  # For make_factor_transducer4
        lattice, best_paths = get_lattice_and_best_paths(params, ctc_output, batch, sp, decoding_graph=y_long)
        lattice = lattice.detach().to('cpu')
        best_paths = best_paths.detach().to('cpu')

        decoding_results = get_texts_with_timestamp(best_paths)
        token_ids_indices = decoding_results.hyps
        timestamps = decoding_results.timestamps
        token_ids_indices, _ = handle_emtpy_texts(token_ids_indices, rg=[1,1])
        # rs_texts = get_str_by_range(book, token_ids_indices)
        token_ids_indices = [list(map(lambda x: x - 1, rg)) for rg in token_ids_indices]
    
    key = "alignment-with-long-text"
    rs = token_ids_indices
    return {key: rs}


def post_process():
    pass


def align_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    sp: Optional[spm.SentencePieceProcessor],
    rank: int,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    
    model.eval()

    target_sample_rate = 16000
    resample_transform = Resample(target_sample_rate=target_sample_rate)
    fbank_transform = LhotseFbank()  # 16kHz

    frame_rate = 0.01
    segment_size = 30  # 30 seconds
    overlap = 2  # 2 seconds
    step = segment_size - overlap
    frame_subsampling_rate = params.subsampling_factor
    word_start_symbols = {i for i in range(sp.vocab_size()) if sp.id_to_piece(i).startswith('▁')}

    max_duration = 1200  # seconds
    batch_size = int(max_duration/segment_size)

    segment_size = int(segment_size / frame_rate)
    overlap = int(overlap / frame_rate)
    step = int(step / frame_rate)

    # Note:
    # 1) if the beginning of the audio is not in the text, the audio will get poor alignment

    results = defaultdict(list)
    for batch_idx, batch in enumerate(dl):
        # Each batch contains a single long audio and its long text
        waveform, sample_rate, text, speaker_id, chapter_id, meta_data = batch
        waveform = waveform.squeeze()
        text = text[0]

        # Step (0): get the factor transducer for the long text
        y_long = make_factor_transducer4_skip(
            sp.encode(text, out_type=int), 
            word_start_symbols=word_start_symbols, 
            return_str=False,
            skip_penalty=-0.5,  # tie breaking and avoid long skips
            return_penalty=-18.0,   # ok, now we allow return to the start states from word ends, but with big penalty (in the log domain; it seems -15~-20 is good)
            # return_penalty=None,    # no "return arc" is allowed
        )
        # y_long.shape, y_long.num_arcs

        # Step (1): resample the long audio to 16kHz
        if sample_rate != target_sample_rate:
            waveform = resample_transform(waveform, sample_rate)
            sample_rate = target_sample_rate

        # Step (2): feature extraction on the long audio
        features = fbank_transform(waveform, sampling_rate=sample_rate)
        features = features.unsqueeze(0)  # (N, T, D)
        # features.shape
        waveform = None

        # Step (3): cut the long audio features into overlapping segments
        padding_size = (segment_size - (features.size(1) % segment_size)) % segment_size
        features_padded = torch.nn.functional.pad(features, (0, 0, 0, padding_size))  # Pad the tensor with zeros
        features_padded = features_padded.unfold(dimension=1, size=segment_size, step=step)
        features_padded = features_padded.permute(0, 1, 3, 2)
        # features_padded is of shape (N, L, segment_size, D), where L is the number of segments.
        segment_lengths = [segment_size] * features_padded.size(1)
        segment_lengths[-1] -= padding_size

        features_padded = features_padded[0]  # (L, segment_size, D)
        assert len(segment_lengths) == features_padded.size(0)

        # Step (4): do alignment for batches
        for i in range(0, features_padded.size(0), batch_size):
            batch_features = features_padded[i: i+batch_size]
            actual_batch_size = batch_features.size(0)
            batch = {
                "supervisions": {
                    "text": text,
                    "cut": None,
                    "sequence_idx": torch.arange(actual_batch_size),
                    "start_frame": torch.zeros(actual_batch_size),
                    "num_frames": torch.Tensor(segment_lengths[i: i+batch_size]).int(),
                },
                "inputs": batch_features,
                "y_long": y_long,
            }

            hyps_dict = align_one_batch(batch, params, model, sp)
            
            k = "alignment-with-long-text"
            results[k].extend(hyps_dict[k])  # TODO
        
        # Step (5): post-process the alignment for each audio, save results

        if batch_idx % 10 == 0:
            batch_str = f"{batch_idx}/{len(dl)}"
            logging.info(f"batch {batch_str}")
        
        # logging.info(f"batch_idx={batch_idx}")
        # if batch_idx > 200:
        #     break
    return results



def save_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
):
    if params.method in ("attention-decoder", "rnn-lm"):
        # Set it to False since there are too many logs.
        enable_log = False
    else:
        enable_log = True
    test_set_wers = dict()
    for key, results in results_dict.items():
        recog_path = params.exp_dir / f"recogs-{test_set_name}-{key}.txt"
        results = sorted(results)
        store_transcripts(filename=recog_path, texts=results)
        if enable_log:
            logging.info(f"The transcripts are stored in {recog_path}")

        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = params.exp_dir / f"errs-{test_set_name}-{key}.txt"
        with open(errs_filename, "w") as f:
            wer = write_error_stats(
                f, f"{test_set_name}-{key}", results, enable_log=enable_log
            )
            test_set_wers[key] = wer

        if enable_log:
            logging.info("Wrote detailed error stats to {}".format(errs_filename))

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])
    errs_info = params.exp_dir / f"wer-summary-{test_set_name}.txt"
    with open(errs_info, "w") as f:
        print("settings\tWER", file=f)
        for key, val in test_set_wers:
            print("{}\t{}".format(key, val), file=f)

    s = "\nFor {}, WER of different settings are:\n".format(test_set_name)
    note = "\tbest for {}".format(test_set_name)
    for key, val in test_set_wers:
        s += "{}\t{}{}\n".format(key, val, note)
        note = ""
    logging.info(s)


@torch.no_grad()
def run(rank, world_size, args):
    params = get_params()
    params.update(vars(args))

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/alignment/log-align")
    logging.info("Alignment started")
    logging.info(params)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.vocab_size = sp.get_piece_size()

    logging.info("About to create model")
    model = Conformer(
        num_features=params.feature_dim,
        nhead=params.nhead,
        d_model=params.encoder_dim,
        num_classes=params.vocab_size,
        subsampling_factor=params.subsampling_factor,
        num_encoder_layers=params.num_encoder_layers,
        num_decoder_layers=params.num_decoder_layers,
    )

    if not params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
        elif params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
    else:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg + 1
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg + 1:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            filename_start = filenames[-1]
            filename_end = filenames[0]
            logging.info(
                "Calculating the averaged model over iteration checkpoints"
                f" from {filename_start} (excluded) to {filename_end}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )
        else:
            assert params.avg > 0, params.avg
            start = params.epoch - params.avg
            assert start >= 1, start
            filename_start = f"{params.exp_dir}/epoch-{start}.pt"
            filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
            logging.info(
                f"Calculating the averaged model over epoch range from "
                f"{start} (excluded) to {params.epoch}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )

    model.to(device)
    model.eval()
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    long_dataset = LibrispeechLongAudioDataset(
        root = "/exp/rhuang/meta/icefall/egs/librispeech/ASR/download/",
    )
    # waveform, sample_rate, text, speaker_id, audio_id, meta_data = \
    #     long_dataset[0]    
    # print(f"[{audio_id}] len(text) = {len(text)}")
    
    if world_size > 1:
        sampler = DistributedSampler(long_dataset)
    else:
        sampler = None

    dataloader = DataLoader(
        long_dataset, 
        batch_size=1,
        shuffle=False, 
        num_workers=0,
        sampler=sampler,
        # prefetch_factor=2,
    )

    test_sets = ["libri_long"]
    test_dls = [dataloader]

    for test_set, test_dl in zip(test_sets, test_dls):
        results_dict = align_dataset(
            dl=test_dl,
            params=params,
            model=model,
            sp=sp,
            rank=rank,
        )

        save_results(params=params, test_set_name=test_set, results_dict=results_dict)

    logging.info("Done!")


def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    args.lang_dir = Path(args.lang_dir)
    args.lm_dir = Path(args.lm_dir)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()

    # import cProfile
    # cProfile.run('main()', "output.prof")
    # import pstats
    # p = pstats.Stats('output.prof')
    # p.sort_stats('cumulative').print_stats(10)  # Print the 10 most time-consuming functions
