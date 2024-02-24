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
from torch.nn.utils.rnn import pad_sequence
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
from alignment import align_long_text, handle_failed_groups, to_audacity_label_format
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

    parser.add_argument(
        "--part",
        type=str,
        default=None,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--return-penalty",
        type=float,
        default=None,
        help="",
    )

    parser.add_argument(
        "--segment-size",
        type=int,
        default=15,  # the reasonably shorter the better; 30 is worse than 15
        help="in seconds",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=2,
        help="in seconds",
    )

    parser.add_argument(
        "--skip-penalty",
        type=float,
        default=-0.5,
        help="Typically it shoud be smaller than 0.0 (negative)",
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
            # "search_beam": 15,  # For bad chapters
            # "output_beam": 6,
            # "min_active_states": 30,
            # "max_active_states": 10000,
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
    batch_features: torch.Tensor,
    text,
    y_long,
    segment_lengths,
    params: AttributeDict,
    model: nn.Module,
    sp: Optional[spm.SentencePieceProcessor],
    is_standard_forced_alignment: bool = False,
):  # -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    
    actual_batch_size = batch_features.size(0)
    batch = {
        "supervisions": {
            "text": text,
            "cut": None,
            "sequence_idx": torch.arange(actual_batch_size),
            "start_frame": torch.zeros(actual_batch_size),
            "num_frames": torch.Tensor(segment_lengths).int(),
        },
        "inputs": batch_features,
        "y_long": y_long,
    }


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

    y_long = batch["y_long"]
    # book = supervisions["text"]

    if False:
        batch_wer = get_batch_wer(params, ctc_output, batch, sp, decoding_graph=y_long)
    else:  # For make_factor_transducer4
        lattice, best_paths = get_lattice_and_best_paths(params, ctc_output, batch, sp, decoding_graph=y_long)
        lattice = lattice.detach().to('cpu')
        best_paths = best_paths.detach().to('cpu')

        decoding_results = get_texts_with_timestamp(best_paths)
        token_ids_indices = decoding_results.hyps
        timestamps = decoding_results.timestamps
        if not is_standard_forced_alignment:
            token_ids_indices, _ = handle_emtpy_texts(token_ids_indices, rg=[1,1])
            # rs_texts = get_str_by_range(book, token_ids_indices)
            token_ids_indices = [list(map(lambda x: x - 1, rg)) for rg in token_ids_indices]
    
    return token_ids_indices, timestamps


def realign(
    params: AttributeDict,
    model: nn.Module,
    sp: Optional[spm.SentencePieceProcessor],
    alignment_results, 
    to_realign,
    text,
    features, # (1, T, D)  # for one audio
    frame_rate,
    word_start_symbols,
):
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    assert features.ndim == 3
    features = features.squeeze(0)

    text = text.split()

    # to_realign_lengths = [ee - ss for ss, ee in to_realign]
    failed = []

    # Get the unaligned segments
    segments = list()
    for ss, ee in to_realign:  # ss, ee are word indices in the long text
        ss_t = alignment_results[ss] * params.subsampling_factor  # frame index in the long audio
        ee_t = alignment_results[ee] * params.subsampling_factor

        # We have removed too short segments here in to_realign
        # logging.info(f"Realigning: [{ss_t}, {ee_t}] text {ss}:{ee}")
        if ee_t - ss_t < 20:
            failed.append((ss, ee))
            continue

        segments.append(
            (features[ss_t: ee_t], text[ss: ee], (ss, ee))
        )
    
    segments = sorted(segments, key=lambda x: x[0].size(0))

    # Batchify the segments
    batches = [[]]
    cur_batch_duration = 0
    for seg in segments:
        f, _, _ = seg
        batches[-1].append(seg)
        cur_batch_duration += f.size(0) * frame_rate
        # These unaligned parts can be hard. So reduce the batch size by a half
        if cur_batch_duration > params.max_duration * 0.5:
            batches.append([])
            cur_batch_duration = 0

    if len(batches[-1]) == 0:
        batches = batches[:-1]

    # Forced alignment
    for batch in batches:
        batch_segment_lengths = [seg[0].size(0) for seg in batch]
        batch_features = pad_sequence([seg[0] for seg in batch], batch_first=True)
        ys = sp.encode([" ".join(seg[1]) for seg in batch], out_type=int)
        y_long = k2.ctc_graph(ys, modified=False, device=device)
        y_long = [y_long[i] for i in range(y_long.shape[0])]

        hyps, timestamps = align_one_batch(batch_features, text, y_long, batch_segment_lengths, params, model, sp, is_standard_forced_alignment=True)
        
        for seg, hyp, times in zip(batch, hyps, timestamps):
            times = [t for h, t in zip(hyp, times) if h in word_start_symbols]
            ss, ee = seg[2]
            ss_t = alignment_results[ss]
            for i, t in zip(range(ss, ee), times):
                if i not in alignment_results:
                    alignment_results[i] = ss_t + t
                else:
                    alignment_results[i] = int((alignment_results[i] + ss_t + t) / 2)

    return failed


def align_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    sp: Optional[spm.SentencePieceProcessor],
    rank: int,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    
    model.eval()
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    target_sample_rate = 16000
    resample_transform = Resample(target_sample_rate=target_sample_rate)
    fbank_transform = LhotseFbank()  # 16kHz

    frame_rate = 0.01  # from feature extraction
    segment_size = params.segment_size  # 30 seconds
    overlap = params.overlap  # 2 seconds
    step = segment_size - overlap
    frame_subsampling_rate = params.subsampling_factor
    word_start_symbols = {i for i in range(sp.vocab_size()) if sp.id_to_piece(i).startswith('▁')}

    max_duration = params.max_duration  # seconds
    batch_size = int(max_duration/segment_size)

    segment_size = int(segment_size / frame_rate)
    overlap = int(overlap / frame_rate)
    step = int(step / frame_rate)

    # Note:
    # 1) if the beginning of the audio is not in the text, the audio will get poor alignment

    for batch_idx, batch in enumerate(dl):
        # Each batch contains a single long audio and its long text
        waveform, sample_rate, text, speaker_id, chapter_id, meta_data = batch
        waveform = waveform.squeeze()
        text = text[0]

        # if "2961/960/960" in meta_data["audio_path"][0]:
        #     breakpoint()

        # Skip them, which can cause OOM or other k2 errors
        bad_chapters = {"6870", "6872", "6855", "6852", "6879", "6850", "6397", "137482", "137483", "6872", "41259", "41260", "153202", "171138"}
        bad_chapters |= {"48362", "136862", "136863", "159609", "216547", "276749", "124368", "131722", "75788", "66741", "16003", "7574", "245687"}
        bad_chapters |= {"32399", "48361", "26872", "96429", "29340", "167590", "19207", "136848", "171107", "98645", "6522", "136846", "282383"}
        bad_chapters |= {"12472", "167609", "32402", "142315", "143948", "95963", "19212", "108627", "136854", "19215", "3792", "102353", "216567"}
        bad_chapters |= {"135136", "6517", "141758", "133295", "16048", "245697", "156115", "142276", "19373", "134809", "282357", "154880", "19361"}
        bad_chapters |= {"19214", "56787", "47030", "495", "56787", "276748", "171134", "171134", "141148", "73028"}
        bad_chapters = {}
        if chapter_id[0] in bad_chapters:
            logging.info(f"Skip bad chapter: [{batch_idx}/{len(dl)}] {meta_data['audio_path']}")
            continue

        logging.info(f"Processing: [{batch_idx}/{len(dl)}] {meta_data['audio_path']}")

        audio_path = meta_data["audio_path"][0]
        # pt_path = audio_path.replace("LibriSpeechOriginal/LibriSpeech/", "LibriSpeechAligned/LibriSpeech/").replace("/books/", "/ali/")
        # pt_path = f"{dl.dataset.root}/{pt_path}"
        pt_path = audio_path.replace("LibriSpeechOriginal/LibriSpeech/", "").replace("mp3/", f"ali_{params.return_penalty}_segment{params.segment_size}_overlap{params.overlap}/")
        pt_path = f"{params.exp_dir}/{pt_path}"
        pt_path = Path(pt_path).parent / (Path(pt_path).parent.stem + ".pt")
        if pt_path.exists():
            logging.info(f"Skip: {pt_path}")
            continue

        # Step (0): get the factor transducer for the long text
        y_long = make_factor_transducer4_skip(
            sp.encode(text, out_type=int), 
            word_start_symbols=word_start_symbols, 
            return_str=False,
            skip_penalty=params.skip_penalty,  # default=-0.5, tie breaking and avoid long skips
            # return_penalty=-18.0,   # ok, now we allow return to the start states from word ends, but with big penalty (in the log domain; it seems -15~-20 is good)  
            # An additional note: -100 or None may be good on small test set with artificial long text; -18.0 is better on the training data with real long text (book)
            return_penalty=params.return_penalty,
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
        features_padded = features_padded[0]  # (L, segment_size, D)
        
        segment_lengths = [segment_size] * features_padded.size(0)
        segment_lengths[-1] -= padding_size
        segment_lengths = torch.tensor(segment_lengths)

        output_segment_lengths = torch.div(segment_lengths, params.subsampling_factor, rounding_mode="floor").int()
        output_frame_offset = torch.arange(0, segment_lengths.size(0) * step, step)
        output_frame_offset = torch.div(output_frame_offset, params.subsampling_factor, rounding_mode="floor").int()

        assert len(segment_lengths) == features_padded.size(0)
        assert len(segment_lengths) == len(output_frame_offset)

        # Discard the last chunk if it is too short
        if segment_lengths[-1] < 20:  # which is 5 frames or 0.2 secs
            features_padded = features_padded[:-1]
            segment_lengths = segment_lengths[:-1]
            output_segment_lengths = output_segment_lengths[:-1]
            output_frame_offset = output_frame_offset[:-1]

        # Step (4): do alignment for batches
        results_hyps = list()
        results_timestamps = list()
        for i in range(0, features_padded.size(0), batch_size):
            batch_features = features_padded[i: i+batch_size]
            batch_segment_lengths = segment_lengths[i: i+batch_size]

            try:
                hyps, timestamps = align_one_batch(batch_features, text, [y_long], batch_segment_lengths, params, model, sp)
                results_hyps.extend(hyps)
                results_timestamps.extend(timestamps)
            # except torch.cuda.CudaError as e:
            except Exception as e:
                import traceback
                logging.error(f"Exception occurred: {e}")
                traceback.print_exc()

                actual_batch_size = batch_features.size(0)
                half_size = actual_batch_size // 2 + 1

                hyps, timestamps = align_one_batch(batch_features[:half_size], text, y_long, batch_segment_lengths[:half_size], params, model, sp)
                results_hyps.extend(hyps)
                results_timestamps.extend(timestamps)

                hyps, timestamps = align_one_batch(batch_features[half_size:], text, y_long, batch_segment_lengths[half_size:], params, model, sp)
                results_hyps.extend(hyps)
                results_timestamps.extend(timestamps)
        
        # Step (5): post-process the alignment for each audio as well as the unaligned parts
        
        rs = {
            "meta_data": meta_data,
            "hyps": results_hyps,
            "timestamps": results_timestamps,
            "output_frame_offset": output_frame_offset,
        }

        # TODO: 
        # The beginning and ending of the audio is hard to be aligned to the book.
        # We might need to use VAD or something to handle it. Or predefine the start/end of the audio and text.
        # Ignored them for now cos we just use the alignment for ASR training.

        # alignment_results is a dict: 
        # word_index (in the long text) => frame_index (in the long audio)
        alignment_results, to_realign = align_long_text(
            rs, 
            num_segments_per_chunk=5,
            neighbor_threshold=5,
            device=device,
        )

        # Step (6): do standard forced alignment (without factor transducer) on the unaligned parts
        if len(to_realign) > 0:
            logging.info(f"Second-pass: [{batch_idx}/{len(dl)}] len(to_realign) = {len(to_realign)} min:{min(map(lambda x: x[1]-x[0], to_realign))} max:{max(map(lambda x: x[1]-x[0], to_realign))}")
            failed_groups = realign(params, model, sp, alignment_results, to_realign, text, features, frame_rate, word_start_symbols)
            handle_failed_groups(failed_groups, alignment_results)

        # Step (7): save the results
        save_rs = {
            "meta_data": meta_data,
            "alignment_results": alignment_results,
        }
        Path(pt_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_rs, pt_path)

        if False:
            audacity_labels_str = to_audacity_label_format(params, frame_rate, alignment_results, text)
            with open(str(pt_path)[:-3] + "_audacity.txt", "w") as fout:
                print(audacity_labels_str, file=fout)

        logging.info(f"Saved: {pt_path}")

        # if batch_idx % 5 == 0:
        #     batch_str = f"{batch_idx}/{len(dl)}"
        #     logging.info(f"batch {batch_str}")
        
        # logging.info(f"batch_idx={batch_idx}")
        # if batch_idx > 200:
        #     break
    return



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
        device = torch.device("cuda", rank)
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

    ########################################################
    # Make dataset and dataloader
    ########################################################

    long_dataset = LibrispeechLongAudioDataset(
        root = "/exp/rhuang/meta/icefall/egs/librispeech/ASR/download/",
        # skip_loading_audio = False,
        # skip_text_normalization = True,
        # manifest_file = None,
    )
    # waveform, sample_rate, text, speaker_id, audio_id, meta_data = \
    #     long_dataset[0]    
    # print(f"[{audio_id}] len(text) = {len(text)}")

    if params.part is not None:
        params.part = params.part.split("/")
        params.part = (int(params.part[0]), int(params.part[1]))
        long_dataset.filter(lambda audio_path, text_path: int(audio_path.split("/")[-2]) % params.part[1] == params.part[0] % params.part[1])
    
    def filter_done(audio_path, text_path):
        pt_path = audio_path.replace("LibriSpeechOriginal/LibriSpeech/", "").replace("mp3/", f"ali_{params.return_penalty}_segment{params.segment_size}_overlap{params.overlap}/")
        pt_path = f"{params.exp_dir}/{pt_path}"
        pt_path = Path(pt_path).parent / (Path(pt_path).parent.stem + ".pt")
        return not pt_path.exists()
    long_dataset.filter(filter_done)

    # long_dataset.filter(lambda audio_path, text_path: int(audio_path.split("/")[-2]) == 295948)

    if len(long_dataset) == 0:
        logging.info("No audio to process.")
        return
    
    if world_size > 1:
        sampler = DistributedSampler(long_dataset)
    else:
        sampler = None

    # We use a dataloader of batch_size=1 here, instead of just looping over the dataset,
    # because we want to use the DataLoader's multi-threading and prefetching capabilities.
    # It turns out that the DataLoader's prefetching is effective for loading long audios.
    dataloader = DataLoader(
        long_dataset, 
        batch_size=1,
        shuffle=True,
        num_workers=4,
        sampler=sampler,
        prefetch_factor=3,
    )

    test_sets = ["libri_long"]
    test_dls = [dataloader]

    ########################################################
    # Start alignment
    ########################################################

    for test_set, test_dl in zip(test_sets, test_dls):
        align_dataset(
            dl=test_dl,
            params=params,
            model=model,
            sp=sp,
            rank=rank,
        )

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
