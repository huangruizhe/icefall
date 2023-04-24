#!/usr/bin/env python3
#
# Copyright 2021 Xiaomi Corporation (Author: Fangjun Kuang)
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
Usage:
(1) greedy search
./pruned_transducer_stateless2/decode.py \
        --iter 696000 \
        --avg 10 \
        --exp-dir ./pruned_transducer_stateless2/exp \
        --max-duration 100 \
        --decoding-method greedy_search

(2) beam search
./pruned_transducer_stateless2/decode.py \
        --iter 696000 \
        --avg 10 \
        --exp-dir ./pruned_transducer_stateless2/exp \
        --max-duration 100 \
        --decoding-method beam_search \
        --beam-size 4

(3) modified beam search
./pruned_transducer_stateless2/decode.py \
        --iter 696000 \
        --avg 10 \
        --exp-dir ./pruned_transducer_stateless2/exp \
        --max-duration 100 \
        --decoding-method modified_beam_search \
        --beam-size 4

(4) fast beam search
./pruned_transducer_stateless2/decode.py \
        --iter 696000 \
        --avg 10 \
        --exp-dir ./pruned_transducer_stateless2/exp \
        --max-duration 1500 \
        --decoding-method fast_beam_search \
        --beam 4 \
        --max-contexts 4 \
        --max-states 8
"""


import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import sentencepiece as spm
import torch
import torch.nn as nn
from asr_datamodule import SPGISpeechAsrDataModule
from beam_search import (
    beam_search,
    fast_beam_search_one_best,
    greedy_search,
    greedy_search_batch,
    modified_beam_search,
    modified_beam_search_LODR,
)
from train import get_params, get_transducer_model
from egs.spgispeech.ASR.pruned_transducer_stateless2_context.context_collector import ContextCollector
from egs.spgispeech.ASR.pruned_transducer_stateless2_context.context_encoder import ContextEncoder
from egs.spgispeech.ASR.pruned_transducer_stateless2_context.context_encoder_lstm import ContextEncoderLSTM
from egs.spgispeech.ASR.pruned_transducer_stateless2_context.context_encoder_pretrained import ContextEncoderPretrained
from egs.spgispeech.ASR.pruned_transducer_stateless2_context.word_encoder_bert import BertEncoder

from icefall import LmScorer, NgramLm, BiasedNgramLm
from icefall.checkpoint import average_checkpoints, find_checkpoints, load_checkpoint
from icefall.utils import (
    AttributeDict,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
)
from icefall.lexicon import Lexicon
from lhotse import CutSet


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=20,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 0.
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
        default=10,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="pruned_transducer_stateless2/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--lang-dir",
        type=Path,
        default="data/lang_bpe_500",
        help="The lang dir containing word table and LG graph",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Possible values are:
          - greedy_search
          - beam_search
          - modified_beam_search
          - fast_beam_search
          - modified_beam_search_LODR
        """,
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="""An interger indicating how many candidates we will keep for each
        frame. Used only when --decoding-method is beam_search or
        modified_beam_search.""",
    )

    parser.add_argument(
        "--beam",
        type=float,
        default=4,
        help="""A floating point value to calculate the cutoff score during beam
        search (i.e., `cutoff = max-score - beam`), which is the same as the
        `beam` in Kaldi.
        Used only when --decoding-method is fast_beam_search""",
    )

    parser.add_argument(
        "--ngram-lm-scale",
        type=float,
        default=0.01,
        help="""
        Used only when --decoding_method is fast_beam_search_nbest_LG.
        It specifies the scale for n-gram LM scores.
        """,
    )

    parser.add_argument(
        "--max-contexts",
        type=int,
        default=4,
        help="""Used only when --decoding-method is
        fast_beam_search""",
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=8,
        help="""Used only when --decoding-method is
        fast_beam_search""",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )
    parser.add_argument(
        "--max-sym-per-frame",
        type=int,
        default=1,
        help="""Maximum number of symbols per frame.
        Used only when --decoding_method is greedy_search""",
    )

    parser.add_argument(
        "--num-paths",
        type=int,
        default=200,
        help="""Number of paths for nbest decoding.
        Used only when the decoding method is fast_beam_search_nbest,
        fast_beam_search_nbest_LG, and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--nbest-scale",
        type=float,
        default=0.5,
        help="""Scale applied to lattice scores when computing nbest paths.
        Used only when the decoding method is fast_beam_search_nbest,
        fast_beam_search_nbest_LG, and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--simulate-streaming",
        type=str2bool,
        default=False,
        help="""Whether to simulate streaming in decoding, this is a good way to
        test a streaming model.
        """,
    )

    parser.add_argument(
        "--decode-chunk-size",
        type=int,
        default=16,
        help="The chunk size for decoding (in frames after subsampling)",
    )

    parser.add_argument(
        "--left-context",
        type=int,
        default=64,
        help="left context can be seen during decoding (in frames after subsampling)",
    )

    parser.add_argument(
        "--use-shallow-fusion",
        type=str2bool,
        default=False,
        help="""Use neural network LM for shallow fusion.
        If you want to use LODR, you will also need to set this to true
        """,
    )

    parser.add_argument(
        "--lm-type",
        type=str,
        default="rnn",
        help="Type of NN lm",
        choices=["rnn", "transformer"],
    )

    parser.add_argument(
        "--lm-scale",
        type=float,
        default=0.3,
        help="""The scale of the neural network LM
        Used only when `--use-shallow-fusion` is set to True.
        """,
    )

    parser.add_argument(
        "--tokens-ngram",
        type=int,
        default=3,
        help="""Token Ngram used for rescoring.
            Used only when the decoding method is
            modified_beam_search_ngram_rescoring, or LODR
            """,
    )

    parser.add_argument(
        "--backoff-id",
        type=int,
        default=500,
        help="""ID of the backoff symbol.
                Used only when the decoding method is
                modified_beam_search_ngram_rescoring""",
    )

    parser.add_argument(
        "--context-dir",
        type=str,
        default="data/rare_words/",
        help="",
    )

    parser.add_argument(
        "--slides",
        type=str,
        default="/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_heuristics2/context/",
        help="",
    )

    parser.add_argument(
        "--n-distractors",
        type=int,
        default=100,
        help="",
    )

    parser.add_argument(
        "--keep-ratio",
        type=float,
        default=1.0,
        help="",
    )

    parser.add_argument(
        "--no-encoder-biasing",
        type=str2bool,
        default=False,
        help=""".
        """,
    )

    parser.add_argument(
        "--no-decoder-biasing",
        type=str2bool,
        default=False,
        help=""".
        """,
    )

    parser.add_argument(
        "--no-wfst-lm-biasing",
        type=str2bool,
        default=True,
        help=""".
        """,
    )

    parser.add_argument(
        "--is-full-context",
        type=str2bool,
        default=False,
        help="",
    )

    parser.add_argument(
        "--is-predefined",
        type=str2bool,
        default=False,
        help="",
    )

    parser.add_argument(
        "--is-pretrained-context-encoder",
        type=str2bool,
        default=False,
        help="",
    )

    parser.add_argument(
        "--biased-lm-scale",
        type=float,
        default=0.0,
        help="",
    )

    return parser


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    context_collector: ContextCollector,
    sp: spm.SentencePieceProcessor,
    batch: dict,
    word_table: Optional[k2.SymbolTable] = None,
    decoding_graph: Optional[k2.Fsa] = None,
    ngram_lm: Optional[NgramLm] = None,
    ngram_lm_scale: float = 1.0,
    LM: Optional[LmScorer] = None,
) -> Dict[str, List[List[str]]]:
    """Decode one batch and return the result in a dict. The dict has the
    following format:

        - key: It indicates the setting used for decoding. For example,
               if greedy_search is used, it would be "greedy_search"
               If beam search with a beam size of 7 is used, it would be
               "beam_7"
        - value: It contains the decoding result. `len(value)` equals to
                 batch size. `value[i]` is the decoding result for the i-th
                 utterance in the given batch.
    Args:
      params:
        It's the return value of :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or HLG, Used
        only when --decoding_method is fast_beam_search.
    Returns:
      Return the decoding result. See above description for the format of
      the returned dict.
    """
    device = model.device
    feature = batch["inputs"]
    assert feature.ndim == 3

    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    encoder_out, encoder_out_lens = model.encoder(x=feature, x_lens=feature_lens)
    
    model.scratch_space = dict()
    model.scratch_space["sp"] = sp
    model.scratch_space["biased_lm_scale"] = params.biased_lm_scale

    if not params.no_wfst_lm_biasing:
        fsa_list, fsa_sizes, num_words_per_utt2 = \
            context_collector.get_context_word_wfst(batch)
        biased_lm_list = [
            BiasedNgramLm(
                fst=fsa, 
                backoff_id=context_collector.backoff_id
            ) for fsa in fsa_list
        ]
        model.scratch_space["biased_lm_list"] = biased_lm_list

    if not model.no_encoder_biasing:
        word_list, word_lengths, num_words_per_utt = \
            context_collector.get_context_word_list(batch)
        word_list = word_list.to(device)
        contexts, contexts_mask = model.context_encoder.embed_contexts(
            word_list,
            word_lengths,
            num_words_per_utt,
        )
        model.scratch_space["contexts"] = contexts
        model.scratch_space["contexts_mask"] = contexts_mask

        encoder_biasing_out, attn = model.encoder_biasing_adapter.forward(encoder_out, contexts, contexts_mask)
        encoder_out = encoder_out + encoder_biasing_out
    
    hyps = []

    if params.decoding_method == "fast_beam_search":
        hyp_tokens = fast_beam_search_one_best(
            model=model,
            decoding_graph=decoding_graph,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam,
            max_contexts=params.max_contexts,
            max_states=params.max_states,
        )
        for hyp in sp.decode(hyp_tokens):
            hyps.append(hyp.split())
    elif params.decoding_method == "greedy_search" and params.max_sym_per_frame == 1:
        hyp_tokens = greedy_search_batch(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
        )
        for hyp in sp.decode(hyp_tokens):
            hyps.append(hyp.split())
    elif params.decoding_method == "modified_beam_search":
        hyp_tokens = modified_beam_search(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam_size,
        )
        for hyp in sp.decode(hyp_tokens):
            hyps.append(hyp.split())
    elif params.decoding_method == "modified_beam_search_LODR":
        hyp_tokens = modified_beam_search_LODR(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam_size,
            LODR_lm=ngram_lm,
            LODR_lm_scale=ngram_lm_scale,
            LM=LM,
        )
        for hyp in sp.decode(hyp_tokens):
            hyps.append(hyp.split())
    else:
        batch_size = encoder_out.size(0)

        for i in range(batch_size):
            # fmt: off
            encoder_out_i = encoder_out[i:i+1, :encoder_out_lens[i]]
            # fmt: on
            if params.decoding_method == "greedy_search":
                hyp = greedy_search(
                    model=model,
                    encoder_out=encoder_out_i,
                    max_sym_per_frame=params.max_sym_per_frame,
                )
            elif params.decoding_method == "beam_search":
                hyp = beam_search(
                    model=model,
                    encoder_out=encoder_out_i,
                    beam=params.beam_size,
                )
            else:
                raise ValueError(
                    f"Unsupported decoding method: {params.decoding_method}"
                )
            hyps.append(sp.decode(hyp).split())

    if params.decoding_method == "greedy_search":
        return {"greedy_search": hyps}
    elif params.decoding_method == "fast_beam_search":
        return {
            (
                f"beam_{params.beam}_"
                f"max_contexts_{params.max_contexts}_"
                f"max_states_{params.max_states}"
            ): hyps
        }
    else:
        return {f"beam_size_{params.beam_size}": hyps}


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    context_collector: ContextCollector,
    sp: spm.SentencePieceProcessor,
    word_table: Optional[k2.SymbolTable] = None,
    decoding_graph: Optional[k2.Fsa] = None,
    ngram_lm: Optional[NgramLm] = None,
    ngram_lm_scale: float = 1.0,
    LM: Optional[LmScorer] = None,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or HLG, Used
        only when --decoding_method is fast_beam_search.
    Returns:
      Return a dict, whose key may be "greedy_search" if greedy search
      is used, or it may be "beam_7" if beam size of 7 is used.
      Its value is a list of tuples. Each tuple contains two elements:
      The first is the reference transcript, and the second is the
      predicted result.
    """
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    if params.decoding_method == "greedy_search":
        log_interval = 100
    else:
        log_interval = 20

    results = defaultdict(list)
    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

        hyps_dict = decode_one_batch(
            params=params,
            model=model,
            context_collector=context_collector,
            sp=sp,
            decoding_graph=decoding_graph,
            word_table=word_table,
            batch=batch,
            ngram_lm=ngram_lm,
            ngram_lm_scale=ngram_lm_scale,
            LM=LM,
        )

        for name, hyps in hyps_dict.items():
            this_batch = []
            assert len(hyps) == len(texts)
            for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts):
                ref_words = ref_text.split()
                this_batch.append((cut_id, ref_words, hyp_words))

            results[name].extend(this_batch)

        num_cuts += len(texts)

        if batch_idx % log_interval == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    return results


def save_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
):
    test_set_wers = dict()
    test_set_cers = dict()
    for key, results in results_dict.items():
        recog_path = params.res_dir / f"recogs-{test_set_name}-{params.suffix}.txt"
        results = sorted(results)
        store_transcripts(filename=recog_path, texts=results)
        logging.info(f"The transcripts are stored in {recog_path}")

        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        wers_filename = params.res_dir / f"wers-{test_set_name}-{params.suffix}.txt"
        with open(wers_filename, "w") as f:
            wer = write_error_stats(
                f, f"{test_set_name}-{key}", results, enable_log=True
            )
            test_set_wers[key] = wer

        # we also compute CER for spgispeech dataset.
        results_char = []
        for res in results:
            results_char.append((res[0], list("".join(res[1])), list("".join(res[2]))))
        cers_filename = params.res_dir / f"cers-{test_set_name}-{params.suffix}.txt"
        with open(cers_filename, "w") as f:
            cer = write_error_stats(
                f, f"{test_set_name}-{key}", results_char, enable_log=True
            )
            test_set_cers[key] = cer

        logging.info("Wrote detailed error stats to {}".format(wers_filename))

    test_set_wers = {k: v for k, v in sorted(test_set_wers.items(), key=lambda x: x[1])}
    test_set_cers = {k: v for k, v in sorted(test_set_cers.items(), key=lambda x: x[1])}
    errs_info = params.res_dir / f"wer-summary-{test_set_name}-{params.suffix}.txt"
    with open(errs_info, "w") as f:
        print("settings\tWER\tCER", file=f)
        for key in test_set_wers:
            print(
                "{}\t{}\t{}".format(key, test_set_wers[key], test_set_cers[key]),
                file=f,
            )

    s = "\nFor {}, WER/CER of different settings are:\n".format(test_set_name)
    note = "\tbest for {}".format(test_set_name)
    for key in test_set_wers:
        s += "{}\t{}\t{}{}\n".format(key, test_set_wers[key], test_set_cers[key], note)
        note = ""
    logging.info(s)

def rare_word_score(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
    cuts,
):
    from collections import namedtuple
    from score import main as score_main
    from lhotse import CutSet

    logging.info(f"test_set_name: {test_set_name}")
    cuts = cuts[0]
    cuts = [c for c in cuts]
    cuts = CutSet.from_cuts(cuts)

    args = namedtuple('A', ['refs', 'hyps', 'lenient'])
    if params.n_distractors > 0:
        args.refs = params.context_dir / f"ref/{test_set_name}.biasing_{params.n_distractors}.tsv"
    else:
        args.refs = params.context_dir / f"ref/{test_set_name}.biasing_100.tsv"
    args.lenient = True

    for key, results in results_dict.items():
        print()
        logging.info(f"{key}")
        args.hyps = dict()

        for cut_id, ref, hyp in results:
            u_id = cuts[cut_id].supervisions[0].id
            hyp = " ".join(hyp)
            hyp = hyp.lower()
            args.hyps[u_id] = hyp
        
        score_main(args)
        print()

@torch.no_grad()
def main():
    parser = get_parser()
    SPGISpeechAsrDataModule.add_arguments(parser)
    LmScorer.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    assert params.decoding_method in (
        "greedy_search",
        "beam_search",
        "fast_beam_search",
        "modified_beam_search",
        "modified_beam_search_LODR",
    )
    params.res_dir = params.exp_dir / params.decoding_method

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    if "fast_beam_search" in params.decoding_method:
        params.suffix += f"-beam-{params.beam}"
        params.suffix += f"-max-contexts-{params.max_contexts}"
        params.suffix += f"-max-states-{params.max_states}"
    elif "beam_search" in params.decoding_method:
        params.suffix += f"-{params.decoding_method}-beam-size-{params.beam_size}"
    else:
        params.suffix += f"-context-{params.context_size}"
        params.suffix += f"-max-sym-per-frame-{params.max_sym_per_frame}"
    
    if params.use_shallow_fusion:
        if params.lm_type == "rnn":
            params.suffix += f"-rnnlm-lm-scale-{params.lm_scale}"
        elif params.lm_type == "transformer":
            params.suffix += f"-transformer-lm-scale-{params.lm_scale}"

        if "LODR" in params.decoding_method:
            params.suffix += (
                f"-LODR-{params.tokens_ngram}gram-scale-{params.ngram_lm_scale}"
            )
    
    if not params.no_wfst_lm_biasing:
        params.suffix += f"-wfst-biasing-{params.biased_lm_scale}"
    if not params.no_encoder_biasing:
        params.suffix += f"-encoder-biasing"
    if not params.no_decoder_biasing:
        params.suffix += f"-decoder-biasing"
    
    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    params.suffix += f"-{timestr}"

    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("Decoding started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.unk_id = sp.piece_to_id("<unk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(params)

    logging.info("About to load context collector")
    params.context_dir = Path(params.context_dir)
    if params.is_pretrained_context_encoder:
        # Use pretrained encoder, e.g., BERT
        bert_encoder = BertEncoder(device=device)
        context_collector = ContextCollector(
            path_rare_words=params.context_dir,
            slides=params.slides,
            sp=None,
            bert_encoder=bert_encoder,
            is_predefined=params.is_predefined,
            n_distractors=params.n_distractors,
            keep_ratio=params.keep_ratio,
            is_full_context=params.is_full_context,
            backoff_id=params.backoff_id,
        )
        # bert_encoder.free_up()
    else:
        context_collector = ContextCollector(
            path_rare_words=params.context_dir,
            slides=params.slides,
            sp=sp,
            bert_encoder=None,
            is_predefined=params.is_predefined,
            n_distractors=params.n_distractors,
            keep_ratio=params.keep_ratio,
            is_full_context=params.is_full_context,
            backoff_id=params.backoff_id,
        )

    logging.info("About to create model")
    model = get_transducer_model(params)

    model.no_encoder_biasing = params.no_encoder_biasing
    model.no_decoder_biasing = params.no_decoder_biasing
    model.no_wfst_lm_biasing = params.no_wfst_lm_biasing

    if params.iter > 0:
        filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
            : params.avg
        ]
        if len(filenames) == 0:
            raise ValueError(
                f"No checkpoints found for --iter {params.iter}, --avg {params.avg}"
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
            if start >= 0:
                filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
        logging.info(f"averaging {filenames}")
        model.to(device)
        model.load_state_dict(average_checkpoints(filenames, device=device))

    model.to(device)
    model.eval()
    model.device = device

    # only load N-gram LM when needed
    if "ngram" in params.decoding_method or "LODR" in params.decoding_method:
        lm_filename = f"{params.tokens_ngram}gram.fst.txt"
        logging.info(f"lm filename: {lm_filename}")
        ngram_lm = NgramLm(
            str(params.lang_dir / lm_filename),
            backoff_id=params.backoff_id,
            is_binary=False,
        )
        logging.info(f"num states: {ngram_lm.lm.num_states}")
        ngram_lm_scale = params.ngram_lm_scale
    else:
        ngram_lm = None
        ngram_lm_scale = None

    # only load the neural network LM if doing shallow fusion
    if params.use_shallow_fusion:
        LM = LmScorer(
            lm_type=params.lm_type,
            params=params,
            device=device,
            lm_scale=params.lm_scale,
        )
        LM.to(device)
        LM.eval()

    else:
        LM = None

    if params.decoding_method == "fast_beam_search":
        decoding_graph = k2.trivial_graph(params.vocab_size - 1, device=device)
        word_table = None
    else:
        decoding_graph = None
        word_table = None

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
    args.on_the_fly_feats = True
    spgispeech = SPGISpeechAsrDataModule(args)

    dev_cuts = spgispeech.dev_cuts()
    val_cuts = spgispeech.val_cuts()

    ec53_cuts_file = "/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_ec53_norm.jsonl.gz"
    logging.info(f"Loading cuts from: {ec53_cuts_file}")
    ec53_cuts = CutSet.from_file(ec53_cuts_file)
    ec53_cuts.describe()

    # from lhotse.utils import fix_random_seed
    # fix_random_seed(12358)
    # ec53_cuts = ec53_cuts.sample(n_cuts=500)
    # ec53_cuts.describe()

    dev_dl = spgispeech.test_dataloaders(dev_cuts)
    val_dl = spgispeech.test_dataloaders(val_cuts)
    ec53_dl = spgispeech.test_dataloaders(ec53_cuts)

    # test_sets = ["dev", "val"]
    # test_dl = [dev_dl, val_dl]
    test_sets = ["ec53"]
    test_dl = [ec53_dl]

    for test_set, test_dl in zip(test_sets, test_dl):
        results_dict = decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
            context_collector=context_collector,
            sp=sp,
            word_table=word_table,
            decoding_graph=decoding_graph,
            ngram_lm=ngram_lm,
            ngram_lm_scale=ngram_lm_scale,
            LM=LM,
        )

        save_results(
            params=params,
            test_set_name=test_set,
            results_dict=results_dict,
        )

        # rare_word_score(
        #     params=params,
        #     test_set_name=test_set,
        #     results_dict=results_dict,
        #     cuts=test_dl.sampler.cuts,
        # )


    logging.info("Done!")


if __name__ == "__main__":
    main()
