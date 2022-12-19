#!/usr/bin/env python3
# Copyright    2022  Xiaomi Corp.        (authors: Fangjun Kuang)
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

This script takes as input (1) `lang_dir`, which should contain::

    - lang_dir/bpe.model,

as well as (2) `biasing_list.txt` which contains one biasing phrase
on each line.
It generates the following files in the directory `lang_dir`:

    - context_graph.fst.bin

"""
import argparse
import glob
import logging
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import k2
import sentencepiece as spm
import torch
from kaldifst.utils import k2_to_openfst
from prepare_lang import Lexicon


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--biasing-list",
        type=str,
        help="""The file that contains a list of biasing phrases.
        Each line of the file contains a pair of:
          - the phrase
          - the relative weight for biasing this phrase, compared
            to others phrases
        They are seperated by a space.
        """,
    )

    parser.add_argument(
        "--bpe-model-file",
        type=str,
        help="""
        Path to a bpe or sentencepiece model.
        """,
    )

    parser.add_argument(
        "--backoff-id",
        type=int,
        help="""
        The id of the backoff token. This token serves for the
        "failure transition" in an WFST ( --
        `Failure transitions for Joint n-gram Models and G2P Conversion`).
        The idea is that the arc with the backoff token should be
        traversed only in the event that no valid normal transition
        exists leaving the current state.

        This id is usually taken from an un-used integer in token2id.
        """,
    )

    parser.add_argument(
        "--context-dir",
        type=str,
        help="""
        Path to the `lang/context` directory.
        """,
    )

    parser.add_argument(
        "--context-id",
        type=str,
        default="",
        help="""
        The id of the context. It is usually taken from the id of the utterance.
        """,
    )

    parser.add_argument(
        "--nbest-size",
        type=int,
        default=1,
        help="""
        The maximum number of different tokenization for each lexicon entry.
        """,
    )

    return parser.parse_args()


def read_biasing_list(filename: str) -> Dict[str, float]:
    """.

    Args:
      a:
        .
      b:
        .
    Returns:
      Return x.
    """
    biasing_list = {}

    with open(filename, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip(" \t\r\n")
            if len(line) == 0:
                continue
            line = line.rsplit(maxsplit=1)
            phrase = line[0]
            weight = float(line[-1])
            biasing_list[phrase] = weight
    logging.info(f"len(biasing_list) = {len(biasing_list)}")

    return biasing_list


def generate_lexicon_for_biasing_list(
    model_file: str, biasing_list: Dict[str, float], nbest_size: int
) -> Tuple[Lexicon, Dict[str, int]]:
    """Generate a lexicon for the biasing list from a BPE model.

    Args:
      model_file:
        Path to a sentencepiece model.
      biasing_list:
        A dictionary of {biasing word: weight}.
      nbest_size:
        The maximum number of alternative tokenizations to put in
        the lexicon for each word.
    Returns:
      Return a tuple with two elements:
        - A lexicon as a list of tuples: (word, token sequence).
        - A dict representing the token symbol, mapping from tokens to IDs.
          This mapping is derived from the sentencepiece model.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_file))

    biasing_phrases = list(biasing_list.keys())

    # Convert word to word piece IDs instead of word piece strings
    # to avoid OOV tokens.
    words_pieces_ids: List[List[List[int]]] = [
        sp.nbest_encode_as_ids(w, nbest_size=nbest_size) for w in biasing_phrases
    ]

    # Now convert word piece IDs back to word piece strings.
    words_pieces: List[List[List[str]]] = [
        [sp.id_to_piece(ids) for ids in ids_list] for ids_list in words_pieces_ids
    ]

    lexicon = []
    for phrase, alternatives in zip(biasing_phrases, words_pieces):
        for pieces in alternatives:
            lexicon.append((phrase, pieces))

    token2id: Dict[str, int] = dict()
    for i in range(sp.vocab_size()):
        token2id[sp.id_to_piece(i)] = i

    return lexicon, token2id


def generate_context_graph_simple(
    lexicon: Lexicon,
    biasing_list: Dict[str, float],
    token2id: Dict[str, int],
    backoff_id: int,
    bonus_per_token: float = 0.1,
) -> k2.Fsa:
    """Generate the context graph (in kaldifst format) given
    the lexicon of the biasing list.

    This context graph is a WFST as in
    `https://arxiv.org/abs/1808.02480`
    or
    `https://wenet.org.cn/wenet/context.html`.
    It is simple, as it does not have the capability to detect
    word boundaries. So, if a biasing word (e.g., 'us', the country)
    happens to be the prefix of another word (e.g., 'useful'),
    it will still be detected. This is not desired.
    However, this context graph is easy to understand.

    Args:
      lexicon:
        The input lexicon for the biasing list.
      biasing_list:
        A dictionary of {biasing word: weight}.
      token2id:
        A dict mapping tokens to IDs.
      backoff_id:
        The id of the backoff token. It serves for failure arcs.
      bonus_per_token:
        The bonus for each token during decoding, which will hopefully
        boost the token up to survive beam search.

    Returns:
      Return an instance of `k2.Fsa` representing the context graph.
    """

    start_state = 0
    next_state = 1  # the next un-allocated state, will be incremented as we go.
    arcs = []

    flip = (
        -1
    )  # note: `k2_to_openfst` will multiply it with -1. So it will become +1 in the end.

    arcs.append([start_state, start_state, backoff_id, 0, 0.0])

    for word, tokens in lexicon:
        assert len(tokens) > 0, f"{word} has no pronunciations"
        cur_state = start_state

        tokens = [token2id[i] for i in tokens]

        for i in range(len(tokens) - 1):
            arcs.append([cur_state, next_state, tokens[i], 0, flip * bonus_per_token])
            arcs.append(
                [
                    next_state,
                    start_state,
                    backoff_id,
                    0,
                    flip * -bonus_per_token * (i + 1),
                ]
            )

            cur_state = next_state
            next_state += 1

        # now for the last token of this word
        i = len(tokens) - 1
        arcs.append([cur_state, start_state, tokens[i], 0, flip * bonus_per_token])

    final_state = next_state
    arcs.append([start_state, final_state, -1, -1, 0])
    arcs.append([final_state])

    arcs = sorted(arcs, key=lambda arc: arc[0])
    arcs = [[str(i) for i in arc] for arc in arcs]
    arcs = [" ".join(arc) for arc in arcs]
    arcs = "\n".join(arcs)

    fsa = k2.Fsa.from_str(arcs, acceptor=False)
    fsa = k2.arc_sort(fsa)
    return fsa


def generate_context_graph(
    lexicon: Lexicon,
    biasing_list: Dict[str, float],
    token2id: Dict[str, int],
    backoff_id: int,
    bonus_per_token: float = 0.1,
) -> k2.Fsa:
    """Generate the context graph (in kaldifst format) given
    the lexicon of the biasing list.

    Args:
      lexicon:
        The input lexicon for the biasing list.
      biasing_list:
        A dictionary of {biasing word: weight}.
      token2id:
        A dict mapping tokens to IDs.
      backoff_id:
        The id of the backoff token. It serves for failure arcs.
      bonus_per_token:
        The bonus for each token during decoding, which will hopefully
        boost the token up to survive beam search.

    Returns:
      Return an instance of `k2.Fsa` representing the context graph.
    """

    start_state = 0
    pending_state = 1
    next_state = 2  # the next un-allocated state, will be incremented as we go.
    arcs = []

    # note: `k2_to_openfst` will multiply it with -1. So it will become +1 in the end.
    flip = -1

    arcs.append([start_state, start_state, backoff_id, 0, 0.0])

    for word, tokens in lexicon:
        assert len(tokens) > 0, f"{word} has no pronunciations"
        cur_state = start_state

        tokens = [token2id[i] for i in tokens]

        bonus_per_token = 1.0 / len(tokens)

        for i in range(len(tokens) - 1):
            arcs.append([cur_state, next_state, tokens[i], 0, flip * bonus_per_token])
            arcs.append(
                [
                    next_state,
                    start_state,
                    backoff_id,
                    0,
                    flip * -bonus_per_token * (i + 1),
                ]
            )

            cur_state = next_state
            next_state += 1

        # now for the last token of this word
        i = len(tokens) - 1
        arcs.append([cur_state, pending_state, tokens[i], 0, flip * bonus_per_token])

    for token, token_id in token2id.items():
        if token.startswith("▁"):
            arcs.append([pending_state, start_state, token_id, 0, 0.0])
    arcs.append([pending_state, start_state, backoff_id, 0, flip * -1.0])

    final_state = next_state
    arcs.append([start_state, final_state, -1, -1, 0])
    arcs.append([final_state])

    arcs = sorted(arcs, key=lambda arc: arc[0])
    arcs = [[str(i) for i in arc] for arc in arcs]
    arcs = [" ".join(arc) for arc in arcs]
    arcs = "\n".join(arcs)

    fsa = k2.Fsa.from_str(arcs, acceptor=False)
    fsa = k2.arc_sort(fsa)
    return fsa


def generate_context_graph_nfa(
    lexicon: Lexicon,
    biasing_list: Dict[str, float],
    token2id: Dict[str, int],
    backoff_id: int,
    bonus_per_token: float = 0.1,
) -> k2.Fsa:
    """Generate the context graph (in kaldifst format) given
    the lexicon of the biasing list.

    This context graph is a WFST capable of detecting word boundaries.
    It is epsilon-free and non-deterministic.

    Args:
      lexicon:
        The input lexicon for the biasing list.
      biasing_list:
        A dictionary of {biasing word: weight}.
      token2id:
        A dict mapping tokens to IDs.
      backoff_id:
        The id of the backoff token. It serves for failure arcs.
      bonus_per_token:
        The bonus for each token during decoding, which will hopefully
        boost the token up to survive beam search.

    Returns:
      Return an instance of `k2.Fsa` representing the context graph.
    """

    start_state = 0
    # if the path go through this state, then a word boundary is detected
    boundary_state = 1
    # if the path go through this state, then it is not a word boundary
    non_boundary_state = 2
    next_state = 3  # the next un-allocated state, will be incremented as we go.
    arcs = []

    # note: `k2_to_openfst` will multiply it with -1. So it will become +1 in the end.
    flip = -1

    for token, token_id in token2id.items():
        arcs.append([start_state, start_state, token_id, 0, 0.0])

    for word, tokens in lexicon:
        assert len(tokens) > 0, f"{word} has no pronunciations"
        cur_state = start_state

        tokens = [token2id[i] for i in tokens]

        my_bonus_per_token = flip * bonus_per_token * biasing_list[word]

        for i in range(len(tokens) - 1):
            arcs.append([cur_state, next_state, tokens[i], 0, my_bonus_per_token])
            arcs.append(
                [next_state, start_state, backoff_id, 0, -my_bonus_per_token * (i + 1)]
            )
            if i == 0:
                arcs.append(
                    [boundary_state, next_state, tokens[i], 0, my_bonus_per_token]
                )

            cur_state = next_state
            next_state += 1

        # now for the last token of this word
        i = len(tokens) - 1
        arcs.append([cur_state, boundary_state, tokens[i], 0, my_bonus_per_token])
        arcs.append(
            [cur_state, non_boundary_state, tokens[i], 0, -my_bonus_per_token * i]
        )

    for token, token_id in token2id.items():
        if token.startswith("▁"):
            arcs.append([boundary_state, start_state, token_id, 0, 0.0])
        else:
            arcs.append([non_boundary_state, start_state, token_id, 0, 0.0])

    final_state = next_state
    arcs.append([start_state, final_state, -1, -1, 0])
    arcs.append([boundary_state, final_state, -1, -1, 0])
    arcs.append([final_state])

    arcs = sorted(arcs, key=lambda arc: arc[0])
    arcs = [[str(i) for i in arc] for arc in arcs]
    arcs = [" ".join(arc) for arc in arcs]
    arcs = "\n".join(arcs)

    fsa = k2.Fsa.from_str(arcs, acceptor=False)
    fsa = k2.arc_sort(fsa)
    # fsa = k2.determinize(fsa)  # No weight pushing is needed.
    return fsa


def main():
    args = get_args()

    files = glob.glob(f"{args.context_dir}/*.txt")
    for filename in files:
        context_id = os.path.basename(os.path.normpath(filename))
        context_id = context_id[:-4]

        biasing_list = read_biasing_list(f"{args.context_dir}/{context_id}.txt")

        lexicon, token2id = generate_lexicon_for_biasing_list(
            args.bpe_model_file,
            biasing_list,
            nbest_size=args.nbest_size,
        )

        bonus_per_token = 0.1
        context_graph = generate_context_graph_nfa(
            lexicon,
            biasing_list,
            token2id,
            args.backoff_id,
            bonus_per_token,
        )
        logging.info(
            f"{context_id} context graph shape: {context_graph.shape}, num_arcs: {context_graph.num_arcs}"
        )

        # context_graph.draw(Path(args.output_dir) / f"{args.context_id}.svg", title=f"context_graph_{args.context_id}")

        fst = k2_to_openfst(context_graph, olabels="aux_labels")
        output_file_name = Path(args.context_dir) / f"{args.context_id}.fst.bin"
        fst.write(str(output_file_name))
        logging.info(f"{context_id} context graph saved to: {output_file_name}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
