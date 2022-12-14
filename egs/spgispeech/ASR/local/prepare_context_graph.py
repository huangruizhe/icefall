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
import logging
import math
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
        Each line of the file contains one phrase.
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
        "--wildcard-id",
        type=int,
        help="""
        The id of the wildcard token. It can match with any other tokens.
        This id is usually taken from an un-used integer in token2id.
        """,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="""
        Path to the output directory.
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

    return parser.parse_args()


def read_biasing_list(filename: str) -> List:
    """.

    Args:
      a:
        .
      b:
        .
    Returns:
      Return x.
    """
    biasing_list = []

    with open(filename, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip(" \t\r\n")
            if len(line) == 0:
                continue
            biasing_list.append(line)
    logging.info(f"len(biasing_list) = {len(biasing_list)}")

    return biasing_list


def generate_lexicon_for_biasing_list(
    model_file: str, biasing_list: List[str], nbest_size: int
) -> Tuple[Lexicon, Dict[str, int]]:
    """Generate a lexicon for the biasing list from a BPE model.

    Args:
      model_file:
        Path to a sentencepiece model.
      biasing_list:
        A list of strings representing biasing words.
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

    # Convert word to word piece IDs instead of word piece strings
    # to avoid OOV tokens.
    words_pieces_ids: List[List[List[int]]] = [
        sp.nbest_encode_as_ids(w, nbest_size=nbest_size) for w in biasing_list
    ]

    # Now convert word piece IDs back to word piece strings.
    words_pieces: List[List[List[str]]] = [
        [sp.id_to_piece(ids) for ids in ids_list] for ids_list in words_pieces_ids
    ]

    lexicon = []
    for word, alternatives in zip(biasing_list, words_pieces):
        for pieces in alternatives:
            lexicon.append((word, pieces))

    token2id: Dict[str, int] = dict()
    for i in range(sp.vocab_size()):
        token2id[sp.id_to_piece(i)] = i

    return lexicon, token2id


def generate_context_graph_simple(
    lexicon: Lexicon,
    token2id: Dict[str, int],
    wildcard_id: int,
    bonus_per_token: float = 0.1,
) -> k2.Fsa:
    """Generate the context graph (in kaldifst format) given
    the lexicon of the biasing list.

    Args:
      lexicon:
        The input lexicon for the biasing list.
      token2id:
        A dict mapping tokens to IDs.
      wildcard_id:
        The id of the wildcard token. It can match with any other tokens.
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

    arcs.append([start_state, start_state, wildcard_id, 0, 0.0])

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
                    wildcard_id,
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
    token2id: Dict[str, int],
    wildcard_id: int,
    bonus_per_token: float = 0.1,
) -> k2.Fsa:
    """Generate the context graph (in kaldifst format) given
    the lexicon of the biasing list.

    Args:
      lexicon:
        The input lexicon for the biasing list.
      token2id:
        A dict mapping tokens to IDs.
      wildcard_id:
        The id of the wildcard token. It can match with any other tokens.
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

    flip = (
        -1
    )  # note: `k2_to_openfst` will multiply it with -1. So it will become +1 in the end.

    arcs.append([start_state, start_state, wildcard_id, 0, 0.0])

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
                    wildcard_id,
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
    arcs.append([pending_state, start_state, wildcard_id, 0, flip * -1.0])

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


def main():
    args = get_args()

    biasing_list = read_biasing_list(args.biasing_list)

    lexicon, token2id = generate_lexicon_for_biasing_list(
        args.bpe_model_file,
        biasing_list,
        nbest_size=1,
    )

    bonus_per_token = 0.1
    context_graph = generate_context_graph(
        lexicon,
        token2id,
        args.wildcard_id,
        bonus_per_token,
    )
    logging.info(
        f"Context graph shape: {context_graph.shape}, num_arcs: {context_graph.num_arcs}"
    )

    # context_graph.draw(Path(args.output_dir) / f"{args.context_id}.svg", title=f"context_graph_{args.context_id}")

    output_file_name = Path(args.output_dir) / f"{args.context_id}.fst.bin"
    fst = k2_to_openfst(context_graph, olabels="aux_labels")
    fst.write(str(output_file_name))
    logging.info(f"Context graph saved to: {output_file_name}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
