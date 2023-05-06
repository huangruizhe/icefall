# Copyright      2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../LICENSE for clarification regarding multiple authors
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


import math
from typing import List
from pathlib import Path
from collections import defaultdict

import k2
import kaldifst
import torch
from kaldifst.utils import k2_to_openfst

from icefall.lexicon import Lexicon
from icefall.lexicon import UniqLexicon

class HMMTrainingGraphCompiler(object):

    # sil_word = "!SIL"
    # sil_token = "<sil>"

    @staticmethod
    def hmm_topo_one_state(
        max_token: int,
        sil_id: int,
    ) -> k2.Fsa:
        num_tokens = max_token
        # assert (
        #     sil_id <= max_token
        # ), f"sil_id={sil_id} should be less or equal to max_token={max_token}"

        # Plusing 3 here to include the start, loop and final state
        num_states = num_tokens + 3

        # ref: https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/local/prepare_lang.py#L248

        start_state = 0
        loop_state = num_states - 2  # words enter and leave from here
        final_state = num_states - 1
        arcs = []

        eps = 0
        arcs.append([start_state, loop_state, eps, eps, 0])

        for i in range(1, max_token + 1):
            cur_state = i  # state_id = token_id
            arcs.append([loop_state, cur_state, i, i, 0])
            arcs.append([cur_state, cur_state, i, eps, 0])
            arcs.append([cur_state, loop_state, eps, eps, 0])

        arcs.append([loop_state, final_state, -1, -1, 0])
        arcs.append([final_state])

        arcs = sorted(arcs, key=lambda arc: arc[0])
        arcs = [[str(i) for i in arc] for arc in arcs]
        arcs = [" ".join(arc) for arc in arcs]
        arcs = "\n".join(arcs)

        fst = k2.Fsa.from_str(arcs, acceptor=False)
        fst = k2.remove_epsilon(fst)  # Credit: Matthew W
        fst = k2.expand_ragged_attributes(fst)
        return fst

    def hmm_topo_one_state_simplified(
        max_token: int,
        sil_id: int,
    ) -> k2.Fsa:
        num_tokens = max_token
        # assert (
        #     sil_id <= max_token
        # ), f"sil_id={sil_id} should be less or equal to max_token={max_token}"

        num_states = num_tokens + 2

        # ref: https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/local/prepare_lang.py#L248

        start_state = 0
        final_state = num_states - 1
        arcs = []

        eps = 0
        # arcs.append([start_state, start_state, eps, eps, 0])

        for i in range(1, max_token + 1):
            cur_state = i  # state_id = token_id
            arcs.append([start_state, cur_state, i, i, 0])

        for i in range(1, max_token + 1):
            cur_state = i  # state_id = token_id
            arcs.append([cur_state, cur_state, i, eps, 0])
            arcs.append([cur_state, cur_state, i, i, 0])
            for j in range(1, max_token + 1):
                if j == i:
                    continue
                next_state = j
                arcs.append([cur_state, next_state, j, j, 0])

        for i in range(1, max_token + 1):
            cur_state = i  # state_id = token_id
            arcs.append([cur_state, final_state, -1, -1, 0])
        arcs.append([final_state])

        arcs = sorted(arcs, key=lambda arc: arc[0])
        arcs = [[str(i) for i in arc] for arc in arcs]
        arcs = [" ".join(arc) for arc in arcs]
        arcs = "\n".join(arcs)

        fst = k2.Fsa.from_str(arcs, acceptor=False)
        # fst = k2.remove_epsilon(fst)  # Credit: Matthew W
        # fst = k2.expand_ragged_attributes(fst)
        return fst

    def ctc_topo_modified(
        max_token: int,
        sil_id: int,
    ) -> k2.Fsa:
        '''
        This should produce the same topo as `k2.ctc_topo(max_token_id, modified=True)`
        '''
        print("Using my own version of `ctc_topo_modified`")
        num_tokens = max_token
        # assert (
        #     sil_id <= max_token
        # ), f"sil_id={sil_id} should be less or equal to max_token={max_token}"

        num_states = num_tokens + 2

        # ref: https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/local/prepare_lang.py#L248

        start_state = 0
        final_state = num_states - 1
        arcs = []

        blk = 0
        eps = 0
        arcs.append([start_state, start_state, blk, eps, 0])

        for i in range(1, max_token + 1):
            arcs.append([start_state, start_state, i, i, 0])

        for i in range(1, max_token + 1):
            cur_state = i  # state_id = token_id
            arcs.append([start_state, cur_state, i, i, 0])
            arcs.append([cur_state, cur_state, i, eps, 0])
            arcs.append([cur_state, start_state, i, eps, 0])

        arcs.append([start_state, final_state, -1, -1, 0])
        arcs.append([final_state])

        arcs = sorted(arcs, key=lambda arc: arc[0])
        arcs = [[str(i) for i in arc] for arc in arcs]
        arcs = [" ".join(arc) for arc in arcs]
        arcs = "\n".join(arcs)

        fst = k2.Fsa.from_str(arcs, acceptor=False)
        # fst = k2.remove_epsilon(fst)  # Credit: Matthew W
        # fst = k2.expand_ragged_attributes(fst)
        return fst

    def ctc_topo_modified_no_blk(
        max_token: int,
        sil_id: int,
    ) -> k2.Fsa:
        '''
        This should produce the same topo as `k2.ctc_topo(max_token_id, modified=True)`
        '''
        print("Using my own version of `ctc_topo_modified`")
        num_tokens = max_token
        # assert (
        #     sil_id <= max_token
        # ), f"sil_id={sil_id} should be less or equal to max_token={max_token}"

        num_states = num_tokens + 2

        # ref: https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/local/prepare_lang.py#L248

        start_state = 0
        final_state = num_states - 1
        arcs = []

        blk = 0
        eps = 0

        for i in range(1, max_token + 1):
            arcs.append([start_state, start_state, i, i, 0])

        for i in range(1, max_token + 1):
            cur_state = i  # state_id = token_id
            arcs.append([start_state, cur_state, i, i, 0])
            arcs.append([cur_state, cur_state, i, eps, 0])
            arcs.append([cur_state, start_state, i, eps, 0])

        arcs.append([start_state, final_state, -1, -1, 0])
        arcs.append([final_state])

        arcs = sorted(arcs, key=lambda arc: arc[0])
        arcs = [[str(i) for i in arc] for arc in arcs]
        arcs = [" ".join(arc) for arc in arcs]
        arcs = "\n".join(arcs)

        fst = k2.Fsa.from_str(arcs, acceptor=False)
        # fst = k2.remove_epsilon(fst)  # Credit: Matthew W
        # fst = k2.expand_ragged_attributes(fst)
        return fst

    def ctc_topo_modified_debug_for_hmm__debug1__this_can_converge(
        max_token: int,
        sil_id: int,
        lexicon=None,
    ) -> k2.Fsa:
        '''
        This should produce the same topo as `k2.ctc_topo(max_token_id, modified=True)`
        '''
        print("Using `ctc_topo_modified_debug_for_hmm`: debug1")
        num_tokens = max_token
        # assert (
        #     sil_id <= max_token
        # ), f"sil_id={sil_id} should be less or equal to max_token={max_token}"

        num_states = num_tokens + 2

        # ref: https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/local/prepare_lang.py#L248

        start_state = 0
        final_state = num_states - 1
        arcs = []

        blk = 0
        eps = 0
        arcs.append([start_state, start_state, sil_id, eps, 0])

        for i in range(1, max_token + 1):
            arcs.append([start_state, start_state, i, i, 0])

        for i in range(1, max_token + 1):
            cur_state = i  # state_id = token_id
            arcs.append([start_state, cur_state, i, i, 0])
            arcs.append([cur_state, cur_state, i, eps, 0])
            arcs.append([cur_state, start_state, i, eps, 0])

        arcs.append([start_state, final_state, -1, -1, 0])
        arcs.append([final_state])

        arcs = sorted(arcs, key=lambda arc: arc[0])
        arcs = [[str(i) for i in arc] for arc in arcs]
        arcs = [" ".join(arc) for arc in arcs]
        arcs = "\n".join(arcs)

        fst = k2.Fsa.from_str(arcs, acceptor=False)
        # fst = k2.remove_epsilon(fst)  # Credit: Matthew W
        # fst = k2.expand_ragged_attributes(fst)
        return fst

    def ctc_topo_modified_debug_for_hmm__debug2__cannot_converge(
        max_token: int,
        sil_id: int,
        lexicon,
    ) -> k2.Fsa:
        '''
        This should produce the same topo as `k2.ctc_topo(max_token_id, modified=True)`
        '''
        print("Using `ctc_topo_modified_debug_for_hmm`: debug2")
        num_tokens = max_token
        # assert (
        #     sil_id <= max_token
        # ), f"sil_id={sil_id} should be less or equal to max_token={max_token}"

        # ref: https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/local/prepare_lang.py#L248

        start_state = 0
        sil_state = 1
        next_available_state = 2
        arcs = []

        blk = 0
        eps = 0
        # arcs.append([start_state, start_state, sil_id, eps, 0])

        print(f"max_token={max_token}")
        print(f"sil_id={sil_id}")
        print(f"start_tokens={len([i for i in range(1, max_token + 1) if lexicon.token_table.get(i).startswith('▁')])}")

        for i in range(1, max_token + 1):
            arcs.append([start_state, start_state, i, i, 0])

        arcs.append([start_state, sil_state, sil_id, eps, 0])
        arcs.append([sil_state, sil_state, sil_id, eps, 0])
        for i in range(1, max_token + 1):
            token = lexicon.token_table.get(i)
            if token.startswith("▁"):
                arcs.append([sil_state, start_state, i, i, 0])

        for i in range(1, max_token + 1):
            cur_state = next_available_state
            next_available_state += 1

            arcs.append([start_state, cur_state, i, i, 0])
            arcs.append([cur_state, cur_state, i, eps, 0])
            arcs.append([cur_state, start_state, i, eps, 0])

            token = lexicon.token_table.get(i)
            if token.startswith("▁"):
                cur_sil_state = next_available_state
                next_available_state += 1
                arcs.append([start_state, cur_sil_state, sil_id, eps, 0])
                arcs.append([cur_sil_state, cur_sil_state, sil_id, eps, 0])
                arcs.append([cur_sil_state, cur_state, i, i, 0])

        final_state = next_available_state
        arcs.append([start_state, final_state, -1, -1, 0])
        arcs.append([final_state])

        arcs = sorted(arcs, key=lambda arc: arc[0])
        arcs = [[str(i) for i in arc] for arc in arcs]
        arcs = [" ".join(arc) for arc in arcs]
        arcs = "\n".join(arcs)

        fst = k2.Fsa.from_str(arcs, acceptor=False)
        fst = k2.connect(fst)
        # fst = k2.remove_epsilon(fst)  # Credit: Matthew W
        # fst = k2.expand_ragged_attributes(fst)
        return fst
    
    def ctc_topo_modified_debug_for_hmm(
        max_token: int,
        sil_id: int,
        lexicon,
    ) -> k2.Fsa:
        '''
        This should produce the same topo as `k2.ctc_topo(max_token_id, modified=True)`
        '''
        print("Using `ctc_topo_modified_debug_for_hmm`: debug2, removed some non-determinism")
        num_tokens = max_token
        # assert (
        #     sil_id <= max_token
        # ), f"sil_id={sil_id} should be less or equal to max_token={max_token}"

        # ref: https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/local/prepare_lang.py#L248

        start_state = 0
        sil_state = 1
        next_available_state = 2
        arcs = []

        blk = 0
        eps = 0
        # arcs.append([start_state, start_state, sil_id, eps, 0])

        print(f"max_token={max_token}")
        print(f"sil_id={sil_id}")
        print(f"start_tokens={len([i for i in range(1, max_token + 1) if lexicon.token_table.get(i).startswith('▁')])}")

        for i in range(1, max_token + 1):
            arcs.append([start_state, start_state, i, i, 0])

        arcs.append([start_state, sil_state, sil_id, eps, 0])
        arcs.append([sil_state, sil_state, sil_id, eps, 0])
        for i in range(1, max_token + 1):
            token = lexicon.token_table.get(i)
            if token.startswith("▁"):
                arcs.append([sil_state, start_state, i, i, 0])

        for i in range(1, max_token + 1):
            cur_state = next_available_state
            next_available_state += 1

            arcs.append([start_state, cur_state, i, i, 0])
            arcs.append([cur_state, cur_state, i, eps, 0])
            arcs.append([cur_state, start_state, i, eps, 0])

            token = lexicon.token_table.get(i)
            if token.startswith("▁"):
                arcs.append([sil_state, cur_state, i, i, 0])

        final_state = next_available_state
        arcs.append([start_state, final_state, -1, -1, 0])
        arcs.append([final_state])

        arcs = sorted(arcs, key=lambda arc: arc[0])
        arcs = [[str(i) for i in arc] for arc in arcs]
        arcs = [" ".join(arc) for arc in arcs]
        arcs = "\n".join(arcs)

        fst = k2.Fsa.from_str(arcs, acceptor=False)
        fst = k2.connect(fst)
        # fst = k2.remove_epsilon(fst)  # Credit: Matthew W
        # fst = k2.expand_ragged_attributes(fst)
        return fst

    def hmm_topo(
        self,
        max_token: int,
        start_tokens: list,
        sil_id: int = 0,
    ) -> k2.Fsa:
        '''
        HMM topo
        '''
        print("HMM topo")
        num_tokens = max_token
        # assert (
        #     sil_id <= max_token
        # ), f"sil_id={sil_id} should be less or equal to max_token={max_token}"

        start_tokens = set(start_tokens)

        # ref: https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/local/prepare_lang.py#L248

        start_state = 0
        loop_state = 1
        blk_state = 2
        next_available_state = 3
        arcs = []

        blk = sil_id
        arcs.append([start_state, start_state, blk, blk, 0])

        for i in range(1, max_token + 1):
            arcs.append([start_state, loop_state, i, i, 0])

        arcs.append([loop_state, blk_state, blk, blk, 0])
        arcs.append([blk_state, blk_state, blk, blk, 0])

        for i in range(1, max_token + 1):
            cur_state = next_available_state  # state_id
            next_available_state += 1

            arcs.append([loop_state, loop_state, i, i, 0])
            arcs.append([loop_state, cur_state, i, i, 0])
            arcs.append([cur_state, cur_state, i, blk, 0])
            arcs.append([cur_state, loop_state, i, blk, 0])
            
            arcs.append([start_state, cur_state, i, i, 0])

            if i in start_tokens:
                arcs.append([blk_state, loop_state, i, i, 0])
                arcs.append([blk_state, cur_state, i, i, 0])

        final_state = next_available_state
        next_available_state += 1
        arcs.append([start_state, final_state, -1, -1, 0])
        arcs.append([loop_state, final_state, -1, -1, 0])
        arcs.append([blk_state, final_state, -1, -1, 0])    
        arcs.append([final_state])

        arcs = sorted(arcs, key=lambda arc: arc[0])
        arcs = [[str(i) for i in arc] for arc in arcs]
        arcs = [" ".join(arc) for arc in arcs]
        arcs = "\n".join(arcs)

        fst = k2.Fsa.from_str(arcs, acceptor=False)
        # fst = k2.remove_epsilon(fst)  # Credit: Matthew W
        # fst = k2.expand_ragged_attributes(fst)
        return fst

    @staticmethod
    def determinize(k2_fst):
        fst = k2_to_openfst(k2_fst, olabels="aux_labels")

        det_fst = fst
        # det_fst = kaldifst.determinize(det_fst)
        kaldifst.determinize_star(
            det_fst, use_log=True
        )  # Determinize + weight pushing?
        # det_fst = kaldifst.minimize(fst)
        kaldifst.rmepsilon(det_fst)

        k2_fst = k2.Fsa.from_openfst(det_fst.to_str(), acceptor=False)
        # k2.determinize(k2_fst, k2.DeterminizeWeightPushingType.kLogWeightPushing)
        return k2_fst

    @staticmethod
    def minimize(k2_fst, allow_nondet=False):
        fst = k2_to_openfst(k2_fst, olabels="aux_labels")

        min_fst = fst
        kaldifst.minimize(
            min_fst, allow_nondet=allow_nondet,
        )
        kaldifst.rmepsilon(min_fst)

        k2_fst = k2.Fsa.from_openfst(min_fst.to_str(), acceptor=False)
        return k2_fst

    ####################################
    ## Class definition start here    ##
    ####################################

    def __init__(
        self,
        lang_dir: Path,
        device: torch.device,
        uniq_filename="uniq_lexicon.txt",
        oov: str = "<UNK>",
        sil_word = None,
        sil_token = None,
        nn_max_token_id = None,
    ):
        """
        Args:
          lexicon:
            It is built from `data/lang/lexicon.txt`.
          device:
            The device to use for operations compiling transcripts to FSAs.
          oov:
            Out of vocabulary word. When a word in the transcript
            does not exist in the lexicon, it is replaced with `oov`.
        """
        lexicon = UniqLexicon(lang_dir, uniq_filename=uniq_filename)
        L_inv = lexicon.L_inv.to(device)
        assert L_inv.requires_grad is False

        assert oov in lexicon.word_table

        self.L_inv = k2.arc_sort(L_inv)
        self.oov_id = lexicon.word_table[oov]
        self.word_table = lexicon.word_table

        # For debugging purpose only
        self.lexicon = lexicon
        self.start_tokens = {i for i in range(1, max_token_id + 1) if lexicon.token_table.get(i).startswith('▁')}

        if sil_token is not None:
            assert sil_token in lexicon.token_table
            self.sil_token_id = lexicon.token_table[sil_token]
        else:
            self.sil_token_id = None
        if sil_word is not None:
            assert sil_word in lexicon.word_table
            self.sil_word_id = lexicon.word_table[sil_word]
        else:
            self.sil_word_id = None

        self.nn_max_token_id = nn_max_token_id
        # self.sil_modeling = sil_modeling
        # if sil_modeling:
        #     self.sil_token_id = max(lexicon.tokens) + 1
        #     self.sil_word_id = max(self.word_table.ids) + 1

        max_token_id = max(lexicon.tokens)
        # hmm_topo = HMMTrainingGraphCompiler.hmm_topo_one_state_simplified(
        #     max_token_id + 1 if sil_word is not None else max_token_id, self.sil_token_id
        # )  # add one for the <sil> token
        # hmm_topo = k2.ctc_topo(max_token_id, modified=False)
        # hmm_topo = k2.ctc_topo(max_token_id, modified=True)
        # hmm_topo = HMMTrainingGraphCompiler.ctc_topo_modified(
        #     max_token_id, None
        # )
        # hmm_topo = HMMTrainingGraphCompiler.ctc_topo_modified_no_blk(
        #     max_token_id, None
        # )
        # hmm_topo = HMMTrainingGraphCompiler.ctc_topo_modified_debug_for_hmm(
        #     max_token_id + 1 if sil_word is not None else max_token_id, 
        #     lexicon.token_table["#0"],
        #     lexicon,
        # )
        hmm_topo = self.hmm_topo(max_token_id, self.start_tokens)
        print(f"Topo size: {(hmm_topo.shape[0], hmm_topo.num_arcs)}")

        self.topo = hmm_topo.to(device)
        self.device = device

        self.remove_intra_word_blk_flag = True
        print(f"self.remove_intra_word_blk_flag={self.remove_intra_word_blk_flag}")

    def _remove_intra_word_blk(self, decoding_graph, start_tokens, flag=True):
        c_str = k2.to_str_simple(decoding_graph)
        # print(c_str)

        arcs = c_str.split("\n")
        arcs = [x.strip() for x in arcs if len(x.strip()) > 0]
        final_state = int(arcs[-1])
        arcs = arcs[:-1]
        arcs = [tuple(map(int, a.split())) for a in arcs]
        # print(arcs)
        # print(final_state)

        if flag is False:
            new_arcs = arcs
            new_arcs.append([final_state])

            new_arcs = sorted(new_arcs, key=lambda arc: arc[0])
            new_arcs = [[str(i) for i in arc] for arc in new_arcs]
            new_arcs = [" ".join(arc) for arc in new_arcs]
            new_arcs = "\n".join(new_arcs)

            fst = k2.Fsa.from_str(new_arcs, acceptor=False)
            return fst

        state_arcs = defaultdict(list)
        for arc in arcs:
            state_arcs[arc[0]].append(arc)

        new_arcs = []
        for state, arc_list in state_arcs.items():
            eps_self_loop = None
            should_keep_self_loop = False
            for i, arc in enumerate(arc_list):
                if arc[0] == arc[1] and arc[2] == arc[3] == 0:
                    eps_self_loop = i
                if arc[2] in start_tokens or arc[2] == -1:
                    should_keep_self_loop = True
            
            if eps_self_loop is None or should_keep_self_loop:
                new_arcs.extend(arc_list)
            else:
                # print(f"state {state} should remove an arc {eps_self_loop}: {arc_list[eps_self_loop]}")
                new_arcs.extend(arc_list[:eps_self_loop])
                new_arcs.extend(arc_list[eps_self_loop+1:])
        new_arcs.append([final_state])

        new_arcs = sorted(new_arcs, key=lambda arc: arc[0])
        new_arcs = [[str(i) for i in arc] for arc in new_arcs]
        new_arcs = [" ".join(arc) for arc in new_arcs]
        new_arcs = "\n".join(new_arcs)

        fst = k2.Fsa.from_str(new_arcs, acceptor=False)
        return fst

    def remove_intra_word_blk(self, decoding_graphs, start_tokens, flag=True):
        if len(decoding_graphs.shape) == 2:
            decoding_graphs = k2.create_fsa_vec([decoding_graphs])
       
        num_fsas = decoding_graphs.shape[0]
        decoding_graph_list = []
        for i in range(num_fsas):
            decoding_graph_i = self._remove_intra_word_blk(decoding_graphs[i], start_tokens, flag=flag)
            decoding_graph_i = k2.connect(decoding_graph_i)
            decoding_graph_list.append(decoding_graph_i)
        
        decoding_graphs = k2.create_fsa_vec(decoding_graph_list)
        decoding_graphs = decoding_graphs.to(self.device)
        return decoding_graphs


    def compile(self, word_ids_list: List[List[int]]) -> k2.Fsa:
        """Build decoding graphs by composing ctc_topo with
        given transcripts.

        Args:
          texts:
            A list of strings. Each string contains a sentence for an utterance.
            A sentence consists of spaces separated words. An example `texts`
            looks like:

                ['hello icefall', 'CTC training with k2']

        Returns:
          An FsaVec, the composition result of `self.ctc_topo` and the
          transcript FSA.
        """
        transcript_fsa = self.convert_transcript_to_fsa(word_ids_list)

        # NOTE: k2.compose runs on CUDA only when treat_epsilons_specially
        # is False, so we add epsilon self-loops here
        fsa_with_self_loops = k2.remove_epsilon_and_add_self_loops(
            transcript_fsa
        )

        fsa_with_self_loops = k2.arc_sort(fsa_with_self_loops)

        decoding_graph = k2.compose(
            self.topo, fsa_with_self_loops, treat_epsilons_specially=False
        )

        # if len(decoding_graph.shape) == 2:
        #     decoding_graph = k2.connect(decoding_graph)
        #     decoding_graph = OneStateHMMTrainingGraphCompiler.determinize(decoding_graph)
        #     decoding_graph = k2.create_fsa_vec([decoding_graph])
        # else:
        #     num_fsas = decoding_graph.shape[0]
        #     decoding_graph_list = []
        #     for i in range(num_fsas):
        #         decoding_graph_i = decoding_graph[i]
        #         decoding_graph_i = k2.connect(decoding_graph_i)
        #         decoding_graph_i = OneStateHMMTrainingGraphCompiler.determinize(decoding_graph_i)
        #         decoding_graph_list.append(decoding_graph_i)
        #     decoding_graph = k2.create_fsa_vec(decoding_graph_list)

        # if len(decoding_graph.shape) == 2:
        #     decoding_graph = k2.connect(decoding_graph)
        #     decoding_graph = HMMTrainingGraphCompiler.minimize(decoding_graph, allow_nondet=True)
        #     decoding_graph = k2.create_fsa_vec([decoding_graph])
        # else:
        #     num_fsas = decoding_graph.shape[0]
        #     decoding_graph_list = []
        #     for i in range(num_fsas):
        #         decoding_graph_i = decoding_graph[i]
        #         decoding_graph_i = k2.connect(decoding_graph_i)
        #         decoding_graph_i = HMMTrainingGraphCompiler.minimize(decoding_graph_i, allow_nondet=True)
        #         decoding_graph_list.append(decoding_graph_i)
        #     decoding_graph = k2.create_fsa_vec(decoding_graph_list)

        decoding_graph = self.remove_intra_word_blk(decoding_graph, self.start_tokens, flag=self.remove_intra_word_blk_flag)

        assert decoding_graph.requires_grad is False

        decoding_graph = k2.connect(decoding_graph)        
        return decoding_graph

    def texts_to_ids(self, texts: List[str]) -> List[List[int]]:
        """Convert a list of texts to a list-of-list of word IDs.

        Args:
          texts:
            It is a list of strings. Each string consists of space(s)
            separated words. An example containing two strings is given below:

                ['HELLO ICEFALL', 'HELLO k2']
        Returns:
          Return a list-of-list of word IDs.
        """
        word_ids_list = []
        for text in texts:
            word_ids = []
            for word in text.split():
                if word in self.word_table:
                    word_ids.append(self.word_table[word])
                else:
                    word_ids.append(self.oov_id)
            word_ids_list.append(word_ids)
        return word_ids_list

    def linear_fsa_with_sil(self, word_ids_list, sil_word_id, device, sil_prob_utt=0.8, sil_prob_word=0.2):
        # ... inserting silence phones with probability 0.2
        # between the words and with probability 0.8 
        # at the beginning and end of the sentences. 
        # These constants have been selected intuitively 
        # and have not been tuned.

        non_sil_prob_utt = math.log(1.0 - sil_prob_utt)
        sil_prob_utt = math.log(sil_prob_utt)

        non_sil_prob_word = math.log(1.0 - sil_prob_word)
        sil_prob_word = math.log(sil_prob_word)

        fsa_list = []
        for word_ids in word_ids_list:
            s = ""
            cur_state_id = 0
            for i, w in enumerate(word_ids):
                if i == 0:
                    sil_prob = sil_prob_utt
                    non_sil_prob = non_sil_prob_utt
                else:
                    sil_prob = sil_prob_word
                    non_sil_prob = non_sil_prob_word
                
                s += f"{cur_state_id} {cur_state_id + 2} {w} {non_sil_prob}\n"
                s += f"{cur_state_id} {cur_state_id + 1} {sil_word_id} {sil_prob}\n"
                s += f"{cur_state_id + 1} {cur_state_id + 2} {w} 0\n"
                cur_state_id += 2

            # The last word
            s += f"{cur_state_id} {cur_state_id + 1} {sil_word_id} {sil_prob_utt}\n"
            s += f"{cur_state_id} {cur_state_id + 2} -1 {non_sil_prob_utt}\n"
            s += f"{cur_state_id + 1} {cur_state_id + 2} -1 0\n"
            s += f"{cur_state_id + 2}\n"

            fsa = k2.Fsa.from_str(s)
            fsa = k2.arc_sort(fsa)
            # fsa.aux_labels = fsa.labels.clone()
            fsa_list.append(fsa)
        
        fsa_vec = k2.create_fsa_vec(fsa_list)
        return fsa_vec.to(device)

    def convert_transcript_to_fsa(
            self, 
            word_ids_list: List[List[int]],
        ) -> k2.Fsa:
        """Convert a list of transcript texts to an FsaVec.

        Args:
          texts:
            A list of strings. Each string contains a sentence for an utterance.
            A sentence consists of spaces separated words. An example `texts`
            looks like:

                ['hello icefall', 'CTC training with k2']

        Returns:
          Return an FsaVec, whose `shape[0]` equals to `len(texts)`.
        """
        # word_ids_list = []
        # for text in texts:
        #     word_ids = []
        #     for word in text.split():
        #         if word in self.word_table:
        #             word_ids.append(self.word_table[word])
        #         else:
        #             word_ids.append(self.oov_id)
        #     word_ids_list.append(word_ids)

        if self.sil_word_id is None:
            word_fsa = k2.linear_fsa(word_ids_list, self.device)
        else:
            word_fsa = self.linear_fsa_with_sil(word_ids_list, self.sil_word_id, self.device)

        word_fsa_with_self_loops = k2.add_epsilon_self_loops(word_fsa)

        fsa = k2.intersect(
            self.L_inv, word_fsa_with_self_loops, treat_epsilons_specially=False
        )
        
        if self.sil_word_id is not None:
            labels = fsa.labels.clone()
            aux_labels = fsa.aux_labels.clone()
            if isinstance(fsa.labels, k2.RaggedTensor):
                labels.values[labels.values == self.sil_word_id] = 0
                aux_labels.values[aux_labels.values == self.sil_token_id] = self.nn_max_token_id + 1
            else:
                labels[labels == self.sil_word_id] = 0
                aux_labels[aux_labels == self.sil_token_id] = self.nn_max_token_id + 1
            fsa.labels = labels
            fsa.aux_labels = aux_labels

        # fsa has word ID as labels and token ID as aux_labels, so
        # we need to invert it
        ans_fsa = fsa.invert_()
        return k2.arc_sort(ans_fsa)
