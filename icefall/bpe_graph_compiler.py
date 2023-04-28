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


from pathlib import Path
from typing import List, Union
from collections import defaultdict

import k2
import sentencepiece as spm
import torch


class BpeCtcTrainingGraphCompiler(object):
    def __init__(
        self,
        lang_dir: Path,
        device: Union[str, torch.device] = "cpu",
        sos_token: str = "<sos/eos>",
        eos_token: str = "<sos/eos>",
    ) -> None:
        """
        Args:
          lang_dir:
            This directory is expected to contain the following files:

                - bpe.model
                - words.txt
          device:
            It indicates CPU or CUDA.
          sos_token:
            The word piece that represents sos.
          eos_token:
            The word piece that represents eos.
        """
        lang_dir = Path(lang_dir)
        model_file = lang_dir / "bpe.model"
        sp = spm.SentencePieceProcessor()
        sp.load(str(model_file))
        self.sp = sp
        self.word_table = k2.SymbolTable.from_file(lang_dir / "words.txt")
        self.device = device

        self.sos_id = self.sp.piece_to_id(sos_token)
        self.eos_id = self.sp.piece_to_id(eos_token)

        assert self.sos_id != self.sp.unk_id()
        assert self.eos_id != self.sp.unk_id()

        self.start_tokens = {token_id for token_id in range(sp.vocab_size()) if sp.id_to_piece(token_id).startswith("▁")}
        self.remove_intra_word_blk_flag = True
        print(f"self.remove_intra_word_blk_flag={self.remove_intra_word_blk_flag}")

    def texts_to_ids(self, texts: List[str]) -> List[List[int]]:
        """Convert a list of texts to a list-of-list of piece IDs.

        Args:
          texts:
            It is a list of strings. Each string consists of space(s)
            separated words. An example containing two strings is given below:

                ['HELLO ICEFALL', 'HELLO k2']
        Returns:
          Return a list-of-list of piece IDs.
        """
        return self.sp.encode(texts, out_type=int)

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
            condition1 = False
            condition2 = False
            eps_arc_i = None
            for i, arc in enumerate(arc_list):
                if arc[0] == arc[1] and arc[2] > 0:
                    condition1 = True  # We should process this kind of state
                elif arc[0] != arc[1] and arc[2] > 0 and arc[2] not in start_tokens:
                    condition2 = True
                elif arc[0] != arc[1] and arc[2] == 0:
                    eps_arc_i = i
            
            if condition1 and condition2:
                # print(f"state {state} should remove an arc {eps_self_loop}: {arc_list[eps_self_loop]}")
                new_arcs.extend(arc_list[:eps_arc_i])
                new_arcs.extend(arc_list[eps_arc_i+1:])
            else:
                new_arcs.extend(arc_list)
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
        decoding_graphs = k2.arc_sort(decoding_graphs)
        decoding_graphs = decoding_graphs.to(self.device)
        return decoding_graphs

    def compile(
        self,
        piece_ids: List[List[int]],
        modified: bool = False,
    ) -> k2.Fsa:
        """Build a ctc graph from a list-of-list piece IDs.

        Args:
          piece_ids:
            It is a list-of-list integer IDs.
          modified:
           See :func:`k2.ctc_graph` for its meaning.
        Return:
          Return an FsaVec, which is the result of composing a
          CTC topology with linear FSAs constructed from the given
          piece IDs.
        """
        graph = k2.ctc_graph(piece_ids, modified=modified, device=self.device)

        graph = self.remove_intra_word_blk(graph, self.start_tokens, flag=self.remove_intra_word_blk_flag)
        return graph
