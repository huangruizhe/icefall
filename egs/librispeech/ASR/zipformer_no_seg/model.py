# Copyright    2021-2023  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Zengwei Yao)
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

from typing import List, Optional, Union, Tuple
import math
from collections import defaultdict
import logging

import k2
import torch
import torch.nn as nn
from encoder_interface import EncoderInterface

from icefall.utils import add_sos, make_pad_mask
from scaling import ScaledLinear
from icefall.decode import get_lattice, one_best_decoding
from icefall.utils import get_alignments, get_texts

import kaldialign


def compute_wer(results):
    subs = defaultdict(int)
    ins = defaultdict(int)
    dels = defaultdict(int)

    # `words` stores counts per word, as follows:
    #   corr, ref_sub, hyp_sub, ins, dels
    words = defaultdict(lambda: [0, 0, 0, 0, 0])
    num_corr = 0
    ERR = "*"

    # if compute_CER:
    #     for i, res in enumerate(results):
    #         cut_id, ref, hyp = res
    #         ref = list("".join(ref))
    #         hyp = list("".join(hyp))
    #         results[i] = (cut_id, ref, hyp)
    sclite_mode = False

    cut_wers = {}

    for cut_id, ref, hyp in results:
        ali = kaldialign.align(ref, hyp, ERR, sclite_mode=sclite_mode)
        cut_wer = [0, 0, 0, 0]  # corr, sub, ins, dels
        for ref_word, hyp_word in ali:
            if ref_word == ERR:
                ins[hyp_word] += 1
                words[hyp_word][3] += 1
                cut_wer[2] += 1
            elif hyp_word == ERR:
                dels[ref_word] += 1
                words[ref_word][4] += 1
                cut_wer[3] += 1
            elif hyp_word != ref_word:
                subs[(ref_word, hyp_word)] += 1
                words[ref_word][1] += 1
                words[hyp_word][2] += 1
                cut_wer[1] += 1
            else:
                words[ref_word][0] += 1
                num_corr += 1
                cut_wer[0] += 1
        cut_wers[cut_id] = (sum(cut_wer[1:]) / (sum(cut_wer) - cut_wer[2]), cut_wer)
    ref_len = sum([len(r) for _, r, _ in results])
    sub_errs = sum(subs.values())
    ins_errs = sum(ins.values())
    del_errs = sum(dels.values())
    tot_errs = sub_errs + ins_errs + del_errs
    tot_err_rate = "%.2f" % (100.0 * tot_errs / ref_len)

    return cut_wers, f"{tot_err_rate} [{tot_errs}/{ref_len}] [ins:{ins_errs}, del:{del_errs}, sub:{sub_errs}]"


class AsrModel(nn.Module):
    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: EncoderInterface,
        decoder: Optional[nn.Module] = None,
        joiner: Optional[nn.Module] = None,
        encoder_dim: int = 384,
        decoder_dim: int = 512,
        vocab_size: int = 500,
        use_transducer: bool = True,
        use_ctc: bool = False,
    ):
        """A joint CTC & Transducer ASR model.

        - Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks (http://imagine.enpc.fr/~obozinsg/teaching/mva_gm/papers/ctc.pdf)
        - Sequence Transduction with Recurrent Neural Networks (https://arxiv.org/pdf/1211.3711.pdf)
        - Pruned RNN-T for fast, memory-efficient ASR training (https://arxiv.org/pdf/2206.13236.pdf)

        Args:
          encoder_embed:
            It is a Convolutional 2D subsampling module. It converts
            an input of shape (N, T, idim) to an output of of shape
            (N, T', odim), where T' = (T-3)//2-2 = (T-7)//2.
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dim) and
            `logit_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, decoder_dim).
            It should contain one attribute: `blank_id`.
            It is used when use_transducer is True.
          joiner:
            It has two inputs with shapes: (N, T, encoder_dim) and (N, U, decoder_dim).
            Its output shape is (N, T, U, vocab_size). Note that its output contains
            unnormalized probs, i.e., not processed by log-softmax.
            It is used when use_transducer is True.
          use_transducer:
            Whether use transducer head. Default: True.
          use_ctc:
            Whether use CTC head. Default: False.
        """
        super().__init__()

        assert (
            use_transducer or use_ctc
        ), f"At least one of them should be True, but got use_transducer={use_transducer}, use_ctc={use_ctc}"

        assert isinstance(encoder, EncoderInterface), type(encoder)

        self.encoder_embed = encoder_embed
        self.encoder = encoder

        self.use_transducer = use_transducer
        if use_transducer:
            # Modules for Transducer head
            assert decoder is not None
            assert hasattr(decoder, "blank_id")
            assert joiner is not None

            self.decoder = decoder
            self.joiner = joiner

            self.simple_am_proj = ScaledLinear(
                encoder_dim, vocab_size, initial_scale=0.25
            )
            self.simple_lm_proj = ScaledLinear(
                decoder_dim, vocab_size, initial_scale=0.25
            )
        else:
            assert decoder is None
            assert joiner is None

        self.use_ctc = use_ctc
        if use_ctc:
            # Modules for CTC head
            self.ctc_output = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(encoder_dim, vocab_size),
                nn.LogSoftmax(dim=-1),
            )
        
        self.scratch_space = {}

    def forward_encoder(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute encoder outputs.
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.

        Returns:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
        """
        # logging.info(f"Memory allocated at entry: {torch.cuda.memory_allocated() // 1000000}M")
        x, x_lens = self.encoder_embed(x, x_lens)
        # logging.info(f"Memory allocated after encoder_embed: {torch.cuda.memory_allocated() // 1000000}M")

        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)

        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)

        return encoder_out, encoder_out_lens

    def forward_ctc(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        reduction: str = "sum",
    ) -> torch.Tensor:
        """Compute CTC loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          targets:
            Target Tensor of shape (sum(target_lengths)). The targets are assumed
            to be un-padded and concatenated within 1 dimension.
        """
        # Compute CTC log-prob
        ctc_output = self.ctc_output(encoder_out)  # (N, T, C)

        if self.scratch_space["my_args"] is not None:
            self.scratch_space["my_args"]["ctc_output"] = ctc_output

        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2),  # (T, N, C)
            targets=targets,
            input_lengths=encoder_out_lens,
            target_lengths=target_lengths,
            reduction=reduction,
        )
        return ctc_loss, torch.tensor(0)
  
    def encode_supervisions(
        self, targets, target_lengths, input_lengths
    ) -> Tuple[torch.Tensor, Union[List[str], List[List[int]]]]:
        """
        Encodes Lhotse's ``batch["supervisions"]`` dict into
        a pair of torch Tensor, and a list of transcription strings or token indexes

        The supervision tensor has shape ``(batch_size, 3)``.
        Its second dimension contains information about sequence index [0],
        start frames [1] and num frames [2].

        The batch items might become re-ordered during this operation -- the
        returned tensor and list of strings are guaranteed to be consistent with
        each other.
        """
        batch_size = input_lengths.size(0)
        supervision_segments = torch.stack(
            (
                torch.arange(batch_size),
                torch.zeros(batch_size),
                input_lengths.cpu(),
            ),
            1,
        ).to(torch.int32)

        indices = torch.argsort(supervision_segments[:, 2], descending=True)
        supervision_segments = supervision_segments[indices]
        # import pdb; pdb.set_trace()

        # res = targets[indices].tolist()
        # res_lengths = target_lengths[indices].tolist()
        # res = [[i + 1 for i in l[:l_len]] for l, l_len in zip(res, res_lengths)]  # hard-coded for torchaudio

        return supervision_segments, None, indices

    def check_lattice(self, lattice):
        best_path = one_best_decoding(
            lattice=lattice,
            use_double_scores=True,
        )
        token_ids = get_texts(best_path)
        hyps = self.scratch_space["sp"].decode(token_ids)
        return hyps
  
    def check_lattice2(self, lattice, indices, i):
        best_path = one_best_decoding(
            lattice=lattice,
            use_double_scores=True,
        )
        token_ids = get_texts(best_path)
        hyps = self.scratch_space["sp"].decode(token_ids)

        cut_id = self.scratch_space['cuts'][indices[i].item()].id
        print(f"[{cut_id}: ref] {self.scratch_space['texts'][indices[i].item()]}")
        print(f"[{cut_id}: hyp] {hyps[i]}")
        # print(f"[cut] {self.scratch_space['cuts'][indices[i].item()]}")

        # print(f"[ref] {self.scratch_space['texts'][i]}")
        # print(f"[hyp] {hyps[i]}")
        # print(f"[cut] {self.scratch_space['cuts'][i]}")
    
    def check_lattice3(self, lattice, indices):
        best_path = one_best_decoding(
            lattice=lattice,
            use_double_scores=True,
        )
        token_ids = get_texts(best_path)
        hyps = self.scratch_space["sp"].decode(token_ids)

        supervisions = self.scratch_space["my_args"]["supervisions"]
        texts = supervisions["text"]
        cuts = supervisions['cut']
        cut_ids = [c.id for c in cuts]
        results = list()
        params = self.scratch_space["params"]
        for i in range(len(texts)):
            i_original = indices[i].item()
            cut_id, hyp_text, ref_text = cut_ids[i_original], hyps[i], texts[i_original]
            hyp_words = hyp_text.split()
            ref_words = ref_text.split()
            results.append((cut_id, ref_words, hyp_words))

        # compute wer for the batch
        cut_wer, wer = compute_wer(results)
        logging.info(f"[epoch {params.cur_epoch} - batch {params.batch_idx_train}] [batch_size: {len(texts)}] wer: {wer}")

        # max_wer_cut_id = max(cut_wer, key=lambda cut_id: cut_wer[cut_id][0])
        # entry = next((cut_id, ref_words, hyp_words) for cut_id, ref_words, hyp_words in results if cut_id == max_wer_cut_id)
        # logging.info(f"[{max_wer_cut_id}: wer] {cut_wer[max_wer_cut_id]}")
        # logging.info(f"[{max_wer_cut_id}: ref] {entry[1]}")
        # logging.info(f"[{max_wer_cut_id}: hyp] {entry[2]}")
        # logging.info(f"[{max_wer_cut_id} batch] {[f'{val[0]:.2f}' for val in sorted(cut_wer.values(), key=lambda x: x[0], reverse=True)]}")

    def get_log_priors_batch(self, log_probs, input_lengths):
        # log_probs:  (N, T, C)
        # input_lengths:  (N,)

        log_probs_flattened = []
        for lp, le in zip(log_probs, input_lengths):
            log_probs_flattened.append(lp[:int(le.item())])      
        log_probs_flattened = torch.cat(log_probs_flattened, 0)

        # Note, the log_probs here is already log_softmax'ed.
        T = log_probs_flattened.size(0)
        log_batch_priors_sum = torch.logsumexp(log_probs_flattened, dim=0, keepdim=True)
        log_priors = log_batch_priors_sum.detach() - math.log(T)

        # Clipping mechanism
        prior_threshold = -12.0
        log_priors = torch.where(log_priors < prior_threshold, prior_threshold, log_priors)

        return log_priors

    def check_eval_wer(self, x, x_lens, y, y_list, my_args):
        # !import code; code.interact(local=vars())
        # _=self.eval()
        # _=self.train()
        # self.training
        
        _=self.eval()
        self.training

        encoder_out, encoder_out_lens = self.forward_encoder(x, x_lens)
        ctc_output = self.ctc_output(encoder_out)  # (N, T, C)
        
        supervision_segments, texts, indices = self.encode_supervisions(None, None, encoder_out_lens)

        if y_list is not None:
            _y_list = [y_list[i] for i in indices.tolist()]
            decoding_graph = k2.create_fsa_vec(_y_list)
            decoding_graph = k2.arc_sort(decoding_graph)
            decoding_graph = decoding_graph.to(encoder_out.device)
        else:
            texts = self.scratch_space["texts"]
            sp = self.scratch_space["sp"]
            make_factor_transducer1 = self.scratch_space["make_factor_transducer1"]
            y_list = [make_factor_transducer1(sp.encode(text, out_type=int), return_str=False, blank_penalty=0) for text in texts]
            _y_list = [y_list[i] for i in indices.tolist()]
            decoding_graph = k2.create_fsa_vec(_y_list)
            decoding_graph = k2.arc_sort(decoding_graph)
            decoding_graph = decoding_graph.to(encoder_out.device)
       
        lattice = get_lattice(
            nnet_output=ctc_output,
            decoding_graph=decoding_graph,
            supervision_segments=supervision_segments,
            search_beam=15,
            output_beam=6,
            min_active_states=30,
            max_active_states=10000,
            subsampling_factor=self.scratch_space["subsampling_factor"],
        )
        # for i in range(len(_y_list)):
        #     self.check_lattice2(lattice, indices, i)

        logging.info("Eval:")
        self.check_lattice3(lattice, indices)
        # logging.info("Train:")

        _=self.train()


    def forward_ctc_long_form(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        targets,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          targets:
            Target Tensor of shape (sum(target_lengths)). The targets are assumed
            to be un-padded and concatenated within 1 dimension.
        """
        # Compute CTC log-prob
        ctc_output = self.ctc_output(encoder_out)  # (N, T, C)

        batch_size = ctc_output.size(0)
        t = (ctc_output.argmax(-1) != 0).sum().item()
        # if t < 10 * batch_size:
        if False:
            # apply blank penalty
            penalty = torch.zeros(ctc_output.size(-1), device=ctc_output.device)
            penalty[0] = 10.0
            # penalty[0] = ctc_output.mean()
            # penalty = ctc_output.mean(dim=-1)
            # log_priors = self.get_log_priors_batch(ctc_output, encoder_out_lens)
            # penalty = 0.2 * log_priors
            # breakpoint()
            ctc_output = ctc_output - penalty
            ctc_output = nn.functional.log_softmax(ctc_output, dim=-1)

        supervision_segments, texts, indices = self.encode_supervisions(targets, target_lengths, encoder_out_lens)
        # my_args = self.scratch_space["my_args"]
        # supervisions = my_args["supervisions"]
        # supervision_segments = torch.stack(
        #     (
        #         supervisions["sequence_idx"],
        #         torch.div(
        #             supervisions["start_frame"],
        #             self.scratch_space["subsampling_factor"],
        #             rounding_mode="floor",
        #         ),
        #         torch.div(
        #             supervisions["num_frames"],
        #             self.scratch_space["subsampling_factor"],
        #             rounding_mode="floor",
        #         ),
        #     ),
        #     1,
        # ).to(torch.int32)

        # indices = torch.argsort(supervision_segments[:, 2], descending=True)
        # print(indices)
        
        # from icefall.lexicon import Lexicon

        # params = self.scratch_space["params"]
        # lexicon = Lexicon('data/lang_bpe_500')
        # max_token_id = max(lexicon.tokens)
        # num_classes = max_token_id + 1  # +1 for the blank

        # params.vocab_size = num_classes
        # # <blk> and <unk> are defined in local/train_bpe_model.py
        # params.blank_id = 0

        # device = torch.device("cuda", 0)

        # HLG = None
        # H = k2.ctc_topo(
        #     max_token=max_token_id,
        #     modified=False,
        #     device=device,
        # )
        # bpe_model = self.scratch_space["sp"]

        # decoding_graph = H
 
        # targets is a list of k2 fst
        y_list = targets
        _y_list = [y_list[i] for i in indices.tolist()]
        decoding_graph = k2.create_fsa_vec(_y_list)
        decoding_graph = k2.arc_sort(decoding_graph)
        decoding_graph = decoding_graph.to(encoder_out.device)
        # print(f"decoding_graph: #states: {decoding_graph.shape[0]}, #arcs: {decoding_graph.num_arcs}")

        # get a moving average of label priors
        if False:
            log_probs = ctc_output  # (N, T, C)
            input_lengths = encoder_out_lens  # (N,)

            log_probs_flattened = []
            for lp, le in zip(log_probs, input_lengths):
                log_probs_flattened.append(lp[:int(le.item())])      
            log_probs_flattened = torch.cat(log_probs_flattened, 0)

            # Note, the log_probs here is already log_softmax'ed.
            T = log_probs_flattened.size(0)
            log_batch_priors_sum = torch.logsumexp(log_probs_flattened, dim=0, keepdim=True)
            log_batch_priors_sum = log_batch_priors_sum.detach()
            if self.scratch_space["log_priors"] is None:
                self.scratch_space["log_priors"] = log_batch_priors_sum - math.log(T)
                self.scratch_space["priors_T"] = T
            else:
                _temp = torch.stack([self.scratch_space["log_priors"] + math.log(self.scratch_space["priors_T"]), log_batch_priors_sum], dim=-1)
                _temp = torch.logsumexp(_temp, dim=-1)
                self.scratch_space["priors_T"] += T
                self.scratch_space["log_priors"] = _temp - math.log(self.scratch_space["priors_T"])
            
            # Clip the priors for stability
            if self.scratch_space["log_priors"] is not None:
                prior_threshold = -12.0
                self.scratch_space["log_priors"] = torch.where(self.scratch_space["log_priors"] < prior_threshold, prior_threshold, self.scratch_space["log_priors"])
                # if self.scratch_space["rank"] == 0:
                #     print("new_log_prior (clipped): ", ["{0:0.2f}".format(i) for i in self.scratch_space["log_priors"][0].tolist()])

        # apply label priors
        if False and self.scratch_space["log_priors"] is not None:
            prior_scaling_factor = 0.25
            log_probs = ctc_output - self.scratch_space["log_priors"] * prior_scaling_factor
        else:
            log_probs = ctc_output
        
        # dense_fsa_vec = k2.DenseFsaVec(
        #     log_probs,
        #     supervision_segments,
        #     allow_truncate=self.scratch_space["subsampling_factor"] - 1,
        # )

        # ctc_loss = k2.ctc_loss(
        #     decoding_graph=decoding_graph,
        #     dense_fsa_vec=dense_fsa_vec,
        #     output_beam=self.scratch_space["ctc_beam_size"],
        #     reduction="sum",
        #     use_double_scores=True,
        #     # delay_penalty=0.05,
        # )
            
        # print(f"log_prob [{self.training}]")
        # print(f"log_prob [{log_probs[0][:50].argmax(dim=-1)}]")

        # https://github.com/k2-fsa/k2/blob/master/k2/python/k2/ctc_loss.py#L138
        lattice = get_lattice(
            nnet_output=log_probs,
            decoding_graph=decoding_graph,
            supervision_segments=supervision_segments,
            # search_beam=15,
            # output_beam=6,
            # min_active_states=30,
            # max_active_states=10000,
            search_beam=20,
            output_beam=5,
            min_active_states=30,
            max_active_states=10000,
            subsampling_factor=self.scratch_space["subsampling_factor"],
        )

        # breakpoint()
        # !print(*list(enumerate(batch["supervisions"]["text"])), sep="\n")
        # !self.check_lattice2(lattice, indices, 0)

        # for i in range(len(_y_list)):
        #     self.check_lattice2(lattice, indices, i)
        # exit(1)
        # self.check_lattice2(lattice, indices, 0)
        # self.check_lattice3(lattice, indices)

        # breakpoint()
        # print(f"num_arcs after pruning: {lattice.arcs.num_elements()}")
        # logging.info(f"LG in k2: #states: {lattice.shape[0]}, #arcs: {lattice.num_arcs}")

        # Option1: best path
        best_path = one_best_decoding(
            lattice=lattice,
            use_double_scores=True,
        )
        tot_scores = best_path.get_tot_scores(
            log_semiring=True, use_double_scores=True,
        )
        loss = -1 * tot_scores
        loss = loss.to(torch.float32)

        # # Option2: total score:
        # tot_scores = lattice.get_tot_scores(
        #     log_semiring=True, use_double_scores=True,
        # )
        # loss = -1 * tot_scores
        # loss = loss.to(torch.float32)

        # breakpoint()
        inf_indices = torch.where(torch.isinf(loss))
        if inf_indices[0].size(0) > 0:
          loss[inf_indices] = 0
          loss[inf_indices].detach()
          self.scratch_space["inf_indices"] = inf_indices

        ctc_loss = loss.sum()

        # Compute an aux entropy loss to encourage the model to avoid empty sequence or to produce diversified results
        # The idea is:
        # 1. As we don't know the ground truth (transcription), so cannot compute some cross entropy loss vs. the ground truth
        # 2. From copilot: [We can encourage the model to avoid empty sequence or to produce diversified results by computing the entropy of the model's output]
        # 3. The empty sequence is basically all blank tokens -- in log_probs, blank has the highest log-prob
        # 4. What we hope is that the model can produce some reasonable non-blank spikes in the posteriorgram.
        # 5. So the loss function is: 
        #       t = the number of spikes
        #       g(t) = (1/a * t)^-1   # where when t is small g(t) is large, when t=a then g(t)=1,  "a" is a hyper-parameter
        #       loss = g(t) * log_probs[:,:,0]
        def aux_loss(_log_probs, _m=4.0):
          # t = len(torch.unique(_log_probs.argmax(-1)))
          t = (log_probs.argmax(-1) != 0).sum().item()
          g_t = (1/_m * max(t - 0.7, 1e-3))**(-1)  # g(t) is a scalar coefficient
          g_t = g_t if g_t > 1 else 0
          return g_t * -torch.log(-_log_probs[...,0]).sum()
                      
        return ctc_loss, torch.tensor(0)  # aux_loss(log_probs, _m=10.0)

    def forward_ctc_long_form2(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        targets,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          targets:
            Target Tensor of shape (sum(target_lengths)). The targets are assumed
            to be un-padded and concatenated within 1 dimension.
        """
        # Compute CTC log-prob
        ctc_output = self.ctc_output(encoder_out)  # (N, T, C)

        if self.scratch_space["my_args"] is not None:
            self.scratch_space["my_args"]["ctc_output"] = ctc_output

        supervision_segments, texts, indices = self.encode_supervisions(targets, target_lengths, encoder_out_lens)
 
        # targets is a list of k2 fst
        y_list = targets
        _y_list = [y_list[i] for i in indices.tolist()]
        decoding_graph = k2.create_fsa_vec(_y_list)
        decoding_graph = k2.arc_sort(decoding_graph)
        decoding_graph = decoding_graph.to(encoder_out.device)
        # print(f"decoding_graph: #states: {decoding_graph.shape[0]}, #arcs: {decoding_graph.num_arcs}")

        log_probs = ctc_output

        # https://github.com/k2-fsa/k2/blob/master/k2/python/k2/ctc_loss.py#L138
        lattice = get_lattice(
            nnet_output=log_probs,
            decoding_graph=decoding_graph,
            supervision_segments=supervision_segments,
            # search_beam=15,
            # output_beam=6,
            # min_active_states=30,
            # max_active_states=10000,
            search_beam=20,
            output_beam=5,
            min_active_states=30,
            max_active_states=10000,
            subsampling_factor=self.scratch_space["subsampling_factor"],
        )

        # breakpoint()
        # !print(*list(enumerate(batch["supervisions"]["text"])), sep="\n")
        # !self.check_lattice2(lattice, indices, 0)

        # for i in range(len(_y_list)):
        #     self.check_lattice2(lattice, indices, i)
        # exit(1)
        # self.check_lattice2(lattice, indices, 0)
        # self.check_lattice3(lattice, indices)

        # breakpoint()
        # print(f"num_arcs after pruning: {lattice.arcs.num_elements()}")
        # logging.info(f"LG in k2: #states: {lattice.shape[0]}, #arcs: {lattice.num_arcs}")

        # Option1: best path
        best_path = one_best_decoding(
            lattice=lattice,
            use_double_scores=True,
        )
        token_ids = get_texts(best_path)
        # TODO: we can get aligment time stamps here
        _indices = {i_old : i_new for i_new, i_old in enumerate(indices.tolist())}
        _token_ids = [token_ids[_indices[i]] for i in range(len(token_ids))]
        token_ids = _token_ids

        if "libri_long_text_str" in self.scratch_space["my_args"] and self.scratch_space["my_args"]["libri_long_text_str"] is not None:
            libri_long_text_str = self.scratch_space["my_args"]["libri_long_text_str"]
            get_uid_key = self.scratch_space["get_uid_key"]
            sp = self.scratch_space["sp"]
            cuts = self.scratch_space["cuts"]
            _texts = [libri_long_text_str[tuple(get_uid_key(c.id)[:2])][max(rg[0]-1, 0): rg[1]-1] for c, rg in zip(cuts, token_ids)]
            _texts = [" ".join(t) for t in _texts]
            token_ids = sp.encode(_texts, out_type=int)

        # compute wer for the batch
        if True:
            ref_texts = self.scratch_space["texts"]
            hyp_texts = self.scratch_space["sp"].decode(token_ids)
            # logging.info(f"hyp_texts: {hyp_texts}")
            cuts = self.scratch_space["cuts"]
            results = [(c.id, ref.split(), hyp.split()) for c, hyp, ref in zip(cuts, hyp_texts, ref_texts)]
            cut_wer, wer = compute_wer(results)
            logging.info(f"[epoch {self.scratch_space['params'].cur_epoch} - batch {self.scratch_space['params'].batch_idx_train}] [batch_size: {len(ref_texts)}] wer: {wer}")

        # use decoding results as supervision
        y = k2.RaggedTensor(token_ids)
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        logging.info(f"[epoch {self.scratch_space['params'].cur_epoch} - batch {self.scratch_space['params'].batch_idx_train}] hyps_lens: {sorted(y_lens.tolist())}")

        loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2),  # (T, N, C)
            targets=y.values,
            input_lengths=encoder_out_lens,
            target_lengths=y_lens,
            reduction='none',
        )

        inf_indices = torch.where(torch.isinf(loss))
        if inf_indices[0].size(0) > 0:
          loss[inf_indices] = 0
          loss[inf_indices].detach()
          self.scratch_space["inf_indices"] = inf_indices

        ctc_loss = loss.sum()
                      
        return ctc_loss, torch.tensor(0)

    def forward_transducer(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        y: k2.RaggedTensor,
        y_lens: torch.Tensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Transducer loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        """
        # Now for the decoder, i.e., the prediction network
        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

        # decoder_out: [B, S + 1, decoder_dim]
        decoder_out = self.decoder(sos_y_padded)

        # Note: y does not start with SOS
        # y_padded : [B, S]
        y_padded = y.pad(mode="constant", padding_value=0)

        y_padded = y_padded.to(torch.int64)
        boundary = torch.zeros(
            (encoder_out.size(0), 4),
            dtype=torch.int64,
            device=encoder_out.device,
        )
        boundary[:, 2] = y_lens
        boundary[:, 3] = encoder_out_lens

        lm = self.simple_lm_proj(decoder_out)
        am = self.simple_am_proj(encoder_out)

        # if self.training and random.random() < 0.25:
        #    lm = penalize_abs_values_gt(lm, 100.0, 1.0e-04)
        # if self.training and random.random() < 0.25:
        #    am = penalize_abs_values_gt(am, 30.0, 1.0e-04)

        with torch.cuda.amp.autocast(enabled=False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),
                am=am.float(),
                symbols=y_padded,
                termination_symbol=blank_id,
                lm_only_scale=lm_scale,
                am_only_scale=am_scale,
                boundary=boundary,
                reduction="sum",
                return_grad=True,
            )

        # ranges : [B, T, prune_range]
        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
        )

        # am_pruned : [B, T, prune_range, encoder_dim]
        # lm_pruned : [B, T, prune_range, decoder_dim]
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=self.joiner.decoder_proj(decoder_out),
            ranges=ranges,
        )

        # logits : [B, T, prune_range, vocab_size]

        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        logits = self.joiner(am_pruned, lm_pruned, project_input=False)

        with torch.cuda.amp.autocast(enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=y_padded,
                ranges=ranges,
                termination_symbol=blank_id,
                boundary=boundary,
                reduction="sum",
            )

        return simple_loss, pruned_loss

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        my_args = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        Returns:
          Return the transducer losses and CTC loss,
          in form of (simple_loss, pruned_loss, ctc_loss)

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        if isinstance(y, tuple):
            y, y_long = y

        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0, (x.shape, x_lens.shape, y.dim0)

        # _=self.eval()
        # _=self.train()
        # if self.training:
        #     breakpoint()
        # # !import code; code.interact(local=vars())

        # if my_args is not None and "libri_long_text" in my_args:
        # if my_args is not None:
        #     self.check_eval_wer(x, x_lens, y, y_long, my_args)

        # Compute encoder outputs
        encoder_out, encoder_out_lens = self.forward_encoder(x, x_lens)

        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        if self.use_transducer:
            # Compute transducer loss
            simple_loss, pruned_loss = self.forward_transducer(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                y=y.to(x.device),
                y_lens=y_lens,
                prune_range=prune_range,
                am_scale=am_scale,
                lm_scale=lm_scale,
            )
        else:
            simple_loss = torch.empty(0)
            pruned_loss = torch.empty(0)

        if self.use_ctc:
            # Compute CTC loss
            targets = y.values   # on CPU
            # (Pdb) !encoder_out.device, encoder_out_lens.device, targets.device, y_lens.device
            # (device(type='cuda', index=0), device(type='cuda', index=0), device(type='cpu'), device(type='cpu'))
            
            # b zipformer_no_seg/model.py:425
            # torch.cuda.get_device_properties(0).total_memory/1024/1024
            # torch.cuda.memory_reserved(0)/1024/1024
            # torch.cuda.memory_allocated(0)/1024/1024

            # if True:
            if not self.training or my_args is None or "libri_long_text" not in my_args:
                ctc_loss, ctc_aux_loss = self.forward_ctc(
                    encoder_out=encoder_out,
                    encoder_out_lens=encoder_out_lens,
                    targets=targets,
                    target_lengths=y_lens,
                )
            else:
                ctc_loss, ctc_aux_loss = torch.tensor(0), torch.tensor(0)

            if my_args is not None and "libri_long_text" in my_args and self.training:
                ctc_loss_long, ctc_aux_loss = self.forward_ctc_long_form2(
                    encoder_out=encoder_out,
                    encoder_out_lens=encoder_out_lens,
                    targets=y_long,
                    target_lengths=y_lens,
                )
                # import logging
                # logging.info(f"ctc_aux_loss = {ctc_aux_loss}")
            else:
                ctc_loss_long = torch.tensor(0)

        else:
            ctc_loss = torch.empty(0)
            ctc_aux_loss = torch.empty(0)

        return simple_loss, pruned_loss, ctc_loss, ctc_loss_long, ctc_aux_loss
