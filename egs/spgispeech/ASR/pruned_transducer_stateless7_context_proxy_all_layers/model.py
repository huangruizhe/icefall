# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang, Wei Kang)
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


import random

import k2
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder_interface import EncoderInterface
from scaling import penalize_abs_values_gt

from icefall.utils import add_sos
from typing import Union, List

import sys
#sys.path.insert(0,'/exp/rhuang/meta/ctc')
#import ctc as ctc_primer
import torch.distributed as dist
import logging
from torch.nn.utils.rnn import pad_sequence


class Transducer(nn.Module):
    """It implements https://arxiv.org/pdf/1211.3711.pdf
    "Sequence Transduction with Recurrent Neural Networks"
    """

    def __init__(
        self,
        encoder: EncoderInterface,
        decoder: nn.Module,
        joiner: nn.Module,
        encoder_dim: int,
        decoder_dim: int,
        joiner_dim: int,
        vocab_size: int,
        context_encoder: nn.Module, 
        encoder_biasing_adapter: nn.Module, 
        decoder_biasing_adapter: nn.Module,
        i_ctc_layers = [],
    ):
        """
        Args:
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dm) and
            `logit_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, decoder_dim).
            It should contain one attribute: `blank_id`.
          joiner:
            It has two inputs with shapes: (N, T, encoder_dim) and (N, U, decoder_dim).
            Its output shape is (N, T, U, vocab_size). Note that its output contains
            unnormalized probs, i.e., not processed by log-softmax.
        """
        super().__init__()
        assert isinstance(encoder, EncoderInterface), type(encoder)
        assert hasattr(decoder, "blank_id")

        self.encoder = encoder
        self.decoder = decoder
        self.joiner = joiner

        self.context_encoder = context_encoder
        self.encoder_biasing_adapter = encoder_biasing_adapter
        self.decoder_biasing_adapter = decoder_biasing_adapter

        self.simple_am_proj = nn.Linear(
            encoder_dim,
            vocab_size,
        )
        self.simple_lm_proj = nn.Linear(decoder_dim, vocab_size)

        # For temporary convenience
        self.scratch_space = None
        self.no_encoder_biasing = None
        self.no_decoder_biasing = None
        self.no_wfst_lm_biasing = None
        self.params = None

        # Apply ctc loss on the layer output (biased)
        self.use_ctc = False and len(i_ctc_layers) > 0
        if self.use_ctc:  
            i_dim3 = self.encoder.encoder_dims[3]
            # i_dim5 = self.encoder.encoder_dims[5]
            # Modules for CTC head
            self.ctc_output3 = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(i_dim3, vocab_size),
                nn.LogSoftmax(dim=-1),
            )
            # self.ctc_output5 = nn.Sequential(
            #     nn.Dropout(p=0.1),
            #     nn.Linear(i_dim5, vocab_size),
            #     nn.LogSoftmax(dim=-1),
            # )
        
        # Apply ctc loss on the `encoder_biasing_out` term only
        self.use_ctc2 = True and len(i_ctc_layers) > 0
        if self.use_ctc2:
            self.i_ctc2_layers = i_ctc_layers
            # self.i_ctc2_layers = [3]
            self.ctc2_outputs = [None] * (len(self.encoder.encoder_dims) + 1)
            for i in range(len(self.encoder.encoder_dims) + 1):
                if i in self.i_ctc2_layers:
                    j = i if i < len(self.encoder.encoder_dims) else -1
                    self.ctc2_outputs[i] = nn.Sequential(
                        # nn.Dropout(p=0.1),
                        # nn.Linear(self.encoder.encoder_dims[j], self.encoder.encoder_dims[j]),
                        # nn.Tanh(),
                        nn.Dropout(p=0.1),
                        nn.Linear(self.encoder.encoder_dims[j], vocab_size),
                        nn.LogSoftmax(dim=-1),
                    )
            self.ctc2_outputs = nn.ModuleList(self.ctc2_outputs)
        
        # Apply ctc loss on the attention weights
        self.use_ctc3 = False and len(i_ctc_layers) > 0
        if self.use_ctc3:
            # self.i_ctc3_layers = [3, 5]
            self.i_ctc3_layers = i_ctc_layers
            
            self.priors_T = 0
            self.log_priors = None  # (D,)
            self.log_priors_sum = None  # (1, D)
            self.max_dim = 150
        
        self.ce_loss = False and len(i_ctc_layers) > 0
        if self.ce_loss:
            # self.ce_layers = [3, 5]
            self.ce_layers = i_ctc_layers


    def pad_probs(self, probs, dim, pad_value=0):
        # Given a tensor of dimensions (*,D), 
        # Let's make it of size (*, D') where D'>=D or D'<D, and we can pad 0s if needed
        
        if probs.size(-1) == dim:
            return probs
        elif probs.size(-1) < dim:
            padding_size = dim - probs.size(1)
            return F.pad(probs, (0, padding_size), 'constant', pad_value)
        else:
            return probs[:, :dim]


    def forward_ctc_primer(self, log_probs, targets, input_lengths, target_lengths, prior_scaling_factor=0.3, is_training=True):
        if targets.dim() == 1:
            segments = targets.split(target_lengths.tolist())
            targets = pad_sequence(
                segments,
                batch_first=True, 
                # padding_value=0,
            ).to(log_probs.device)
        
        log_probs = log_probs.permute(1, 0, 2)  # (T, N, C) -> (N, T, C)
        
        if True and is_training:
            log_probs_flattened = []
            for lp, le in zip(log_probs, input_lengths):
                log_probs_flattened.append(lp[:int(le.item())])      
            log_probs_flattened = torch.cat(log_probs_flattened, 0)  # (T, C)

            # Note, the log_probs here is already log_softmax'ed.
            T = log_probs_flattened.size(0)
            self.priors_T += T
            log_batch_priors_sum = torch.logsumexp(log_probs_flattened, dim=0, keepdim=True)
            log_batch_priors_sum = log_batch_priors_sum.detach()
            log_batch_priors_sum = self.pad_probs(log_batch_priors_sum, self.max_dim, pad_value=-50.0)
            if self.log_priors_sum is None:
                self.log_priors_sum = log_batch_priors_sum
            else:
                _temp = torch.stack([self.log_priors_sum, log_batch_priors_sum], dim=-1)
                self.log_priors_sum = torch.logsumexp(_temp, dim=-1)

        if True and self.log_priors is not None and prior_scaling_factor > 0:
            log_priors = self.log_priors.view(1,1,-1)
            log_priors = log_priors[:, :, :log_probs.size(-1)]
            # print(f"log_probs.shape: {log_probs.shape}")
            # print(f"log_priors.shape: {log_priors.shape}")
            log_probs = log_probs - log_priors * prior_scaling_factor
        
        log_probs = log_probs.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
        input_lengths = input_lengths.to(log_probs.device)
        target_lengths = target_lengths.to(log_probs.device)        
        loss = ctc_primer.ctc_loss(log_probs, targets.to(torch.int64), input_lengths.to(torch.int64), target_lengths.to(torch.int64), blank = 0, reduction = 'none').sum()
        return loss

    def encode_supervisions(self, targets, target_lengths, input_lengths):
        batch_size = len(target_lengths)
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
        res = [targets[i] for i in indices.tolist()]

        return supervision_segments, res, indices

    def forward_k2(self, log_probs, targets, input_lengths, target_lengths, prior_scaling_factor=0.3, is_training=True):
        if targets.dim() == 1:
            targets = targets.split(target_lengths.tolist())

        # Be careful: the targets here are already padded! We need to remove paddings from it
        supervision_segments, new_targets, indices = self.encode_supervisions(targets, target_lengths, input_lengths)
        
        new_targets = [t.tolist() for t in new_targets]
        decoding_graph = k2.ctc_graph(new_targets, modified=False, device=log_probs.device)

        log_probs = log_probs.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)

        if True and is_training:
            log_probs_flattened = []
            for lp, le in zip(log_probs, input_lengths):
                log_probs_flattened.append(lp[:int(le.item())])      
            log_probs_flattened = torch.cat(log_probs_flattened, 0)  # (T, C)

            # Note, the log_probs here is already log_softmax'ed.
            T = log_probs_flattened.size(0)
            self.priors_T += T
            log_batch_priors_sum = torch.logsumexp(log_probs_flattened, dim=0, keepdim=True)
            log_batch_priors_sum = log_batch_priors_sum.detach()
            log_batch_priors_sum = self.pad_probs(log_batch_priors_sum, self.max_dim, pad_value=-50.0)
            if self.log_priors_sum is None:
                self.log_priors_sum = log_batch_priors_sum
            else:
                _temp = torch.stack([self.log_priors_sum, log_batch_priors_sum], dim=-1)
                self.log_priors_sum = torch.logsumexp(_temp, dim=-1)

        if True and self.log_priors is not None and prior_scaling_factor > 0:
            log_priors = self.log_priors.view(1,1,-1)
            log_priors = log_priors[:, :, :log_probs.size(-1)]
            # print(f"log_probs.shape: {log_probs.shape}")
            # print(f"log_priors.shape: {log_priors.shape}")
            log_probs = log_probs - log_priors * prior_scaling_factor

        dense_fsa_vec = k2.DenseFsaVec(
            log_probs,
            supervision_segments,
            allow_truncate=0,
        )

        loss = k2.ctc_loss(
            decoding_graph=decoding_graph,
            dense_fsa_vec=dense_fsa_vec,
            output_beam=10.0,
            reduction="sum",
            use_double_scores=False,
            # delay_penalty=0.05,
        )
        return loss
    
    def forward_k2_simple(self, log_probs, targets, input_lengths, target_lengths, prior_scaling_factor=0.3, is_training=True):
        if targets.dim() == 1:
            targets = targets.split(target_lengths.tolist())

        # Be careful: the targets here are already padded! We need to remove paddings from it
        supervision_segments, new_targets, indices = self.encode_supervisions(targets, target_lengths, input_lengths)
        
        new_targets = [t.tolist() for t in new_targets]
        decoding_graph = k2.ctc_graph(new_targets, modified=False, device=log_probs.device)

        log_probs = log_probs.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)

        # constant penalty
        if False and prior_scaling_factor != 0:
            log_priors = torch.zeros(log_probs.size(-1))
            log_priors[0] += prior_scaling_factor
            log_priors = log_priors.view(1,1,-1).to(log_probs.device)
            # print(f"log_probs.shape: {log_probs.shape}")
            # print(f"log_priors.shape: {log_priors.shape}")
            log_probs = log_probs - log_priors
        
        # dynamic penalty -- the goal here is to expose the second best; make those "significant" second-best better than <no-bias>
        if False and prior_scaling_factor is None:
            second_best_values, _ = log_probs[:,:,1:].max(dim=-1)  # (N, T) and (N, T)
            vals = torch.where((second_best_values > -2.3) & (log_probs[:,:,0] > second_best_values), second_best_values - 1.0, log_probs[:,:,0])
            log_probs[:,:,0] = vals.squeeze(-1)

        dense_fsa_vec = k2.DenseFsaVec(
            log_probs,
            supervision_segments,
            allow_truncate=0,
        )

        if True:
            loss = k2.ctc_loss(
                decoding_graph=decoding_graph,
                dense_fsa_vec=dense_fsa_vec,
                output_beam=10.0,
                reduction="sum",
                use_double_scores=False,
                # delay_penalty=0.05,
            )
        
        if False:  
            # Also checkout: /exp/rhuang/meta/k2/k2/python/k2/ctc_loss.py and https://github.com/k2-fsa/icefall/blob/master/icefall/decode.py
            # /exp/rhuang/meta/icefall/egs/librispeech/ASR/pruned_transducer_stateless7_context_proxy_all_layers/scripts/understand_ctc_gradients.ipynb

            # lattice = k2.intersect_dense(
            #     decoding_graph,
            #     dense_fsa_vec,
            #     10,
            # )

            lattice = k2.intersect_dense_pruned(
                decoding_graph,
                dense_fsa_vec,
                search_beam=20,  # 15
                output_beam=8,  # 6
                min_active_states=30,
                max_active_states=10000,
            )

            best_path = k2.shortest_path(lattice, use_double_scores=False)
            forward_scores = best_path.get_tot_scores(use_double_scores=False, log_semiring=True)

            loss = -forward_scores.sum()

        return loss

    def gather_and_update_priors(self, is_ddp=True):
        if not is_ddp:
            new_log_prior = self.log_priors_sum - torch.log(torch.tensor(self.priors_T))
            prior_threshold = -12.0
            new_log_prior = torch.where(new_log_prior < prior_threshold, prior_threshold, new_log_prior)
            self.log_priors = new_log_prior.squeeze().view(1,1,-1)
            self.log_priors_sum = None
            self.priors_T = 0
            return

        tensor = self.log_priors_sum
        # Initialize the gather list on the destination process
        if dist.get_rank() == 0:
            log_priors_sums = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        else:
            log_priors_sums = None
        dist.gather(tensor, gather_list=log_priors_sums, dst=0)

        tensor = torch.tensor([self.priors_T]).to(tensor.device)
        # Initialize the gather list on the destination process
        if dist.get_rank() == 0:
            priors_Ts = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        else:
            priors_Ts = None
        dist.gather(tensor, gather_list=priors_Ts, dst=0)

        if dist.get_rank() == 0:
            log_priors_sums = torch.stack(log_priors_sums)
            priors_Ts = torch.stack(priors_Ts)
            log_priors_sums = torch.logsumexp(log_priors_sums, dim=0, keepdim=True)  # (1,1,D)
            log_priors_sums = log_priors_sums.squeeze()  # (D,)
            log_priors_Ts = priors_Ts.sum().log().to(log_priors_sums.device)
            new_log_prior = log_priors_sums - log_priors_Ts

            _a1 = log_priors_sums.exp().sum()
            _b1 = priors_Ts.sum()
            # assert abs(_a1 - _b1) / _b1 < 1e-4, f"{_a1} vs. {_b1}"
            if abs(_a1 - _b1) / _b1 > 1e-4:
                logging.error(f"prior prob may have error: {_a1} vs. {_b1}: {abs(_a1 - _b1) / _b1}")
            
            logging.info("new_priors: " + str(["{0:0.2f}".format(i) for i in new_log_prior.exp().tolist()]))
            logging.info("new_log_prior: " + str(["{0:0.2f}".format(i) for i in new_log_prior.tolist()]))
            if self.log_priors is not None:
                _a1 = new_log_prior.exp()
                _b1 = self.log_priors.exp()
                logging.info("diff%: " + str(["{0:0.2f}".format(i) for i in ((_a1 - _b1)/_b1*100).tolist()]))
              
            prior_threshold = -12.0
            new_log_prior = torch.where(new_log_prior < prior_threshold, prior_threshold, new_log_prior)
            new_log_prior = new_log_prior.squeeze()
            logging.info("new_log_prior (clipped): " + str(["{0:0.2f}".format(i) for i in new_log_prior.tolist()]))
        else:
            # create an empty tensor of the same shape as self.log_priors_sum
            new_log_prior = torch.zeros_like(self.log_priors_sum)
            new_log_prior = new_log_prior.squeeze()
        
        dist.broadcast(new_log_prior, src=0)
        # logging.info(f"[{dist.get_rank()}] new_log_prior={new_log_prior}")
        self.log_priors = new_log_prior.squeeze()
        # print(f"self.log_priors_sum.shape: {self.log_priors_sum.shape}")
        # print(f"new_log_prior.shape: {new_log_prior.shape}")
        self.log_priors_sum = None
        self.priors_T = 0
            
    def compute_cross_entropy_loss(self, probs, x_lens, num_words_per_utt, gt_rare_words_indices):
        # `probs` of shape (N, T, D)
        # `x_lens` corresponds to the T dimension
        # `num_words_per_utt` corresponds to the D dimension

        assert probs.size(0) == len(gt_rare_words_indices) == len(x_lens) == len(num_words_per_utt)
        probs_list = [ps[:xl, max(gt_idx, default=0) + 1: nwpu].sum(dim=-1) for ps, xl, nwpu, gt_idx in zip(probs, x_lens, num_words_per_utt, gt_rare_words_indices)]

        all_neg_class_probs = torch.cat(probs_list)  # these probabilities should be close to 0

        # logging.info(f"ratio (>0.2): {torch.sum(all_neg_class_probs > 0.2) / all_neg_class_probs.size(0)}")        
        # torch.sum(all_neg_class_probs > 0.2) / all_neg_class_probs.size(0)

        # targets = torch.zeros_like(all_neg_class_probs)
        targets = torch.full_like(all_neg_class_probs, 1e-10)
        weights = 1 + 2 / (1 + torch.exp(3.5 - 10 * all_neg_class_probs.detach()))
        weights = torch.where(weights > 2.9, 5, weights)
        # weights = None
        
        with torch.cuda.amp.autocast(enabled=False):
            loss = F.binary_cross_entropy(all_neg_class_probs, targets, weight=weights, reduction="sum")

        # with torch.cuda.amp.autocast(enabled=False):
        #     loss_fn = nn.BCELoss(weight=weights, reduction="sum")
        #     loss = loss_fn(all_neg_class_probs, targets)

        # Another option is NLLLOSS?

        return loss

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        contexts: dict,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
    ) -> torch.Tensor:
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
          word_list: 
            A list of words, where each word is a list of token ids.
            The list of tokens for each word has been padded.
          word_lengths:
            The number of tokens per word
          num_words_per_utt:
            The number of words in the context for each utterance
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
          Return the transducer loss.

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0
        # assert x.size(0) == x_lens.size(0) == y.dim0 or x.size(0) * 2 == x_lens.size(0) * 2 == y.dim0

        contexts_h, contexts_mask = self.context_encoder.embed_contexts(
            contexts
        )

        # if x.size(0) * 2 == y.dim0:
        #     x = torch.cat((x, x), dim=0)
        #     x_lens = torch.cat((x_lens, x_lens), dim=-1)

        if self.training and self.use_ctc3:
            self.encoder.need_attn_weights = True
        encoder_out, x_lens, intermediate_out = self.encoder(x, x_lens, contexts=(contexts_h, contexts_mask, self.encoder_biasing_adapter))
        assert torch.all(x_lens > 0)
        self.encoder.need_attn_weights = False

        ctc_loss = torch.tensor(0).to(encoder_out.device)

        if self.use_ctc and self.training:
            # Compute CTC loss
            targets = y.values

            row_splits = y.shape.row_splits(1)
            y_lens = row_splits[1:] - row_splits[:-1]

            ctc_output3 = self.ctc_output3(intermediate_out[3])  # (T, N, C)
            # ctc_output5 = self.ctc_output5(intermediate_out[5])  # (T, N, C)

            ctc_loss3 = F.ctc_loss(
                log_probs=ctc_output3,
                targets=targets,
                input_lengths=x_lens,
                target_lengths=y_lens,
                reduction="sum",
            )
            # ctc_loss5 = F.ctc_loss(
            #     log_probs=ctc_output5,
            #     targets=targets,
            #     input_lengths=x_lens,
            #     target_lengths=y_lens,
            #     reduction="sum",
            # )
            # ctc_loss = (ctc_loss3 + ctc_loss5)/2
            ctc_loss = ctc_loss3            

        # assert x.size(0) == contexts_h.size(0) == contexts_mask.size(0)
        # assert contexts_h.ndim == 3
        # assert contexts_h.ndim == 2
        if self.params.irrelevance_learning and self.training:
            need_weights = True
        else:
            # need_weights = False
            need_weights = True if self.training and self.use_ctc3 else False
        encoder_biasing_out, attn_enc = self.encoder_biasing_adapter[-1].forward(encoder_out, contexts_h, contexts_mask, need_weights=need_weights)
        # breakpoint()
        encoder_out = encoder_out + encoder_biasing_out

        # Apply ctc loss on the `encoder_biasing_out` term only
        if self.use_ctc2 and self.training:
            assert not self.use_ctc
            y_rare = contexts['y_rare']
            targets = y_rare.values

            row_splits = y_rare.shape.row_splits(1)
            y_lens = row_splits[1:] - row_splits[:-1]

            ctc_loss = torch.tensor(0).to(encoder_out.device)

            for i in range(len(self.encoder.encoder_dims)):
                if self.ctc2_outputs[i] is not None:
                    assert intermediate_out[i] is not None
                    ctc2_output = self.ctc2_outputs[i](intermediate_out[i])  # (T,N,C)
                    ctc_loss = ctc_loss + F.ctc_loss(
                        log_probs=ctc2_output,
                        targets=targets,
                        input_lengths=x_lens,
                        target_lengths=y_lens,
                        reduction="sum",
                    )

            i = len(self.encoder.encoder_dims)
            if self.ctc2_outputs[i] is not None and i in self.i_ctc2_layers:
                ctc2_output = self.ctc2_outputs[i](encoder_biasing_out.permute(1, 0, 2))  # (T,N,C)
                ctc_loss = ctc_loss + F.ctc_loss(
                    log_probs=ctc2_output,
                    targets=targets,
                    input_lengths=x_lens,
                    target_lengths=y_lens,
                    reduction="sum",
                )
            
            ctc_loss = ctc_loss / len(self.i_ctc2_layers)
        
        # Apply ctc loss on the attention weights
        if self.use_ctc3 and self.training:
            assert not self.use_ctc and not self.use_ctc2
            gt_rare_words_indices = contexts['gt_rare_words_indices']
            y_lens = [len(wl) for wl in gt_rare_words_indices]
            y_lens = torch.tensor(y_lens, dtype=torch.int64)
            gt_rare_words_indices = [w for wl in gt_rare_words_indices for w in wl]
            gt_rare_words_indices = torch.tensor(gt_rare_words_indices, dtype=torch.int64)
            gt_rare_words_indices += 1  # shift 1 due to the no-bias token

            ctc_loss = torch.tensor(0).to(encoder_out.device)
            for i in range(len(self.encoder.encoder_dims)):
                if i in self.i_ctc3_layers:
                    assert intermediate_out[i] is not None
                    # _ctc_loss = F.ctc_loss(
                    #     log_probs=(intermediate_out[i]+1.0e-20).log().permute(1, 0, 2),  # (T,N,C)
                    #     targets=gt_rare_words_indices,
                    #     input_lengths=self.encoder.intermediate_x_lens[i],
                    #     target_lengths=y_lens,
                    #     reduction="sum",
                    # )
                    _ctc_loss = self.forward_k2_simple(
                        log_probs=(intermediate_out[i] + 1.0e-20).log().permute(1, 0, 2),  # (T,N,C)
                        targets=gt_rare_words_indices, 
                        input_lengths=self.encoder.intermediate_x_lens[i], 
                        target_lengths=y_lens,
                        prior_scaling_factor=None,
                        is_training=True,
                    )
                    # breakpoint()
                    # !ii=6; attn_enc[ii].argmax(dim=-1), contexts['gt_rare_words_indices'][ii]
                    # !vals, ids = attn_enc[ii].max(-1)
                    # !for j,  (i, v) in enumerate(zip(ids.tolist(), vals.tolist())):  print(j, i, v)
                    #
                    # !vals, ids = attn_enc[ii].topk(3, dim=-1)
                    # !for j in range(len(vals)): print(f"[{j}] Indices: {ids[j].tolist()}, Values: {vals[j].tolist()}")
                    # print(f"CTC loss for layer {i} is {_ctc_loss}")
                    ctc_loss = ctc_loss + _ctc_loss
            
            i = len(self.encoder.encoder_dims)
            if i in self.i_ctc3_layers:
                # _ctc_loss = F.ctc_loss(
                #     log_probs=(attn_enc+1.0e-20).log().permute(1, 0, 2),  # (T,N,C)
                #     targets=gt_rare_words_indices,
                #     input_lengths=x_lens,
                #     target_lengths=y_lens,
                #     reduction="sum",
                # )
                _ctc_loss = self.forward_k2_simple(
                    log_probs=(attn_enc + 1.0e-20).log().permute(1, 0, 2),  # (T,N,C)
                    targets=gt_rare_words_indices,
                    input_lengths=x_lens,
                    target_lengths=y_lens,
                    prior_scaling_factor=None,
                    is_training=True,
                )
                # print(f"CTC loss for layer {i} is {_ctc_loss}")
                ctc_loss = ctc_loss + _ctc_loss

            ctc_loss = ctc_loss / len(self.i_ctc3_layers)
        
        if self.ce_loss and self.training:
            assert not self.use_ctc and not self.use_ctc2 and not self.use_ctc3
            
            ctc_loss = torch.tensor(0).to(encoder_out.device)
            for i in range(len(self.encoder.encoder_dims)):
                if i in self.ce_layers:
                    assert intermediate_out[i] is not None
                    _ctc_loss = self.compute_cross_entropy_loss(
                        intermediate_out[i], 
                        self.encoder.intermediate_x_lens[i], 
                        contexts["num_words_per_utt"], 
                        contexts["gt_rare_words_indices"]
                    )
                    ctc_loss = ctc_loss + _ctc_loss
            
            _ctc_loss = self.compute_cross_entropy_loss(attn_enc, x_lens, contexts["num_words_per_utt"], contexts["gt_rare_words_indices"])
            ctc_loss = ctc_loss + _ctc_loss

        # Now for the decoder, i.e., the prediction network
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

        # decoder_out: [B, S + 1, decoder_dim]
        decoder_out = self.decoder(sos_y_padded)

        if self.context_encoder.bi_encoders:
            contexts_dec_h, contexts_dec_mask = self.context_encoder.embed_contexts(
                contexts,
                is_encoder_side=False,
            )
        else:
            contexts_dec_h, contexts_dec_mask = contexts_h, contexts_mask

        decoder_biasing_out, attn_dec = self.decoder_biasing_adapter.forward(decoder_out, contexts_dec_h, contexts_dec_mask, need_weights=need_weights)
        decoder_out = decoder_out + decoder_biasing_out

        # Note: y does not start with SOS
        # y_padded : [B, S]
        y_padded = y.pad(mode="constant", padding_value=0)

        y_padded = y_padded.to(torch.int64)
        boundary = torch.zeros((x.size(0), 4), dtype=torch.int64, device=x.device)
        boundary[:, 2] = y_lens
        boundary[:, 3] = x_lens

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

        return (simple_loss, pruned_loss, ctc_loss)
