import logging
import random
import torch
import k2

from collections import defaultdict

from icefall.utils import AttributeDict
from icefall.decode import get_lattice as _get_lattice
from icefall.decode import one_best_decoding
from icefall.utils import get_alignments, get_texts, get_texts_with_timestamp

import kaldialign

from factor_transducer import *


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

    return AttributeDict({
        "cut_wers": cut_wers,
        "tot_err_rate": tot_err_rate,
        "tot_errs": tot_errs,
        "ref_len": ref_len,
        "ins_errs": ins_errs,
        "del_errs": del_errs,
        "sub_errs": sub_errs,
        "words": words,
        "tot_wer_str": f"{tot_err_rate} [{tot_errs}/{ref_len}] [ins:{ins_errs}, del:{del_errs}, sub:{sub_errs}]",
    })



def get_shorter_texts(_texts, _batch_idx):
    # _i = int(_batch_idx / 2)
    if min([len(t.split()) for t in _texts]) < 15:
        return _texts
    
    _i = min(_batch_idx, min(len(t) for t in _texts) - 3)
    print(f"_batch_idx = {_batch_idx}, _i={_i}")
    _texts = [t.split() for t in _texts]
    if _batch_idx % 2 == 0:
        _texts = [" ".join(t[_i:]) for t in _texts]
    else:
        _texts = [" ".join(t[:-_i]) for t in _texts]
    return _texts


def modify_texts(_texts, **kwargs):
    if min([len(t.split()) for t in _texts]) > 15:
        texts = [" ".join(t.split()[5:-5]) for t in texts]
    # texts = get_shorter_texts(_texts, _batch_idx = kwargs["batch_idx_train"])
    return texts


def get_random_texts(_texts):
    texts_shuffled = random.sample(_texts, len(_texts))
    return _texts[:-3] + texts_shuffled[-3:]


def get_lattice(params, ctc_output, batch, sp, decoding_graph=None):
    # Option 1:
    if decoding_graph is None:
        H = k2.ctc_topo(
            max_token=params.vocab_size - 1,
            modified=False,
            device=ctc_output.device,
        )
        decoding_graph = H

    # Option 2:    
    # if decoding_graph is None:
    #     texts = batch["supervisions"]["text"]
    #     make_factor_transducer1 = params.my_args["make_factor_transducer1"]
    #     decoding_graph = [make_factor_transducer1(sp.encode(text, out_type=int), return_str=False, blank_penalty=0) for text in texts]

    supervisions = batch["supervisions"]
    supervision_segments = torch.stack(
        (
            supervisions["sequence_idx"],
            torch.div(
                supervisions["start_frame"],
                params.subsampling_factor,
                rounding_mode="floor",
            ),
            torch.div(
                supervisions["num_frames"],
                params.subsampling_factor,
                rounding_mode="floor",
            ),
        ),
        1,
    ).to(torch.int32)

    indices = torch.argsort(supervision_segments[:, 2], descending=True)
    supervision_segments = supervision_segments[indices]

    if isinstance(decoding_graph, list):
        decoding_graph = [decoding_graph[i] for i in indices.tolist()]
        decoding_graph = k2.create_fsa_vec(decoding_graph)
        decoding_graph = k2.arc_sort(decoding_graph)
        decoding_graph = decoding_graph.to(ctc_output.device)

    lattice = _get_lattice(
        nnet_output=ctc_output,
        decoding_graph=decoding_graph,
        supervision_segments=supervision_segments,
        # search_beam=15,
        # output_beam=6,
        search_beam=20,
        output_beam=5,
        min_active_states=30,
        max_active_states=10000,
        subsampling_factor=params.subsampling_factor,
    )
    
    return lattice, indices


def get_lattice_and_best_paths(params, ctc_output, batch, sp, decoding_graph=None):
    lattice, indices = get_lattice(params, ctc_output, batch, sp, decoding_graph)

    best_paths = one_best_decoding(
        lattice=lattice,
        use_double_scores=True,
    )

    _indices = {i_old : i_new for i_new, i_old in enumerate(indices.tolist())}
    best_paths = [best_paths[_indices[i]] for i in range(len(_indices))]
    best_paths = k2.create_fsa_vec(best_paths)

    lattice = [lattice[_indices[i]] for i in range(len(_indices))]
    lattice = k2.create_fsa_vec(lattice)
    
    # This `lattice` and `best_paths` are in the same order as the original batch
    return lattice, best_paths


def get_best_paths(params, ctc_output, batch, sp, decoding_graph=None):
    lattice, best_paths = get_lattice_and_best_paths(params, ctc_output, batch, sp, decoding_graph=decoding_graph)
    
    return best_paths


def get_batch_wer(params, ctc_output, batch, sp, decoding_graph=None, best_paths=None, hyps=None):
    if best_paths is None and hyps is None:
        ctc_output = ctc_output.detach()
        best_paths = get_best_paths(params, ctc_output, batch, sp, decoding_graph)

    texts = batch["supervisions"]["text"]
    cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

    if hyps is None:
        token_ids = get_texts(best_paths)
        hyps_lens = [len(t) for t in token_ids]
        # hyps = [" ".join([word_table[i] for i in ids]) for ids in token_ids]
        hyps = sp.decode(token_ids)
        hyp_ref_lens = [(len(hyp_token_ids), len(sp.encode(ref_text, out_type=int))) for hyp_token_ids, ref_text in zip(token_ids, texts)]
    else:
        hyps_lens = None
        hyp_ref_lens = None

    results = [(cut_id, ref_text.split(), hyp_text.split()) for cut_id, hyp_text, ref_text in zip(cut_ids, hyps, texts)]

    wer = compute_wer(results)
    wer["hyps_lens"] = hyps_lens
    wer["hyp_ref_lens"] = hyp_ref_lens
    wer["alignment_results"] = results

    return wer


def compute_sub_factor_transducer_loss1(params, ctc_output, lattice, best_paths, batch, sp):
    # use decoding results text as ground truth

    lattice = lattice.detach().to('cpu')
    best_paths = best_paths.detach().to('cpu')

    # _indices = {i_old : i_new for i_new, i_old in enumerate(indices.tolist())}
    # best_paths = [best_paths[_indices[i]] for i in range(len(_indices))]
    # best_paths = k2.create_fsa_vec(best_paths)

    # TODO: we can get aligment time stamps here
    # decoding_results = get_texts_with_timestamp(best_path)
    # decoding_results.timestamps
    _token_ids = get_texts(best_paths)

    assert "libri_long_text_str" in params.my_args and params.my_args["libri_long_text_str"] is not None

    # That means _token_ids are actually indices
    libri_long_text_str = params.my_args["libri_long_text_str"]
    cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]
    _texts = [libri_long_text_str[tuple(get_uid_key(cid)[:2])][max(rg[0]-1, 0): rg[-1]-1] for cid, rg in zip(cut_ids, _token_ids)]
    _texts = [" ".join(t) for t in _texts]
    token_ids = sp.encode(_texts, out_type=int)

    # breakpoint()
    # !import code; code.interact(local=vars())
    
    # wer = get_batch_wer(params, ctc_output, batch, sp, decoding_graph=None, best_paths=best_paths)
    batch_wer = get_batch_wer(params, ctc_output, batch, sp, decoding_graph=None, best_paths=None, hyps=_texts)
    logging.info(f"batch_wer [{params.batch_idx_train}]: {batch_wer['tot_wer_str']}")

    new_decoding_graph = k2.ctc_graph(token_ids, modified=False, device=ctc_output.device)
    new_decoding_graph = k2.arc_sort(new_decoding_graph)
    new_decoding_graph = [new_decoding_graph[i] for i in range(new_decoding_graph.shape[0])]

    new_lattice, new_best_paths = get_lattice_and_best_paths(params, ctc_output, batch, sp, decoding_graph=new_decoding_graph)
    return new_lattice, new_best_paths


def compute_sub_factor_transducer_loss2(params, ctc_output, lattice, best_paths, batch, sp):
    # use decoding results text to build extended decoding graph

    lattice = lattice.detach().to('cpu')
    best_paths = best_paths.detach().to('cpu')

    # _indices = {i_old : i_new for i_new, i_old in enumerate(indices.tolist())}
    # best_paths = [best_paths[_indices[i]] for i in range(len(_indices))]
    # best_paths = k2.create_fsa_vec(best_paths)

    # TODO: we can get aligment time stamps here
    # decoding_results = get_texts_with_timestamp(best_path)
    # decoding_results.timestamps
    token_ids_indices = get_texts(best_paths)

    libri_long_text_str = params.my_args["libri_long_text_str"]
    cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

    new_decoding_graph = make_factor_transducer5(libri_long_text_str, cut_ids, token_ids_indices, sp, extension=10, two_ends_bonus=1.0)

    new_lattice, new_best_paths = get_lattice_and_best_paths(params, ctc_output, batch, sp, decoding_graph=new_decoding_graph)
    return new_lattice, new_best_paths


def compute_ctc_loss_long(params, ctc_output, batch, sp, decoding_graph=None):
    lattice, best_paths = get_lattice_and_best_paths(params, ctc_output, batch, sp, decoding_graph)

    # breakpoint()

    # This only works with `make_factor_transducer4`:
    lattice, best_paths = compute_sub_factor_transducer_loss1(params, ctc_output, lattice, best_paths, batch, sp)
    # lattice, best_paths = compute_sub_factor_transducer_loss2(params, ctc_output, lattice, best_paths, batch, sp)

    # Note: interesting:
    # (1) Needs to use `make_factor_transducer4_skip`
    # (2) Needs to use `compute_sub_factor_transducer_loss1` or `compute_sub_factor_transducer_loss2`

    # breakpoint()
    # best_paths[0].shape, best_paths[0].num_arcs

    # scoring_fst = lattice
    scoring_fst = best_paths

    loss = scoring_fst.get_tot_scores(log_semiring=True, use_double_scores=True)
    loss = -1 * loss
    loss = loss.to(torch.float32)

    mask_tt = []
    inf_indices = torch.where(torch.isinf(loss))[0].cpu().tolist()
    if any(mask_tt):
        mask_tt_indices = torch.nonzero(torch.tensor(mask_tt)).squeeze()
        inf_indices = inf_indices.extend(mask_tt_indices.tolist())

    inf_indices = set(inf_indices)
    if len(inf_indices) > 0:
        # ctc_loss = 0
        # ignore_idx = set(inf_indices.tolist())
        # for i in range(len(loss)):
        #     if i not in ignore_idx:
        #         ctc_loss = ctc_loss + loss[i]

        non_inf_indices = [i for i in range(len(loss)) if i not in inf_indices]
        ctc_loss = torch.sum(loss[non_inf_indices])
        
        # mask = torch.ones_like(loss)
        # mask[inf_indices] = 0
        # loss = loss * mask
        # loss[inf_indices].detach()
        # _indices = {i_new : i_old for i_new, i_old in enumerate(indices.tolist())}
        # inf_indices_old = [_indices[i] for i in ignore_idx]  # This are the indices of the inf/ignored utterances in the original batch
        
        inf_indices = list(inf_indices)
        cut_ids = [batch["supervisions"]["cut"][i].id for i in inf_indices]
        logging.warning(f"Found {len(inf_indices)} inf/nan/ignored values in loss for batch_idx_train={params.batch_idx_train}: {cut_ids}")
    else:
        ctc_loss = loss.sum()
        inf_indices = []
    
    return ctc_loss, inf_indices


def get_next_anchor_point(params, ctc_output, batch, sp, decoding_graph=None):
    lattice, indices = get_lattice(params, ctc_output, batch, sp, decoding_graph)

    best_paths = one_best_decoding(
        lattice=lattice,
        use_double_scores=True,
    )

    lattice = lattice.detach()
    best_paths = best_paths.detach()

    _indices = {i_old : i_new for i_new, i_old in enumerate(indices.tolist())}
    best_paths = [best_paths[_indices[i]] for i in range(len(_indices))]
    best_paths = k2.create_fsa_vec(best_paths)

    # TODO: we can get aligment time stamps here
    decoding_results = get_texts_with_timestamp(best_paths)
    timestamps = decoding_results.timestamps
    token_id_indices = decoding_results.hyps

    breakpoint()
    pass




    


