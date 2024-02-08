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
        search_beam=15,
        output_beam=6,
        min_active_states=30,
        max_active_states=10000,
        subsampling_factor=4,
    )
    
    return lattice, indices

def get_best_path(params, ctc_output, batch, sp, decoding_graph=None):
    lattice, indices = get_lattice(params, ctc_output, batch, sp, decoding_graph)

    best_paths = one_best_decoding(
        lattice=lattice,
        use_double_scores=True,
    )

    _indices = {i_old : i_new for i_new, i_old in enumerate(indices.tolist())}
    best_paths = [best_paths[_indices[i]] for i in range(len(_indices))]
    best_paths = k2.create_fsa_vec(best_paths)
    
    return best_paths

def get_batch_wer(params, ctc_output, batch, sp, decoding_graph=None):
    ctc_output = ctc_output.detach()
    best_paths = get_best_path(params, ctc_output, batch, sp, decoding_graph)

    breakpoint()

    token_ids = get_texts(best_paths)
    hyps_lens = [len(t) for t in token_ids]
    # hyps = [" ".join([word_table[i] for i in ids]) for ids in token_ids]
    hyps = sp.decode(token_ids)

    texts = batch["supervisions"]["text"]
    cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]
    results = [(cut_id, ref_text.split(), hyp_text.split()) for cut_id, hyp_text, ref_text in zip(cut_ids, hyps, texts)]
    hyp_ref_lens = [(len(hyp_token_ids), len(sp.encode(ref_text, out_type=int))) for hyp_token_ids, ref_text in zip(token_ids, texts)]

    wer = compute_wer(results)
    wer["hyps_lens"] = hyps_lens
    wer["hyp_ref_lens"] = hyp_ref_lens

    return wer
