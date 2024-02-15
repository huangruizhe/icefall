import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import k2
import lhotse
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


from icefall.utils import (
    AttributeDict,
)

from factor_transducer import *
from no_seg_utils import *




def decode_dataset_oracle_main(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    sp: Optional[spm.SentencePieceProcessor],
    cuts: lhotse.CutSet,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    
    # get long text for each recording
    libri_long_text_str, libri_long_text_fst = get_long_text(cuts, sp=sp, make_fst=True, rank=0, nj=16 if params.full_libri else 6)
    libri_long_text_str = {k: v.split() for k, v in libri_long_text_str.items()}
    logging.info(f"len(libri_long_text_fst) = {len(libri_long_text_fst)}")
    my_args = {"libri_long_text_fst": libri_long_text_fst}
    my_args |= {"libri_long_text_str": libri_long_text_str}  # Only for make_factor_transducer4
    my_args |= {"long_ctc": True}
    my_args |= {
        "make_factor_transducer1": make_factor_transducer1,
        "make_factor_transducer2": make_factor_transducer2,
        "make_factor_transducer3": make_factor_transducer3,
        "make_factor_transducer4": make_factor_transducer4,
    }
    params.my_args = my_args

    logging.info("Start decoding oracle: train mode")
    rs_train = decode_dataset_oracle(dl, params, model, sp, cuts, is_training=True)

    logging.info("Start decoding oracle: eval mode")
    rs_eval = decode_dataset_oracle(dl, params, model, sp, cuts, is_training=False)
    
    return rs_train, rs_eval


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    sp: Optional[spm.SentencePieceProcessor],
    batch: dict,
    is_training: bool,
) -> Dict[str, List[List[str]]]:

    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    
    feature = batch["inputs"]
    assert feature.ndim == 3
    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]

    ctc_output, memory, memory_key_padding_mask = model(feature, supervisions)
    # nnet_output is (N, T, C)

    texts = batch["supervisions"]["text"]
    y = sp.encode(texts, out_type=int)
    y = k2.RaggedTensor(y)

    row_splits = y.shape.row_splits(1)
    target_lengths = row_splits[1:] - row_splits[:-1]
    targets = y.values   # on CPU

    cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

    libri_long_text_fst = params.my_args["libri_long_text_fst"]
    y_long = [libri_long_text_fst[tuple(get_uid_key(cid)[:2])] for cid in cut_ids]
    # y_long = get_decoding_graphs(texts)
    # y_long = [make_factor_transducer1(sp.encode(text, out_type=int), return_str=False, blank_penalty=0) for text in texts]
    # y_long = None

    if False:
        batch_wer = get_batch_wer(params, ctc_output, batch, sp, decoding_graph=y_long)
    else:  # For make_factor_transducer4
        lattice, best_paths = get_lattice_and_best_paths(params, ctc_output, batch, sp, decoding_graph=y_long)
        lattice = lattice.detach().to('cpu')
        best_paths = best_paths.detach().to('cpu')

        token_ids_indices = get_texts(best_paths)
        token_ids_indices, _ = handle_emtpy_texts(token_ids_indices)
        libri_long_text_str = params.my_args["libri_long_text_str"]
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]
        hyps = [libri_long_text_str[tuple(get_uid_key(cid)[:2])][max(rg[0]-1, 0): rg[-1]-1] for cid, rg in zip(cut_ids, token_ids_indices)]
        batch_wer = {"alignment_results": [(cut_id, r.split(), h) for cut_id, r, h in zip(cut_ids, texts, hyps)]}

    # hyps = [h for cut_id, r, h in batch_wer["alignment_results"]]
    rs = batch_wer["alignment_results"]

    key = f"decoding-with-long-text-{'train' if is_training else 'eval'}"
    return {key: rs}


def decode_dataset_oracle(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    sp: Optional[spm.SentencePieceProcessor],
    cuts: lhotse.CutSet,
    is_training: bool,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    """Decode dataset.

    Args:
      
    Returns:
      Return a dict, whose key may be "no-rescore" if no LM rescoring
      is used, or it may be "lm_scale_0.7" if LM rescoring is used.
      Its value is a list of tuples. Each tuple contains two elements:
      The first is the reference transcript, and the second is the
      predicted result.
    """

    if is_training:
        model.train()
    else:
        model.eval()

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    num_cuts = 0

    results = defaultdict(list)
    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]
        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

        hyps_dict = decode_one_batch(
            params=params,
            model=model,
            sp=sp,
            batch=batch,
            is_training=is_training,
        )

        assert len(hyps_dict) == 1
        k = list(hyps_dict.keys())[0]
        results[k].extend(hyps_dict[k])

        num_cuts += len(texts)

        if batch_idx % 100 == 0:
            batch_str = f"{batch_idx}/{num_batches}"
            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
        
        # logging.info(f"batch_idx={batch_idx}")
        # if batch_idx > 200:
        #     break
    return results