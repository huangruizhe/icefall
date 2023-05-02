from egs.librispeech.ASR.pruned_transducer_stateless7_context.context_collector import ContextCollector
from icefall import BiasedNgramLm, BiasedNgramLmStateBonus
# export PYTHONPATH=/export/fs04/a12/rhuang/icefall_align2/:$PYTHONPATH
# export PYTHONPATH=/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/pruned_transducer_stateless7_context:$PYTHONPATH

import logging
import argparse
import sentencepiece as spm
from pathlib import Path
import k2


logging.basicConfig(
    format = "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
    level = 10
)

def parse_opts():
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--context-dir",
        type=str,
        default="data/fbai-speech/is21_deep_bias/",
        help="",
    )

    parser.add_argument(
        "--lang-dir",
        type=Path,
        default="data/lang_bpe_500",
        help="The lang dir containing word table and LG graph",
    )

    opts = parser.parse_args()
    logging.info(f"Parameters: {vars(opts)}")
    return opts


def main(params):
    logging.info("About to load context generator")
    params.context_dir = Path(params.context_dir)
    params.lang_dir = Path(params.lang_dir)

    sp = spm.SentencePieceProcessor()
    sp.load(str(params.lang_dir / "bpe.model"))

    context_collector = ContextCollector(
        path_is21_deep_bias=params.context_dir,
        sp=sp,
        is_predefined=False,
        n_distractors=0,
        keep_ratio=1.0,
        is_full_context=False,
    )

    from collections import namedtuple
    cut = namedtuple('Cut', ['supervisions'])
    supervision = namedtuple('Supervision', ['id'])
    
    uid = "1995-1836-0000"
    text = "THE HON CHARLES SMITH MISS"
    
    supervision.id = uid
    cut.supervisions = [supervision]
    batch = {"supervisions": {"cut": [cut], "text": [text]}}
    fsa_list, fsa_sizes, num_words_per_utt2 = \
        context_collector.get_context_word_wfst(batch)
    
    logging.info(f"{uid}: {fsa_sizes}")
    text_words = set(text.split())
    logging.info(f"{uid} rare words: {[w for w in text_words if w in context_collector.rare_words]}")

    biased_lm = BiasedNgramLm(
        fst=fsa_list[0], 
        backoff_id=500,
    )
    state_bonus = BiasedNgramLmStateBonus(biased_lm)

    seq = text
    seq = sp.encode_as_ids(seq)
    for token in seq:
        new_state_bonus = state_bonus.forward_one_step(token)
        cur_score = new_state_bonus.lm_score - state_bonus.lm_score
        print(f"{sp.id_to_piece(token)}: {cur_score}")
        state_bonus = new_state_bonus
    
    logging.info(state_bonus.lm_score)



if __name__ == '__main__':
    opts = parse_opts()

    main(opts)
