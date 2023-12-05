import logging
import argparse
from pathlib import Path
from collections import namedtuple
import ast
import sentencepiece as spm
from tqdm import tqdm

from egs.spgispeech.ASR.pruned_transducer_stateless7_context.decode_uniphore import rare_word_score
from egs.spgispeech.ASR.pruned_transducer_stateless7_context.context_collector import ContextCollector
from egs.spgispeech.ASR.pruned_transducer_stateless7_context.score import main as score_main

import sys
sys.path.insert(0,'/exp/rhuang/meta/audio_ruizhe/ec21/')
from whisper_normalizer_nlp import icefall_spgi_specific_normalizer
from whisper_normalizer.english import EnglishTextNormalizer


logging.basicConfig(
    format = "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
    level = logging.CRITICAL
)

def parse_opts():
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--per-utt', type=str, default=None, help='')
    parser.add_argument('--context-dir', type=str, default=None, help='')
    parser.add_argument('--lang-dir', type=str, default=None, help='')
    parser.add_argument('--phrase-mappings', type=str, default=None, help='')

    opts = parser.parse_args()
    logging.info(f"Parameters: {vars(opts)}")
    return opts


def parse_per_utt(filename):
    if filename is None:
        return None
    
    with open(filename, "r") as fin:
        lines = fin.readlines()
    
    per_utt = dict()
    for i in range(0, len(lines), 2):
        ref = lines[i].strip()
        hyp = lines[i + 1].strip()

        ref = ref.split(maxsplit=1)
        uid = ref[0][:-1]
        ref = ref[1][4:]
        ref = ast.literal_eval(ref)

        # if not uid.startswith("4320211"): continue

        hyp = hyp.split(maxsplit=1)
        assert hyp[0][:-1] == uid, f"{hyp[0][:-1]} != {uid}"
        hyp = hyp[1][4:]
        hyp = ast.literal_eval(hyp)

        per_utt[uid] = (ref, hyp)
    
    return per_utt


def normalize(text, english_normalizer, phrase_mappings):
    text_norm = english_normalizer(text)
    text_norm = icefall_spgi_specific_normalizer(text_norm)
    text_norm = " ".join(w if w not in phrase_mappings else phrase_mappings[w] for w in text_norm.split())
    return text_norm


def get_key(my_id):
    my_id = my_id.split("_")
    return my_id[0], int(my_id[1])


def main(opts):
    per_utt = parse_per_utt(opts.per_utt)

    if opts.phrase_mappings is not None:
        with open(opts.phrase_mappings) as fin:
            phrase_mappings = [l.strip().split("|") for l in fin]
            phrase_mappings = {k: v for (k, v) in phrase_mappings}
    else:
        phrase_mappings = dict()

    english_normalizer = EnglishTextNormalizer()

    sp = spm.SentencePieceProcessor()
    sp.load(str(Path(opts.lang_dir) / "bpe.model"))

    context_collector = ContextCollector(
        path_rare_words=Path(opts.context_dir),
        sp=sp,
        is_predefined=False,
        n_distractors=0,
        keep_ratio=1.0,
        is_full_context=False,
    )

    args = namedtuple('A', ['refs', 'hyps', 'lenient'])
    args.lenient = True

    refs = {}
    hyps = {}
    print_n_samples = 10
    for uid in tqdm(per_utt):
        ref = " ".join(per_utt[uid][0]).lower()
        hyp = " ".join(per_utt[uid][1]).lower()
        ref = normalize(ref, english_normalizer, phrase_mappings)
        hyp = normalize(hyp, english_normalizer, phrase_mappings)
        batch = {"supervisions": {"text": [ref]}}
        biasing_words = context_collector._get_random_word_lists(batch, no_distractors=True)
        refs[uid] = {"text": ref, "biasing_words": set(biasing_words[0])}
        hyps[uid] = hyp
    args.refs = refs
    args.hyps = hyps

    # sorted_per_utt = sorted(per_utt.items(), key=lambda x: get_key(x[0]))
    # with open("/exp/rhuang/meta/audio_ruizhe/ec21/data/earnings21_updated4/wordspace/4320211.temp.txt", "w") as fout:
    #     for uid, (ref, hyp) in sorted_per_utt:
    #         ref = " ".join(per_utt[uid][0]).lower()
    #         ref = normalize(ref, english_normalizer, phrase_mappings)
    #         ref = ref.split()
    #         print(*ref, sep="\n", file=fout)

    error_details = score_main(args)

    # print(error_details)


if __name__ == '__main__':
    opts = parse_opts()

    main(opts)


