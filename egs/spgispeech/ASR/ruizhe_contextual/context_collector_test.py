from egs.spgispeech.ASR.pruned_transducer_stateless2_context.context_collector import ContextCollector
# export PYTHONPATH=/export/fs04/a12/rhuang/icefall_align2/:$PYTHONPATH
# export PYTHONPATH=/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/pruned_transducer_stateless7_context:$PYTHONPATH

import logging
import argparse
from pathlib import Path
import sentencepiece as spm
from itertools import chain
import ast

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
        default="data/rare_words",
        help="",
    )

    parser.add_argument(
        "--lang-dir",
        type=Path,
        default="data/lang_bpe_500",
        help="The lang dir containing word table and LG graph",
    )

    parser.add_argument(
        "--cuts-file-name",
        type=str,
        default=None,
        help="",
    )

    opts = parser.parse_args()
    logging.info(f"Parameters: {vars(opts)}")
    return opts


def read_ref_biasing_list(filename):
    biasing_list = dict()
    all_cnt = 0
    rare_cnt = 0
    with open(filename, "r") as fin:
        for line in fin:
            line = line.strip().upper()
            if len(line) == 0:
                continue
            line = line.split("\t")
            uid, ref_text, ref_rare_words, context_rare_words = line
            context_rare_words = ast.literal_eval(context_rare_words)

            ref_rare_words = ast.literal_eval(ref_rare_words)
            ref_text = ref_text.split()

            biasing_list[uid] = (context_rare_words, ref_rare_words, ref_text)

            all_cnt += len(ref_text)
            rare_cnt += len(ref_rare_words)
    return biasing_list, rare_cnt / all_cnt

def rare_word_rates_distribution(context_collector):
    from lhotse import CutSet
    import numpy as np
    from tqdm import tqdm

    file_name = '/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_train_shuf.jsonl.gz'
    cuts = CutSet.from_file(file_name)
    print(f'len(cuts) = {len(cuts)}')
    
    ratio_stats = []
    for c in tqdm(cuts):
        assert len(c.supervisions) == 1
        text = c.supervisions[0].text
        text = text.split()
        rare = [w for w in text if w in context_collector.rare_words]
        ratio = len(rare) / len(text)
        ratio_stats.append(ratio)

    ratio_stats = np.asarray(ratio_stats)
    hist, bin_edges = np.histogram(ratio_stats, bins=np.arange(0, 1.01, 0.025), density=True)
    print(bin_edges)
    print(hist)

    hist, _ = np.histogram(ratio_stats, bins=np.arange(0, 1.01, 0.025), density=False)
    print(hist)

    t = 0.01
    print(f"Rare ratio < {t}: {(ratio_stats < t).sum()}")

    # len(cuts) = 5886320
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5886320/5886320 [06:07<00:00, 16010.27it/s]
    # [0.    0.025 0.05  0.075 0.1   0.125 0.15  0.175 0.2   0.225 0.25  0.275
    # 0.3   0.325 0.35  0.375 0.4   0.425 0.45  0.475 0.5   0.525 0.55  0.575
    # 0.6   0.625 0.65  0.675 0.7   0.725 0.75  0.775 0.8   0.825 0.85  0.875
    # 0.9   0.925 0.95  0.975 1.   ]
    # [1.23051686e+01 7.17307248e+00 7.08354965e+00 4.23609997e+00
    # 3.01399856e+00 2.55197815e+00 1.26217399e+00 7.30439392e-01
    # 7.10093913e-01 2.50016989e-01 3.12500849e-01 1.54303538e-01
    # 6.29731309e-02 5.96093994e-02 3.32295900e-02 1.99377540e-02
    # 1.59624349e-02 9.39806195e-03 6.40128297e-03 7.13518803e-04
    # 4.66845160e-03 1.32510635e-03 8.76608815e-04 6.31973797e-04
    # 2.03862515e-04 3.05793773e-04 1.42703761e-04 8.15450060e-05
    # 6.11587545e-05 2.03862515e-05 4.07725030e-05 0.00000000e+00
    # 0.00000000e+00 2.03862515e-05 0.00000000e+00 0.00000000e+00
    # 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
    # [1810804 1055575 1042401  623376  443534  375544  185739  107490  104496
    #    36792   45987   22707    9267    8772    4890    2934    2349    1383
    #      942     105     687     195     129      93      30      45      21
    #       12       9       3       6       0       0       3       0       0
    #        0       0       0       0]
    # Rare ratio < 0.01: 1750630

def ref_to_biasing_ref(cuts_file_name, context_collector):
    from lhotse import CutSet
    from tqdm import tqdm

    logging.info(f"Loading cuts from: {cuts_file_name}")
    cuts = CutSet.from_file(cuts_file_name)
    logging.info(f'len(cuts) = {len(cuts)}')

    for c in tqdm(cuts, mininterval=2):
        assert len(c.supervisions) == 1
        text = c.supervisions[0].text
        text = text.split()
        rare = [w for w in text if w in context_collector.rare_words]

        uid = c.supervisions[0].id

        s = str(rare).replace("'", "\"")
        print(f"{uid}\t{c.supervisions[0].text}\t{s}\t{[]}")

def ref_to_biasing_ref2(cuts_file_name, context_collector):
    from tqdm import tqdm

    texts = []
    with open(cuts_file_name, "r") as fin:
        for line in fin:
            line = line.strip()
            if len(line) == 0:
                continue
            line = line.split(maxsplit=1)
            assert len(line) == 2
            uid = line[0]
            txt = line[1]
            texts.append((uid, txt))

    for c in tqdm(texts, mininterval=2):
        uid, text = c
        text0 = text
        text = text.split()
        rare = [w for w in text if w in context_collector.rare_words]

        s = str(rare).replace("'", "\"")
        print(f"{uid}\t{text0}\t{s}\t{[]}")

def main(params):
    logging.info("About to load context generator")
    params.context_dir = Path(params.context_dir)
    params.lang_dir = Path(params.lang_dir)

    sp = spm.SentencePieceProcessor()
    sp.load(str(params.lang_dir / "bpe.model"))

    context_collector = ContextCollector(
        path_rare_words=params.context_dir,
        sp=sp,
        bert_encoder=None,
        is_predefined=False,
        n_distractors=100,
        keep_ratio=1.0,
        is_full_context=False,
    )
    
    # rare_word_rates_distribution(context_collector)

    # cuts_file_name = "/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_dev.jsonl.gz"
    # cuts_file_name = "/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_val.jsonl.gz"
    cuts_file_name = params.cuts_file_name
    # ref_to_biasing_ref(cuts_file_name, context_collector)
    ref_to_biasing_ref2(cuts_file_name, context_collector)

    # for uid, context_rare_words in chain(
    #     context_collector.test_clean_biasing_list.items(),
    #     # context_collector.test_other_biasing_list.items(),
    # ):
    #     # import pdb; pdb.set_trace()
    #     for w in context_rare_words:
    #         if w in context_collector.common_words:
    #             logging.warning(f"{uid} {w} is a common word")
    #         elif w in context_collector.rare_words:
    #             pass
    #         else:
    #             logging.warning(f"{uid} {w} is a new word")

    exit(0)

    n_distractors = 100
    part = "test-clean"
    biasing_list, _ = read_ref_biasing_list(params.context_dir / f"ref/{part}.biasing_{n_distractors}.tsv")

    new_word_cnt = 0
    common_word_cnt = 0
    for uid, entry in biasing_list.items():
        context_rare_words, ref_rare_words, ref_text = entry
        for w in context_rare_words:
            # if w in ref_rare_words: 
            #     continue

            if w in context_collector.common_words:
                common_word_cnt += 1
                logging.warning(f"{uid} {w} is a common word")
            elif w in context_collector.rare_words:
                pass
            else:
                new_word_cnt += 1
                logging.warning(f"{uid} {w} is a new word")

    logging.info(f"common_word_cnt={common_word_cnt}")
    logging.info(f"new_word_cnt={new_word_cnt}")

    # TODO: checkout: egs/librispeech/ASR/pruned_transducer_stateless7_context/context_generator_debug.py

    from collections import namedtuple
    cut = namedtuple('Cut', ['supervisions'])
    supervision = namedtuple('Supervision', ['id'])

    for uid in context_collector.test_clean_biasing_list.keys():  # ["8224-274381-0007"]: # context_collector.test_clean_biasing_list.keys():
        supervision.id = uid  # "1320-122617-0010"
        cut.supervisions = [supervision]
        batch = {"supervisions": {"cut": [cut]}}

        rs1, ws1, us1 = context_collector.get_context_word_list(batch)
        # print(rs1)

        rs2, ws2, us2 = context_generator.get_context_word_list(batch)
        # print(rs2)

        if ws1 != ws2:
            for i, (s1, s2) in enumerate(zip(ws1, ws2)):
                if s1 == s2:
                    continue
                print(s1, s2)

        print(ws1[25], sp.decode(rs1.tolist())[25], rs1.tolist()[25])
        print(ws2[25], sp.decode(rs2.tolist())[25], rs2.tolist()[25])
        print(context_collector.all_words2pieces["DRUMHEAD"])

        assert ws1 == ws2, f"{uid}:\n ws1={ws1},\n ws2={ws2},\n, rs1={sp.decode(rs1.tolist())},\n, rs2={sp.decode(rs2.tolist())},\n"
        assert us1 == us2, f"{uid}: us1={us1}, us2={us2}"



if __name__ == '__main__':
    opts = parse_opts()

    main(opts)


