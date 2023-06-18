from egs.librispeech.ASR.pruned_transducer_stateless7_context.context_collector import ContextCollector
from egs.librispeech.ASR.pruned_transducer_stateless7_context.context_generator_debug import ContextGenerator
# export PYTHONPATH=/export/fs04/a12/rhuang/icefall_align2/:$PYTHONPATH
# export PYTHONPATH=/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/pruned_transducer_stateless7_context:$PYTHONPATH

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


def read_ref_biasing_list(filename):
    biasing_list = dict()
    all_cnt = 0
    rare_cnt = 0
    list_size = 0
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
            list_size += len(context_rare_words)
    return biasing_list, rare_cnt / all_cnt, list_size / rare_cnt

def main(params):
    logging.info("About to load context generator")
    params.context_dir = Path(params.context_dir)
    params.lang_dir = Path(params.lang_dir)

    sp = spm.SentencePieceProcessor()
    sp.load(str(params.lang_dir / "bpe.model"))

    context_collector = ContextCollector(
        path_is21_deep_bias=params.context_dir,
        sp=sp,
        is_predefined=True,
        n_distractors=500,
        keep_ratio=1.0,
        is_full_context=False,
    )

    # context_generator = ContextGenerator(
    #     path_is21_deep_bias=params.context_dir,
    #     sp=sp,
    #     is_predefined=True,
    #     n_distractors=100,
    #     keep_ratio=1.0,
    #     is_full_context=False,
    # )
    
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

    if False:  # check if the predefined biasing list contains common words
        n_distractors = 2000
        part = "test-other"
        biasing_list, _ = read_ref_biasing_list(params.context_dir / f"ref/{part}.biasing_{n_distractors}.tsv")

        new_word_cnt = 0
        common_word_cnt = 0
        all_words = set()
        new_words = set()
        for uid, entry in biasing_list.items():
            context_rare_words, ref_rare_words, ref_text = entry
            for w in context_rare_words:
                # if w in ref_rare_words: 
                #     continue

                if w in context_collector.common_words:
                    common_word_cnt += 1
                    logging.warning(f"{uid} {w} is a common word")
                    all_words.add(w)
                elif w in context_collector.rare_words:
                    all_words.add(w)
                else:
                    new_words.add(w)
                    new_word_cnt += 1
                    logging.warning(f"{uid} {w} is a new word")

        logging.info(f"common_word_cnt={common_word_cnt}")
        logging.info(f"new_word_cnt={len(new_words)}")
        logging.info(f"all_word_cnt={len(all_words)}")

    # TODO: checkout: egs/librispeech/ASR/pruned_transducer_stateless7_context/context_generator_debug.py

    if False:  # compare two implementations
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

    if False:  # compute distractor ratio from the biasing list:
        n_distractors = 500
        part = "test-other"
        biasing_list, rare_frequency, distractor_ratio = read_ref_biasing_list(params.context_dir / f"ref/{part}.biasing_{n_distractors}.tsv")
        logging.info(f"distractor_ratio = {distractor_ratio}")

    if False:  # compute rare word count distribution (how many rare words are there in each utterance?)
        from collections import Counter
        import numpy as np

        n_distractors = 100
        part = "test-other"
        biasing_list, rare_frequency, distractor_ratio = read_ref_biasing_list(params.context_dir / f"ref/{part}.biasing_{n_distractors}.tsv")
        stats = []
        for uid, context in biasing_list.items():
            stats.append(len(context[1]))
        stats = Counter(stats)
        logging.info(stats)
        logging.info(f"len(biasing_list) = {len(biasing_list)}")

        # def utt_rare_ratio(my_stats, keep_ratio):
        #     cnt_all = 0
        #     cnt_utt_rare = 0  # number of utterances containing at least one rare word
        #     for k, v in my_stats.items():
        #         cnt_all += v
        #         if int(k * keep_ratio) > 0:
        #             cnt_utt_rare += v
        #     logging.info(f"keep_ratio={keep_ratio}: {cnt_utt_rare/cnt_all:.4f}")
        def stats_with_ratio(my_stats, keep_ratio):
            new_stats = Counter()
            for k, v in my_stats.items():
                new_stats[int(k * keep_ratio)] += v
            # logging.info(f"keep_ratio={keep_ratio}: {cnt_utt_rare/cnt_all:.4f}")
            print(f"ratio={keep_ratio}:", dict(new_stats))
        
        for x in np.arange(0, 1.05, 0.1):
            stats_with_ratio(stats, x)
    
    if True:  # get the train rare words stats
        from collections import Counter
        import lhotse
        from lhotse import load_manifest, CutSet
        from tqdm import tqdm
        cuts_ = []
        cuts = CutSet.from_file("data/fbank/librispeech_cuts_train-all-shuf.jsonl.gz")
        logging.info("loaded cuts")
        for c in tqdm(cuts):
            if not c.id.endswith("sp0.9") and not c.id.endswith("sp1.1"):
                cuts_.append(c)
        logging.info(len(cuts_))

        counter = Counter()
        for c in cuts_:
            text = c.supervisions[0].text
            for w in text.split():
                if w not in context_collector.common_words:
                    counter[w] += 1

        a = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        with open("/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/ruizhe_contextual/exp/train_rare_words.txt", "w") as fout:
            for k, v in a:
                print(f"{k}\t{v}", file=fout)


if __name__ == '__main__':
    opts = parse_opts()

    main(opts)


# [6, 6, 6, 7, 3, 7, 8, 8, 4, 5, 3, 6, 4, 4, 6, 4, 6, 4, 5, 3, 5, 4, 4, 3, 4, 8, 2, 4, 5, 6, 3, 5, 5, 7, 3, 4, 4, 2, 4, 5, 4, 8, 5, 8, 9, 4, 3, 3, 6, 7, 7, 3, 4, 4, 4, 9, 6, 5, 4, 4, 3, 3, 3, 3, 2, 4, 5, 6, 6, 7, 3, 4, 6, 4, 2, 4, 4, 6, 5, 4, 4, 6, 5, 3, 3, 5, 2, 4, 7, 5, 5, 4, 4, 5, 3, 3, 4, 4, 3, 5, 4, 3, 5, 5]
# [6, 6, 6, 7, 3, 7, 8, 8, 4, 5, 3, 6, 4, 4, 6, 4, 6, 4, 5, 3, 5, 4, 4, 3, 4, 5, 2, 4, 5, 6, 3, 5, 5, 7, 3, 4, 4, 2, 4, 5, 4, 8, 5, 8, 9, 4, 3, 3, 6, 7, 7, 3, 4, 4, 4, 9, 6, 5, 4, 4, 3, 3, 3, 3, 2, 4, 5, 6, 6, 7, 3, 4, 6, 4, 2, 4, 4, 6, 5, 4, 4, 6, 5, 3, 3, 5, 2, 4, 7, 5, 5, 4, 4, 5, 3, 3, 4, 4, 3, 5, 4, 3, 5, 5]