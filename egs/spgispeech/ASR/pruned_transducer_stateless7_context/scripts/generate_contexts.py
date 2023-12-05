from egs.spgispeech.ASR.pruned_transducer_stateless7_context.context_collector import ContextCollector

import logging
import argparse
from pathlib import Path
import sentencepiece as spm
from itertools import chain
import ast
import numpy as np
import random
import copy
from tqdm import tqdm
import json


import sys
sys.path.append('/exp/draj/jsalt2023/icefall/egs/librispeech/ASR/zipformer')
# sys.path.append('/exp/rhuang/meta/icefall/egs/librispeech/ASR/pruned_transducer_stateless7_context/')
from asr_datamodule_uniphore import UniphoreAsrDataModule

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

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/uniphore_contexts/",
        help="",
    )

    parser.add_argument(
        "--n-distractors",
        type=int,
        default=100,
        help="",
    )

    UniphoreAsrDataModule.add_arguments(parser)

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


def _get_random_word_lists(self, batch):
    texts = batch["supervisions"]["text"]

    new_words = []
    rare_words_list = []
    for text in texts:
        rare_words = []
        for word in text.split():
            if self.is_full_context or word not in self.common_words:
                rare_words.append(word)

            if self.all_words2pieces is not None and word not in self.all_words2pieces:
                new_words.append(word)
                # self.all_words2pieces[word] = self.sp.encode(word, out_type=int)
            if self.all_words2embeddings is not None and word not in self.all_words2embeddings:
                new_words.append(word)
                # logging.info(f"New word detected: {word}")
                # self.all_words2embeddings[word] = self.bert_encoder.encode_strings([word])[0]
        
        rare_words = list(set(rare_words))  # deduplication

        if self.keep_ratio < 1.0 and len(rare_words) > 0:
            # # method 1:
            # keep_size = int(len(rare_words) * self.keep_ratio)
            # if keep_size > 0:
            #     rare_words = random.sample(rare_words, keep_size)
            # else:
            #     rare_words = []
            
            # method 2:
            x = np.random.rand(len(rare_words))
            new_rare_words = []
            for xi in range(len(rare_words)):
                if x[xi] < self.keep_ratio:
                    new_rare_words.append(rare_words[xi])
            rare_words = new_rare_words

        rare_words_list.append(rare_words)
    
    self.temp_dict = None
    if len(new_words) > 0:
        self.temp_dict = self.add_new_words(new_words, return_dict=True, silent=True)

    if self.ratio_distractors is not None:
        n_distractors_each = []
        for rare_words in rare_words_list:
            n_distractors_each.append(len(rare_words) * self.ratio_distractors)
        n_distractors_each = np.asarray(n_distractors_each, dtype=int)
    else:
        if self.n_distractors == -1:  # variable context list sizes
            n_distractors_each = np.random.randint(low=10, high=500, size=len(texts))
            # n_distractors_each = np.random.randint(low=80, high=300, size=len(texts))
        else:
            n_distractors_each = np.full(len(texts), self.n_distractors, int)
    distractors_cnt = n_distractors_each.sum()

    distractors = random.sample(  # without replacement
        self.rare_words, 
        distractors_cnt
    )  # TODO: actually the context should contain both rare and common words
    # distractors = random.choices(  # random choices with replacement
    #     self.rare_words, 
    #     distractors_cnt,
    # )

    rare_words_list0 = copy.deepcopy(rare_words_list)

    distractors_pos = 0
    for i, rare_words in enumerate(rare_words_list):
        rare_words.extend(distractors[distractors_pos: distractors_pos + n_distractors_each[i]])
        distractors_pos += n_distractors_each[i]
        # random.shuffle(rare_words)
        # logging.info(rare_words)
    assert distractors_pos == len(distractors)

    return rare_words_list0, rare_words_list


def main(params):
    logging.info("About to load context generator")
    params.context_dir = Path(params.context_dir)
    params.lang_dir = Path(params.lang_dir)

    sp = spm.SentencePieceProcessor()
    sp.load(str(params.lang_dir / "bpe.model"))

    context_collector = ContextCollector(
        path_rare_words=params.context_dir,
        sp=sp,
        is_predefined=False,
        n_distractors=500,
        keep_ratio=1.0,
        is_full_context=False,
    )
    context_collector._get_random_word_lists = _get_random_word_lists

    uniphore = UniphoreAsrDataModule(opts)

    test_healthcare_cuts = uniphore.test_healthcare_cuts()
    test_banking_cuts = uniphore.test_banking_cuts()
    test_insurance_cuts = uniphore.test_insurance_cuts()

    test_healthcare_dl = uniphore.test_dataloaders(test_healthcare_cuts)
    test_banking_dl = uniphore.test_dataloaders(test_banking_cuts)
    test_insurance_dl = uniphore.test_dataloaders(test_insurance_cuts)

    test_sets = ["healthcare", "banking", "insurance"]
    test_dls = [test_healthcare_dl, test_banking_dl, test_insurance_dl]
    test_cuts = [test_healthcare_cuts, test_banking_cuts, test_insurance_cuts]

    # get rare word list
    for test_set, test_cs in zip(test_sets, test_cuts):
        logging.info(f"Generating contexts for {test_set}")
        total_words_cnt = 0
        rare_words_cnt = 0
        with open(opts.output_dir + f"/ref/{test_set}.biasing_{opts.n_distractors}.tsv", "w") as fout:
            for c in tqdm(test_cs):                
                uid = c.id
                text = c.supervisions[0].text.lower()
                batch = {"supervisions": {"text": [text]}}
                rare_words_list0, rare_words_list = _get_random_word_lists(context_collector, batch)
                print(f"{uid}\t{text}\t{json.dumps(rare_words_list0[0])}\t{json.dumps(rare_words_list[0])}", file=fout)

                text = text.split()
                total_words_cnt += len(text)
                for word in text:
                    if word not in context_collector.common_words:
                        rare_words_cnt += 1
            print(f"{test_set}: {rare_words_cnt}/{total_words_cnt} = {rare_words_cnt/total_words_cnt*100:.2f}%")


if __name__ == '__main__':
    opts = parse_opts()

    main(opts)


# [6, 6, 6, 7, 3, 7, 8, 8, 4, 5, 3, 6, 4, 4, 6, 4, 6, 4, 5, 3, 5, 4, 4, 3, 4, 8, 2, 4, 5, 6, 3, 5, 5, 7, 3, 4, 4, 2, 4, 5, 4, 8, 5, 8, 9, 4, 3, 3, 6, 7, 7, 3, 4, 4, 4, 9, 6, 5, 4, 4, 3, 3, 3, 3, 2, 4, 5, 6, 6, 7, 3, 4, 6, 4, 2, 4, 4, 6, 5, 4, 4, 6, 5, 3, 3, 5, 2, 4, 7, 5, 5, 4, 4, 5, 3, 3, 4, 4, 3, 5, 4, 3, 5, 5]
# [6, 6, 6, 7, 3, 7, 8, 8, 4, 5, 3, 6, 4, 4, 6, 4, 6, 4, 5, 3, 5, 4, 4, 3, 4, 5, 2, 4, 5, 6, 3, 5, 5, 7, 3, 4, 4, 2, 4, 5, 4, 8, 5, 8, 9, 4, 3, 3, 6, 7, 7, 3, 4, 4, 4, 9, 6, 5, 4, 4, 3, 3, 3, 3, 2, 4, 5, 6, 6, 7, 3, 4, 6, 4, 2, 4, 4, 6, 5, 4, 4, 6, 5, 3, 3, 5, 2, 4, 7, 5, 5, 4, 4, 5, 3, 3, 4, 4, 3, 5, 4, 3, 5, 5]