import logging
import argparse
from pathlib import Path
from collections import namedtuple
import ast
from enum import Enum
import edit_distance
from tqdm import tqdm
from glob import glob
import sentencepiece as spm

from egs.spgispeech.ASR.pruned_transducer_stateless7_context.context_collector import ContextCollector

logging.basicConfig(
    format = "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
    level = logging.CRITICAL
)

def parse_opts():
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--per-utt', type=str, default=None, help='')
    parser.add_argument('--ner-words', type=str, default=None, help='')
    parser.add_argument('--commonwords', type=str, default=None, help='')
    parser.add_argument('--cuts', type=str, default=None, help='')
    parser.add_argument('--lang-dir', type=str, default=None, help='')
    parser.add_argument('--context-dir', type=str, default=None, help='')
    parser.add_argument('--slides', type=str, default=None, help='')

    opts = parser.parse_args()
    logging.info(f"Parameters: {vars(opts)}")
    return opts


class Code(Enum):
    match = 1
    substitution = 2
    insertion = 3
    deletion = 4


class WordError(object):
    def __init__(self):
        self.errors = {
            Code.substitution: 0,
            Code.insertion: 0,
            Code.deletion: 0,
        }
        self.ref_words = 0

    def get_wer(self):
        assert self.ref_words != 0
        errors = (
            self.errors[Code.substitution]
            + self.errors[Code.insertion]
            + self.errors[Code.deletion]
        )
        print(f"errors={errors}, ref_words={self.ref_words}")
        return 100.0 * errors / self.ref_words

    def get_result_string(self):
        return (
            f"error_rate={self.get_wer()}, "
            f"ref_words={self.ref_words}, "
            f"subs={self.errors[Code.substitution]}, "
            f"ins={self.errors[Code.insertion]}, "
            f"dels={self.errors[Code.deletion]}"
        )


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

        hyp = hyp.split(maxsplit=1)
        assert hyp[0][:-1] == uid, f"{hyp[0][:-1]} != {uid}"
        hyp = hyp[1][4:]
        hyp = ast.literal_eval(hyp)

        per_utt[uid] = (ref, hyp)
    
    return per_utt


def parse_spacy_tags(filename):
    spacy_ner_results = dict()
    with open(filename, "r") as fin:
        for line in fin:
            line = line.strip()
            if len(line) == 0:
                continue

            line = line.split(sep="\t", maxsplit=1)
            uid = line[0]
            rs = ast.literal_eval(line[1])

            spacy_ner_results[uid] = rs
    return spacy_ner_results

def parse_commonword_list(filename):
    try:
        with open(filename, "r") as fin:
            common_words = [l.strip() for l in fin if len(l) > 0]
        return set(common_words)
    except:
        return set()

def get_key(my_id):
    my_id = my_id.split("_")
    return my_id[0], int(my_id[1])


def read_ref_biasing_list(filename):
    biasing_list = dict()
    with open(filename, "r") as fin:
        for line in fin:
            line = line.strip()
            if len(line) == 0:
                continue
            line = line.split()
            word, weight = line
            biasing_list[word] = weight
    return biasing_list

def parse_slides(slides_path, common_words):
    biasing_files = glob(f"{slides_path}/*.txt")
    # print(f"biasing_files={biasing_files}")
    ec53_biasing_list = dict()
    for filename in biasing_files:
        biasing_list = read_ref_biasing_list(filename)
        biasing_list2 = {word: weight for word, weight in biasing_list.items() if word not in common_words}
        # biasing_list2 = {word: weight for word, weight in biasing_list.items() if word in self.all_words}  # only contain "known" words
        # biasing_list2 = {word: weight for word, weight in biasing_list.items() if word in self.all_words and word not in self.common_words}
        # biasing_list2 = biasing_list

        base_name = filename.split("/")[-1]
        base_name = base_name.replace(".txt", "")  # one ec

        # print(f"EC {base_name} biasing list size: {len(biasing_list)} -> keep rare only: {len(biasing_list2)}")
        ec53_biasing_list[base_name] = biasing_list2
    return ec53_biasing_list


def score(per_utt, spacy_ner_results, context_collector, biasing_list):
    wer = WordError()
    
    all_entities = set([e[1] for uid, elist in spacy_ner_results.items() for e in elist])
    entity_wer = {e: WordError() for e in all_entities}
    entity_wer[None] = WordError()
    entity_wer["<RARE>"] = WordError()
    entity_wer["<COMMON>"] = WordError()

    spacy_ner_results_indexed = {
        uid : {k : v for k, v in elist} for uid, elist in spacy_ner_results.items()
    }

    # fout_debug = open("debug1-fac.txt", "w")

    for uid, (ref, hyp) in per_utt.items():  # tqdm(per_utt.items()):
        _ref = " ".join(ref).lower()
        batch = {"supervisions": {"text": [_ref]}}
        biasing_words = context_collector._get_random_word_lists(batch, no_distractors=True)
        biasing_words = set(biasing_words[0])

        # https://docs.python.org/2/library/difflib.html#difflib.SequenceMatcher.get_opcodes
        rec_id = get_key(uid)[0]
        sm = edit_distance.SequenceMatcher(a=ref, b=hyp)
        opcodes = sm.get_opcodes()
        for elem in opcodes:
            tag, i1, i2, j1, j2 = elem
            if tag == "equal":
                wer.ref_words += 1
                # entities
                if uid in spacy_ner_results_indexed and i1 in spacy_ner_results_indexed[uid]:
                    entity_wer[spacy_ner_results_indexed[uid][i1]].ref_words += 1
                else:
                    entity_wer[None].ref_words += 1
                # rare
                if i1 < len(ref) and ref[i1].lower() in biasing_words:
                    entity_wer["<RARE>"].ref_words += 1
                else:
                    entity_wer["<COMMON>"].ref_words += 1
            elif tag == "replace":
                if uid in spacy_ner_results_indexed and \
                    i1 in spacy_ner_results_indexed[uid] and \
                    spacy_ner_results_indexed[uid][i1] == "PERCENT" and \
                    ref[i1][:-1] == hyp[j1]:
                    # treat it as equal
                    wer.ref_words += 1
                    if uid in spacy_ner_results_indexed and i1 in spacy_ner_results_indexed[uid]:
                        entity_wer[spacy_ner_results_indexed[uid][i1]].ref_words += 1
                    else:
                        entity_wer[None].ref_words += 1
                    # rare
                    if i1 < len(ref) and ref[i1].lower() in biasing_words:
                        entity_wer["<RARE>"].ref_words += 1
                        # entity_wer["<RARE>"].errors[Code.substitution] += 1
                    else:
                        entity_wer["<COMMON>"].ref_words += 1
                        # entity_wer["<COMMON>"].errors[Code.substitution] += 1
                elif uid in spacy_ner_results_indexed and \
                    i1 in spacy_ner_results_indexed[uid] and \
                    spacy_ner_results_indexed[uid][i1] == "MONEY" and \
                    ref[i1][1:] == hyp[j1]:
                    # treat it as equal
                    wer.ref_words += 1
                    if uid in spacy_ner_results_indexed and i1 in spacy_ner_results_indexed[uid]:
                        entity_wer[spacy_ner_results_indexed[uid][i1]].ref_words += 1
                    else:
                        entity_wer[None].ref_words += 1
                    # rare
                    if i1 < len(ref) and ref[i1].lower() in biasing_words:
                        entity_wer["<RARE>"].ref_words += 1
                        # entity_wer["<RARE>"].errors[Code.substitution] += 1
                    else:
                        entity_wer["<COMMON>"].ref_words += 1
                        # entity_wer["<COMMON>"].errors[Code.substitution] += 1
                else:
                    wer.ref_words += 1
                    wer.errors[Code.substitution] += 1
                    # rare
                    if i1 < len(ref) and ref[i1].lower() in biasing_words:
                        entity_wer["<RARE>"].ref_words += 1
                    else:
                        entity_wer["<COMMON>"].ref_words += 1
                    # entity
                    if uid in spacy_ner_results_indexed and i1 in spacy_ner_results_indexed[uid]:
                        entity_wer[spacy_ner_results_indexed[uid][i1]].ref_words += 1
                        if not (i1 < len(ref) and rec_id in biasing_list and ref[i1].lower() in biasing_list[rec_id]):
                            entity_wer[spacy_ner_results_indexed[uid][i1]].errors[Code.substitution] += 1
                            if i1 < len(ref) and ref[i1].lower() in biasing_words:
                                entity_wer["<RARE>"].errors[Code.substitution] += 1
                            else:
                                entity_wer["<COMMON>"].errors[Code.substitution] += 1
                    else:
                        entity_wer[None].ref_words += 1
                        entity_wer[None].errors[Code.substitution] += 1
                        if i1 < len(ref) and ref[i1].lower() in biasing_words:
                            entity_wer["<RARE>"].errors[Code.substitution] += 1
                        else:
                            entity_wer["<COMMON>"].errors[Code.substitution] += 1
                    # # rare
                    # if i1 < len(ref) and ref[i1] not in common_words:
                    #     entity_wer["<RARE>"].ref_words += 1
                    #     entity_wer["<RARE>"].errors[Code.substitution] += 1
                    # else:
                    #     entity_wer["<COMMON>"].ref_words += 1
                    #     entity_wer["<COMMON>"].errors[Code.substitution] += 1
            elif tag == "delete":
                wer.ref_words += 1
                wer.errors[Code.deletion] += 1
                # rare
                if i1 < len(ref) and ref[i1].lower() in biasing_words:
                    entity_wer["<RARE>"].ref_words += 1
                else:
                    entity_wer["<COMMON>"].ref_words += 1
                # entity
                if uid in spacy_ner_results_indexed and i1 in spacy_ner_results_indexed[uid]:
                    entity_wer[spacy_ner_results_indexed[uid][i1]].ref_words += 1
                    if not (i1 < len(ref) and rec_id in biasing_list and ref[i1].lower() in biasing_list[rec_id]):
                        entity_wer[spacy_ner_results_indexed[uid][i1]].errors[Code.deletion] += 1
                        if i1 < len(ref) and ref[i1].lower() in biasing_words:
                            entity_wer["<RARE>"].errors[Code.deletion] += 1
                        else:
                            entity_wer["<COMMON>"].errors[Code.deletion] += 1
                else:
                    entity_wer[None].ref_words += 1
                    entity_wer[None].errors[Code.deletion] += 1
                    if i1 < len(ref) and ref[i1].lower() in biasing_words:
                        entity_wer["<RARE>"].errors[Code.deletion] += 1
                    else:
                        entity_wer["<COMMON>"].errors[Code.deletion] += 1
                # # rare
                # if i1 < len(ref) and ref[i1] not in common_words:
                #     entity_wer["<RARE>"].errors[Code.deletion] += 1
                # else:
                #     entity_wer["<COMMON>"].errors[Code.deletion] += 1
            elif tag == "insert":
                wer.errors[Code.insertion] += 1
                # if hyp_tokens[hyp_idx] in biasing_words:
                #     b_wer.errors[Code.insertion] += 1
                # else:
                #     u_wer.errors[Code.insertion] += 1
                # rare
                if j1 < len(hyp) and hyp[j1].lower() in biasing_words:
                    entity_wer["<RARE>"].errors[Code.insertion] += 1
                else:
                    entity_wer["<COMMON>"].errors[Code.insertion] += 1
            else:
                logging.error("Cannot reach here!")

            # # For debug            
            # if tag != "equal" and tag != "insert" and uid in spacy_ner_results_indexed and \
            #     i1 in spacy_ner_results_indexed[uid] and \
            #     spacy_ner_results_indexed[uid][i1] == "ORG":
            #     print(uid, tag, f"<{i1}, {j1}>", f"[{ref[i1] if i1 < len(ref) else None}]", f"[{hyp[j1] if j1 < len(hyp) else None}]", ref, file=fout_debug)

    # fout_debug.close()

    # Report results
    e_wer_str = ""
    for e in [None, "PERSON", "PRODUCT", "FAC", "ORG", "GPE", "LOC", "LAW", "WORK_OF_ART", "MONEY", "PERCENT", "DATE", "TIME", "CARDINAL", "ORDINAL", "QUANTITY", "EVENT", "NORP", "LANGUAGE"]:
        e_wer = entity_wer[e]
        print(f"WER [{e}]: {e_wer.get_result_string()}")
        e_wer_str += f"\t{e_wer.get_wer():.2f}"
    print(f"WER: {wer.get_result_string()} {wer.get_wer():.2f}")
    print(f"Entity: {e_wer_str}")
    print(f"Rare/Common: {entity_wer['<COMMON>'].get_wer():.2f}/{entity_wer['<RARE>'].get_wer():.2f}")

def main(opts):
    per_utt = parse_per_utt(opts.per_utt)
    spacy_ner_results = parse_spacy_tags(opts.ner_words)
    common_words = parse_commonword_list(opts.commonwords)
    biasing_list = parse_slides(opts.slides, common_words)

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
    score(per_utt, spacy_ner_results, context_collector, biasing_list)
    

if __name__ == '__main__':
    opts = parse_opts()

    main(opts)




