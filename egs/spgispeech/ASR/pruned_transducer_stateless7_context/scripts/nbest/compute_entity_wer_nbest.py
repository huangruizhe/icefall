import logging
import argparse
from pathlib import Path
from collections import namedtuple, defaultdict
import ast
from enum import Enum
import edit_distance
from tqdm import tqdm
import copy
import math

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
    parser.add_argument('--ner-words', type=str, default=None, help='')
    parser.add_argument('--commonwords', type=str, default=None, help='')
    parser.add_argument('--cuts', type=str, default=None, help='')
    parser.add_argument('--phrase-mappings', type=str, default=None, help='')
    parser.add_argument('--topk', type=int, default=None, help='')
    parser.add_argument('--per-utt-out', type=str, default=None, help='')

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

    def reset(self):
        self.errors[Code.substitution] = 0
        self.errors[Code.insertion] = 0
        self.errors[Code.deletion] = 0
        self.ref_words = 0
    
    def copy(self):
        return copy.deepcopy(self)

    def get_wer(self):
        # assert self.ref_words != 0
        if self.ref_words == 0:
            return math.inf
        
        errors = (
            self.errors[Code.substitution]
            + self.errors[Code.insertion]
            + self.errors[Code.deletion]
        )
        return 100.0 * errors / self.ref_words

    def get_result_string(self):
        return (
            f"error_rate={self.get_wer()}, "
            f"ref_words={self.ref_words}, "
            f"subs={self.errors[Code.substitution]}, "
            f"ins={self.errors[Code.insertion]}, "
            f"dels={self.errors[Code.deletion]}"
        )

    def add_another(self, another_word_error):
        self.errors[Code.substitution] += another_word_error.errors[Code.substitution]
        self.errors[Code.insertion] += another_word_error.errors[Code.insertion]
        self.errors[Code.deletion] += another_word_error.errors[Code.deletion]
        self.ref_words += another_word_error.ref_words


def parse_per_utt(filename, topk=None):
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

        if topk is not None and int(uid.split("---")[1]) >= topk:
            continue

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


def score(per_utt, spacy_ner_results, common_words, opts):
    wer = WordError()
    
    all_entities = set([e[1] for uid, elist in spacy_ner_results.items() for e in elist])
    entity_wer = {e: WordError() for e in all_entities}
    entity_wer["<None>"] = WordError()
    entity_wer["<Entity>"] = WordError()
    entity_wer["<RARE>"] = WordError()
    entity_wer["<COMMON>"] = WordError()

    uid_wer = {}
    uid_entity_wer = {}

    spacy_ner_results_indexed = {
        uid : {k : v for k, v in elist} for uid, elist in spacy_ner_results.items()
    }

    # fout_debug = open("debug1-fac.txt", "w")

    for uid, (ref, hyp) in tqdm(per_utt.items()):  # tqdm(per_utt.items()):
        wer.reset()
        for _wer in entity_wer.values():
            _wer.reset()
        
        _uid = uid.split("---")[0]

        # https://docs.python.org/2/library/difflib.html#difflib.SequenceMatcher.get_opcodes
        sm = edit_distance.SequenceMatcher(a=ref, b=hyp)
        opcodes = sm.get_opcodes()
        for elem in opcodes:
            tag, i1, i2, j1, j2 = elem
            if tag == "equal":
                wer.ref_words += 1
                # entities
                if _uid in spacy_ner_results_indexed and i1 in spacy_ner_results_indexed[_uid]:
                    entity_tag = spacy_ner_results_indexed[_uid][i1]
                    entity_wer[entity_tag].ref_words += 1
                    entity_wer["<Entity>"].ref_words += 1
                else:
                    entity_wer["<None>"].ref_words += 1

                # rare
                if i1 < len(ref) and ref[i1] not in common_words:
                    entity_wer["<RARE>"].ref_words += 1
                else:
                    entity_wer["<COMMON>"].ref_words += 1
            elif tag == "replace":
                if _uid in spacy_ner_results_indexed and \
                    i1 in spacy_ner_results_indexed[_uid] and \
                    spacy_ner_results_indexed[_uid][i1] == "PERCENT" and \
                    ref[i1][:-1] == hyp[j1]:
                    # treat it as equal
                    wer.ref_words += 1
                    if _uid in spacy_ner_results_indexed and i1 in spacy_ner_results_indexed[_uid]:
                        entity_tag = spacy_ner_results_indexed[_uid][i1]
                        entity_wer[entity_tag].ref_words += 1
                        entity_wer["<Entity>"].ref_words += 1
                    else:
                        entity_wer["<None>"].ref_words += 1

                    # rare
                    if i1 < len(ref) and ref[i1] not in common_words:
                        entity_wer["<RARE>"].ref_words += 1
                        entity_wer["<RARE>"].errors[Code.substitution] += 1
                    else:
                        entity_wer["<COMMON>"].ref_words += 1
                        entity_wer["<COMMON>"].errors[Code.substitution] += 1
                elif _uid in spacy_ner_results_indexed and \
                    i1 in spacy_ner_results_indexed[_uid] and \
                    spacy_ner_results_indexed[_uid][i1] == "MONEY" and \
                    ref[i1][1:] == hyp[j1]:
                    # treat it as equal
                    wer.ref_words += 1
                    if _uid in spacy_ner_results_indexed and i1 in spacy_ner_results_indexed[_uid]:
                        entity_tag = spacy_ner_results_indexed[_uid][i1]
                        entity_wer[entity_tag].ref_words += 1
                        entity_wer["<Entity>"].ref_words += 1
                    else:
                        entity_wer["<None>"].ref_words += 1
                    
                    # rare
                    if i1 < len(ref) and ref[i1] not in common_words:
                        entity_wer["<RARE>"].ref_words += 1
                        entity_wer["<RARE>"].errors[Code.substitution] += 1
                    else:
                        entity_wer["<COMMON>"].ref_words += 1
                        entity_wer["<COMMON>"].errors[Code.substitution] += 1
                else:
                    wer.ref_words += 1
                    wer.errors[Code.substitution] += 1
                    # entity
                    if _uid in spacy_ner_results_indexed and i1 in spacy_ner_results_indexed[_uid]:
                        entity_tag = spacy_ner_results_indexed[_uid][i1]
                        entity_wer[entity_tag].ref_words += 1
                        entity_wer[entity_tag].errors[Code.substitution] += 1
                        entity_wer["<Entity>"].ref_words += 1
                        entity_wer["<Entity>"].errors[Code.substitution] += 1
                    else:
                        entity_wer["<None>"].ref_words += 1
                        entity_wer["<None>"].errors[Code.substitution] += 1
                    
                    # rare
                    if i1 < len(ref) and ref[i1] not in common_words:
                        entity_wer["<RARE>"].ref_words += 1
                        entity_wer["<RARE>"].errors[Code.substitution] += 1
                    else:
                        entity_wer["<COMMON>"].ref_words += 1
                        entity_wer["<COMMON>"].errors[Code.substitution] += 1
            elif tag == "delete":
                wer.ref_words += 1
                wer.errors[Code.deletion] += 1
                # entity
                if _uid in spacy_ner_results_indexed and i1 in spacy_ner_results_indexed[_uid]:
                    entity_tag = spacy_ner_results_indexed[_uid][i1]
                    entity_wer[entity_tag].ref_words += 1
                    entity_wer[entity_tag].errors[Code.deletion] += 1
                    entity_wer["<Entity>"].ref_words += 1
                    entity_wer["<Entity>"].errors[Code.deletion] += 1
                else:
                    entity_wer["<None>"].ref_words += 1
                    entity_wer["<None>"].errors[Code.deletion] += 1

                # rare
                if i1 < len(ref) and ref[i1] not in common_words:
                    entity_wer["<RARE>"].ref_words += 1
                    entity_wer["<RARE>"].errors[Code.deletion] += 1
                else:
                    entity_wer["<COMMON>"].ref_words += 1
                    entity_wer["<COMMON>"].errors[Code.deletion] += 1
            elif tag == "insert":
                wer.errors[Code.insertion] += 1
                # if hyp_tokens[hyp_idx] in biasing_words:
                #     b_wer.errors[Code.insertion] += 1
                # else:
                #     u_wer.errors[Code.insertion] += 1
                # rare
                if j1 < len(hyp) and hyp[j1] not in common_words:
                    entity_wer["<RARE>"].errors[Code.insertion] += 1
                else:
                    entity_wer["<COMMON>"].errors[Code.insertion] += 1
            else:
                logging.error("Cannot reach here!")

            # # For debug
            # if tag != "equal" and tag != "insert" and _uid in spacy_ner_results_indexed and \
            #     i1 in spacy_ner_results_indexed[_uid] and \
            #     spacy_ner_results_indexed[_uid][i1] == "ORG":
            #     print(uid, tag, f"<{i1}, {j1}>", f"[{ref[i1] if i1 < len(ref) else None}]", f"[{hyp[j1] if j1 < len(hyp) else None}]", ref, file=fout_debug)

        uid_wer[uid] = wer.copy()
        uid_entity_wer[uid] = copy.deepcopy(entity_wer)
        # breakpoint()
        # !wer.get_result_string()
        # !entity_wer['ORG'].get_result_string()
        # !spacy_ner_results_indexed[_uid]
        # !import code; code.interact(local=vars())
    # fout_debug.close()

    uid_nbestuids = defaultdict(list)
    for uid in uid_wer:
        _uid = uid.split("---")[0]
        uid_nbestuids[_uid].append(uid)
    
    wer.reset()
    for _wer in entity_wer.values():
        _wer.reset()
    for uid, topk_uids in uid_nbestuids.items():
        my_final_wer_uid = min(topk_uids, key=lambda x: uid_wer[x].get_wer())
        # my_final_wer_uid = uid.split("---")[0] + "---0"
        # my_final_wer_uid = max(topk_uids, key=lambda x: int(x.split("---")[1]))
        wer.add_another(uid_wer[my_final_wer_uid])

        for _tag, _wer in entity_wer.items():
            my_final_wer_uid = min(topk_uids, key=lambda x: uid_entity_wer[x][_tag].get_wer())
            # my_final_wer_uid = uid.split("---")[0] + "---0"
            # my_final_wer_uid = max(topk_uids, key=lambda x: int(x.split("---")[1]))
            _wer.add_another(uid_entity_wer[my_final_wer_uid][_tag])

    if opts.per_utt_out is not None:
        per_utt_original = parse_per_utt(opts.per_utt)

        with open(opts.per_utt_out, "w") as fout:
            for uid, topk_uids in uid_nbestuids.items():
                my_final_wer_uid = min(topk_uids, key=lambda x: uid_wer[x].get_wer())
                # my_final_wer_uid = uid.split("---")[0] + "---0"
                # my_final_wer_uid = max(topk_uids, key=lambda x: int(x.split("---")[1]))

                _ref, _hyp = per_utt_original[my_final_wer_uid]
                print(f"{uid}:\tref={str(_ref)}", file=fout)
                print(f"{uid}:\thyp={str(_hyp)}", file=fout)
        print(f"Saved results to: {opts.per_utt_out}")

    # Report results
    e_wer_str = ""
    for e in ["<Entity>", "<None>", "PERSON", "PRODUCT", "FAC", "ORG", "GPE", "LOC", "LAW", "WORK_OF_ART", "MONEY", "PERCENT", "DATE", "TIME", "CARDINAL", "ORDINAL", "QUANTITY", "EVENT", "NORP", "LANGUAGE"]:
        e_wer = entity_wer[e]
        print(f"WER [{e}]: {e_wer.get_result_string()}")
        e_wer_str += f"\t{e_wer.get_wer():.2f}"
    print(f"WER: {wer.get_result_string()} {wer.get_wer():.2f}")
    print(f"Entity: {e_wer_str}")
    print(f"Rare/Common: {entity_wer['<COMMON>'].get_wer():.2f}/{entity_wer['<RARE>'].get_wer():.2f}")
    print(entity_wer['<COMMON>'].get_result_string())
    print(entity_wer['<RARE>'].get_result_string())


def _normalize(text, english_normalizer, phrase_mappings):
    text_norm = english_normalizer(text)
    text_norm = icefall_spgi_specific_normalizer(text_norm)
    text_norm = " ".join(w if w not in phrase_mappings else phrase_mappings[w] for w in text_norm.split())
    return text_norm


def _normalize_ref(ref, english_normalizer, phrase_mappings, spacy_ner_results, uid):
    my_ner = spacy_ner_results[uid]
    ref = ref.lower().split()
    buffer = []

    prev_wer_tag = None
    my_ner = {int(k): v for k, v in my_ner}

    def process_buffer(bf, _wer_tag):
        if len(bf) == 0:
            return []
        _txt = _normalize(" ".join(bf), english_normalizer, phrase_mappings)
        return [(x, _wer_tag) for x in _txt.split()]

    rs = []
    for iw, w in enumerate(ref):
        wer_tag = my_ner.get(iw, None)
        if wer_tag != prev_wer_tag:
            rs.extend(process_buffer(buffer, prev_wer_tag))
            buffer = [w]
            prev_wer_tag = wer_tag
        else:
            buffer.append(w)
    if len(buffer) > 0:
        rs.extend(process_buffer(buffer, prev_wer_tag))
    
    new_ref = " ".join([x[0] for x in rs])
    new_ner = [(ix, x[1]) for ix, x in enumerate(rs) if x[1] is not None]
    # if not all(x == y for x, y in zip(new_ner, spacy_ner_results[uid])):
    #     print(" ".join(ref), list(enumerate(ref)), spacy_ner_results[uid])
    #     print(new_ref, list(enumerate(new_ref.split())), new_ner)
    #     breakpoint()
    spacy_ner_results[uid] = new_ner

    return new_ref


def normalize_per_utt(per_utt, spacy_ner_results):
    if opts.phrase_mappings is not None:
        with open(opts.phrase_mappings) as fin:
            phrase_mappings = [l.strip().split("|") for l in fin]
            phrase_mappings = {k: v for (k, v) in phrase_mappings}
    else:
        phrase_mappings = dict()

    english_normalizer = EnglishTextNormalizer()

    for uid in tqdm(per_utt):
        hyp = " ".join(per_utt[uid][1]).lower()
        hyp = _normalize(hyp, english_normalizer, phrase_mappings)

        _uid = uid.split("---")[0]
        ref = " ".join(per_utt[uid][0]).lower()
        ref = _normalize_ref(ref, english_normalizer, phrase_mappings, spacy_ner_results, _uid)

        hyp = hyp.split()
        ref = ref.split()
        per_utt[uid] = (ref, hyp)

    return per_utt

def main(opts):
    per_utt = parse_per_utt(opts.per_utt, opts.topk)
    spacy_ner_results = parse_spacy_tags(opts.ner_words)
    common_words = parse_commonword_list(opts.commonwords)

    per_utt = normalize_per_utt(per_utt, spacy_ner_results)

    score(per_utt, spacy_ner_results, common_words, opts)
    

if __name__ == '__main__':
    opts = parse_opts()

    main(opts)




