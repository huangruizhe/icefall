import torch
import random
from pathlib import Path
import sentencepiece as spm
from typing import Union, List
import logging
import ast
import numpy as np
from itertools import chain
# from word_encoder_bert import BertEncoder
from context_wfst import generate_context_graph_nfa
from collections import defaultdict
import k2
import string
import re


class SentenceTokenizer:
    def encode(self, word_list: List, out_type: type = int) -> List:
        """
        Encode a list of words into a list of tokens

        Args:
            word_list: 
                A list of words where each word is a string. 
                E.g., ["nihao", "hello", "你好"]
            out_type:
                This defines the output type. If it is an "int" type, 
                then each token is represented by its interger token id.
        Returns:
                A list of tokenized words, where each tokenization 
                is a list of tokens.
        """
        pass

class ContextCollector(torch.utils.data.Dataset):
    def __init__(
        self, 
        path_is21_deep_bias: Path,
        sp: Union[spm.SentencePieceProcessor, SentenceTokenizer],
        bert_encoder = None,
        n_distractors: int = 100,
        ratio_distractors: int = None,
        is_predefined: bool = False,
        keep_ratio: float = 1.0,
        is_full_context: bool = False,
        backoff_id: int = None,
        confusionp_path: str = None,
    ):
        self.sp = sp
        self.bert_encoder = bert_encoder
        self.path_is21_deep_bias = path_is21_deep_bias
        self.n_distractors = n_distractors
        self.ratio_distractors = ratio_distractors
        self.is_predefined = is_predefined
        self.keep_ratio = keep_ratio
        self.is_full_context = is_full_context   # use all words (rare or common) in the context
        # self.embedding_dim = self.bert_encoder.bert_model.config.hidden_size
        self.backoff_id = backoff_id
        self.confusionp_path = confusionp_path

        logging.info(f"""
            n_distractors={n_distractors},
            ratio_distractors={ratio_distractors},
            is_predefined={is_predefined},
            keep_ratio={keep_ratio},
            is_full_context={is_full_context},
            bert_encoder={bert_encoder.name if bert_encoder is not None else None},
            confusionp_path={confusionp_path},
        """)

        self.common_words = None
        self.rare_words = None
        self.all_words = None
        with open(path_is21_deep_bias / "words/all_rare_words.txt", "r") as fin:
            self.rare_words = [l.strip().upper() for l in fin if len(l) > 0]
        
        with open(path_is21_deep_bias / "words/common_words_5k.txt", "r") as fin:
            self.common_words = [l.strip().upper() for l in fin if len(l) > 0]
        
        self.all_words = self.rare_words + self.common_words  # sp needs a list of strings, can't be a set
        self.common_words = set(self.common_words)
        self.rare_words = set(self.rare_words)

        self.common_words_list = list(self.common_words)
        self.rare_words_list = list(self.rare_words)

        logging.info(f"Number of common words: {len(self.common_words)}. Examples: {random.sample(self.common_words_list, 5)}")
        logging.info(f"Number of rare words: {len(self.rare_words)}. Examples: {random.sample(self.rare_words_list, 5)}")
        logging.info(f"Number of all words: {len(self.all_words)}. Examples: {random.sample(self.all_words, 5)}")
        
        self.test_clean_biasing_list = None
        self.test_other_biasing_list = None
        if is_predefined:
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
                        biasing_list[uid] = context_rare_words

                        ref_rare_words = ast.literal_eval(ref_rare_words)
                        ref_text = ref_text.split()
                        all_cnt += len(ref_text)
                        rare_cnt += len(ref_rare_words)
                return biasing_list, rare_cnt / all_cnt

            self.test_clean_biasing_list, ratio_clean = \
                read_ref_biasing_list(self.path_is21_deep_bias / f"ref/test-clean.biasing_{n_distractors}.tsv")
            self.test_other_biasing_list, ratio_other = \
                read_ref_biasing_list(self.path_is21_deep_bias / f"ref/test-other.biasing_{n_distractors}.tsv")

            logging.info(f"Number of utterances in test_clean_biasing_list: {len(self.test_clean_biasing_list)}, rare ratio={ratio_clean:.2f}")
            logging.info(f"Number of utterances in test_other_biasing_list: {len(self.test_other_biasing_list)}, rare ratio={ratio_other:.2f}")

        self.all_words2pieces = None
        if self.sp is not None:
            all_words2pieces = sp.encode(self.all_words, out_type=int)  # a list of list of int
            self.all_words2pieces = {w: pieces for w, pieces in zip(self.all_words, all_words2pieces)}
            logging.info(f"len(self.all_words2pieces)={len(self.all_words2pieces)}")

        self.all_words2embeddings = None
        if self.bert_encoder is not None:
            all_words = list(chain(self.common_words, self.rare_words))
            all_embeddings = self.bert_encoder.encode_strings(all_words)
            assert len(all_words) == len(all_embeddings)
            self.all_words2embeddings = {w: ebd for w, ebd in zip(all_words, all_embeddings)}
            logging.info(f"len(self.all_words2embeddings)={len(self.all_words2embeddings)}")
        
        if is_predefined:
            new_words_bias = set()
            all_words_bias = set()
            for uid, wlist in chain(self.test_clean_biasing_list.items(), self.test_other_biasing_list.items()):
                for word in wlist:
                    if word not in self.common_words and word not in self.rare_words:
                        new_words_bias.add(word)
                    all_words_bias.add(word)
            # if self.all_words2pieces is not None and word not in self.all_words2pieces:
            #     self.all_words2pieces[word] = self.sp.encode(word, out_type=int)
            # if self.all_words2embeddings is not None and word not in self.all_words2embeddings:
            #     self.all_words2embeddings[word] = self.bert_encoder.encode_strings([word])[0]
            logging.info(f"OOVs in the biasing list: {len(new_words_bias)}/{len(all_words_bias)}")
            if len(new_words_bias) > 0:
                self.add_new_words(list(new_words_bias), silent=True)

        if is_predefined:
            assert self.ratio_distractors is None
            assert self.n_distractors in [100, 500, 1000, 2000]

        if self.confusionp_path is not None:
            # self.text_perturbator = TextPerturbator(self.sp, self.confusionp_path)
            self.text_perturbator = TextPerturbator2()
        else:
            self.text_perturbator = None

        self.temp_dict = None
        self.temp_rare_words = None
        self.gt_rare_words_indices = None
        self.unmerged_rare_words_list = None

    def add_new_words(self, new_words_list, return_dict=False, silent=False):
        if len(new_words_list) == 0:
            if return_dict is True:
                return dict()
            else:
                return
        
        new_words_list = list(set(new_words_list))

        if self.all_words2pieces is not None:
            words_pieces_list = self.sp.encode(new_words_list, out_type=int)
            new_words2pieces = {w: pieces for w, pieces in zip(new_words_list, words_pieces_list)}
            if return_dict:
                return new_words2pieces
            else:
                self.all_words2pieces.update(new_words2pieces)
        
        if self.all_words2embeddings is not None:
            embeddings_list = self.bert_encoder.encode_strings(new_words_list, silent=silent)
            new_words2embeddings = {w: ebd for w, ebd in zip(new_words_list, embeddings_list)}
            if return_dict:
                return new_words2embeddings
            else:
                self.all_words2embeddings.update(new_words2embeddings)
        
        self.all_words.extend(new_words_list)
        self.rare_words.update(new_words_list)

    def discard_some_common_words(words, keep_ratio):
        pass

    def remove_common_words_from_texts(self, texts):
        return [" ".join([word for word in text.split() if word not in self.common_words]) for text in texts]

    def _get_random_word_lists(self, batch):
        texts = batch["supervisions"]["text"]

        new_words = []
        rare_words_list = []
        for i, text in enumerate(texts):
            rare_words = []
            if self.temp_rare_words is not None:
                rare_words.extend(self.temp_rare_words[i])
                new_words.extend(rare_words)
                self.temp_rare_words[i] = set(self.temp_rare_words[i])
            
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
                new_rare_words = [wx for wx, wxi in zip(rare_words, x) if (self.temp_rare_words is not None and wx in self.temp_rare_words[i]) or wxi < self.keep_ratio]
                # for xi in range(len(rare_words)):
                #     if x[xi] < self.keep_ratio:
                #         new_rare_words.append(rare_words[xi])
                rare_words = new_rare_words

            rare_words_list.append(rare_words)
            if self.temp_rare_words is not None:
                self.temp_rare_words[i] = rare_words  # just to see it from outside
        
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
            self.rare_words_list, 
            distractors_cnt,
        )  # TODO: actually the context should contain both rare and common words
        # distractors = random.choices(  # random choices with replacement
        #     self.rare_words, 
        #     distractors_cnt,
        # )

        # Do some perturbation here on the distractor words
        if self.text_perturbator is not None:
            # TODO: be careful: this is only applicable for librispeech (uppercase)
            replacement_set = string.ascii_uppercase + "'      "  # give it more chance to break the word
            distractors = self.text_perturbator.random_perturb_distractors(distractors, prob=0.05, replacement_set=replacement_set)

        distractors_pos = 0
        for i, rare_words in enumerate(rare_words_list):
            rare_words.extend(distractors[distractors_pos: distractors_pos + n_distractors_each[i]])
            distractors_pos += n_distractors_each[i]

            if i < len(rare_words_list) - 1 and distractors_pos >= len(distractors):
                distractors += random.sample(self.rare_words_list, int(sum(n_distractors_each[i:]) * 1.2))

            # random.shuffle(rare_words)
            # logging.info(rare_words)
        # assert distractors_pos <= len(distractors)  # as we do `random_perturb_distractors`, the number of words may decrease, so this is not guaranteed to be true now. But it is still very likely to be true

        return rare_words_list

    def _get_predefined_word_lists(self, batch):
        rare_words_list = []
        for cut in batch['supervisions']['cut']:
            uid = cut.supervisions[0].id
            if uid in self.test_clean_biasing_list:
                rare_words_list.append(self.test_clean_biasing_list[uid])
            elif uid in self.test_other_biasing_list:
                rare_words_list.append(self.test_other_biasing_list[uid])
            else:
                rare_words_list.append([])
                logging.error(f"uid={uid} cannot find the predefined biasing list of size {self.n_distractors}")
        # for wl in rare_words_list:
        #     for w in wl:
        #         if w not in self.all_words2pieces:
        #             self.all_words2pieces[w] = self.sp.encode(w, out_type=int)
        return rare_words_list

    def get_context_word_list(
        self,
        batch: dict,
    ):
        """
        Generate/Get the context biasing list as a list of words for each utterance
        Use keep_ratio to simulate the "imperfect" context which may not have 100% coverage of the ground truth words.
        """
        if self.is_predefined:
            rare_words_list = self._get_predefined_word_lists(batch)
        else:
            rare_words_list = self._get_random_word_lists(batch)
        
        # # get the indices of the gt rare words
        # gt_rare_words_indices = []
        # for j, text in enumerate(batch["supervisions"]["text"]):
        #     my_dict = {w: i for i, w in enumerate(rare_words_list[j])}
        #     gt_rare_words_indices.append([my_dict[w] for w in text.split() if w in my_dict])
        # self.gt_rare_words_indices = gt_rare_words_indices  # Don't forget to "add one" when using it in the biasing module
        # self.unmerged_rare_words_list = rare_words_list

        if self.all_words2embeddings is None:
            # Use SentencePiece to encode the words
            rare_words_pieces_list = []
            max_pieces_len = 0
            for rare_words in rare_words_list:
                rare_words_pieces = [self.all_words2pieces[w] if w in self.all_words2pieces else self.sp.encode(w, out_type=int) for w in rare_words]
                if len(rare_words_pieces) > 0:
                    max_pieces_len = max(max_pieces_len, max(len(pieces) for pieces in rare_words_pieces))
                rare_words_pieces_list.append(rare_words_pieces)
        else:  
            # Use BERT embeddings here
            rare_words_embeddings_list = []
            for rare_words in rare_words_list:
                # for w in rare_words:
                #     if w not in self.all_words2embeddings and (self.temp_dict is not None and w not in self.temp_dict):
                #         import pdb; pdb.set_trace()
                #     if w == "STUBE":
                #         import pdb; pdb.set_trace()
                rare_words_embeddings = [self.all_words2embeddings[w] if w in self.all_words2embeddings else self.temp_dict[w] for w in rare_words]
                rare_words_embeddings_list.append(rare_words_embeddings)

        if self.all_words2embeddings is None:
            # Use SentencePiece to encode the words
            word_list = []
            word_lengths = []
            num_words_per_utt = []
            pad_token = 0
            for rare_words_pieces in rare_words_pieces_list:
                num_words_per_utt.append(len(rare_words_pieces))
                word_lengths.extend([len(pieces) for pieces in rare_words_pieces])

                # # TODO: this is a bug here: this will effectively modify the entries in 'self.all_words2embeddings'!!!
                # for pieces in rare_words_pieces:
                #     pieces += [pad_token] * (max_pieces_len - len(pieces))
                # word_list.extend(rare_words_pieces)

                # Correction:
                rare_words_pieces_padded = list()
                for pieces in rare_words_pieces:
                    rare_words_pieces_padded.append(pieces + [pad_token] * (max_pieces_len - len(pieces)))
                word_list.extend(rare_words_pieces_padded)

            word_list = torch.tensor(word_list, dtype=torch.int32)
            # word_lengths = torch.tensor(word_lengths, dtype=torch.int32)
            # num_words_per_utt = torch.tensor(num_words_per_utt, dtype=torch.int32)
        else:
            # Use BERT embeddings here
            word_list = []
            word_lengths = None
            num_words_per_utt = []
            for rare_words_embeddings in rare_words_embeddings_list:
                num_words_per_utt.append(len(rare_words_embeddings))
                word_list.extend(rare_words_embeddings)
            word_list = torch.stack(word_list)

        return word_list, word_lengths, num_words_per_utt

    def get_context_word_wfst(
        self,
        batch: dict,
    ):
        """
        Get the WFST representation of the context biasing list as a list of words for each utterance
        """
        if self.is_predefined:
            rare_words_list = self._get_predefined_word_lists(batch)
        else:
            rare_words_list = self._get_random_word_lists(batch)
        
        # TODO:
        # We can associate weighted or dynamic weights for each rare word or token

        nbest_size = 1  # TODO: The maximum number of different tokenization for each lexicon entry.

        # Use SentencePiece to encode the words
        rare_words_pieces_list = []
        num_words_per_utt = []
        for rare_words in rare_words_list:
            rare_words_pieces = [self.all_words2pieces[w] if w in self.all_words2pieces else self.temp_dict[w] for w in rare_words]
            rare_words_pieces_list.append(rare_words_pieces)
            num_words_per_utt.append(len(rare_words))

        fsa_list, fsa_sizes = generate_context_graph_nfa(
            words_pieces_list = rare_words_pieces_list, 
            backoff_id = self.backoff_id, 
            sp = self.sp,
        )

        return fsa_list, fsa_sizes, num_words_per_utt


class TextPerturbator:
    def __init__(self, sp, confusionp_path):
        self.sp = sp
        self.confusionp_path = confusionp_path
        self.confusionp = self.load_confusion_p(confusionp_path, sp, k=10)
        self.proxy_fst = self.get_proxy_transducer(self.confusionp, sp, num_edits=1)
        self.proxy_fst = k2.arc_sort(self.proxy_fst)

    def load_confusion_p(self, filename, sp, k=10):
        # Only allow very top-ranked subsitution pairs
        # Disallow insertion and deletions

        with open(filename, "r") as fin:
            lines = fin.readlines()
        lines = [line.strip().split() for line in lines]
        confusionp = [(line[0], line[1], float(line[2])) for line in lines]

        confusionp_dict = defaultdict(dict)
        for wp1, wp2, p in confusionp:
            confusionp_dict[wp1][wp2] = p
        
        # only choose top-k
        top_k_confusionp_dict = {wp1: {key: value for key, value in sorted(wp2_p.items(), key=lambda item: item[1], reverse=True)[:k]} for wp1, wp2_p in confusionp_dict.items()}
        confusionp = [(wp1, wp2, p) for wp1, wp2_p in top_k_confusionp_dict.items() for wp2, p in wp2_p.items()]

        # confusionp = [(0 if wp1 == "*" else sp.piece_to_id(wp1), 0 if wp2 == "*" else sp.piece_to_id(wp2), p) for wp1, wp2, p in confusionp]
        confusionp = [(sp.piece_to_id(wp1), sp.piece_to_id(wp2), p) for wp1, wp2, p in confusionp if wp1 != "*" and wp2 != "*"]
        logging.info(f"len(confusionp) = {len(confusionp)}")
        return confusionp

    def get_proxy_transducer(self, confusionp, sp, num_edits=1):
        arcs = []

        start_state = 0
        for i in range(1, sp.get_piece_size()):
        # for i in range(1, 5):
            arcs.append((start_state, start_state, i, i, 0))
        
        prev_state = start_state
        next_state = start_state
        for i_edit in range(num_edits):
            prev_state = next_state
            next_state += 1
            for wp1, wp2, p in confusionp:
            # for wp1, wp2, p in confusionp[:5]:
                arcs.append((prev_state, next_state, wp1, wp2, p))

        prev_state = next_state
        next_state += 1
        for i in range(1, sp.get_piece_size()):
        # for i in range(1, 5):
            arcs.append((prev_state, next_state, i, i, 0))
            arcs.append((next_state, next_state, i, i, 0))
        
        final_state = next_state + 1
        arcs.append((next_state, final_state, -1, -1, 0))
        arcs.append([final_state])

        new_arcs = arcs
        new_arcs = sorted(new_arcs, key=lambda arc: arc[0])
        new_arcs = [[str(i) for i in arc] for arc in new_arcs]
        new_arcs = [" ".join(arc) for arc in new_arcs]
        new_arcs = "\n".join(new_arcs)

        fst = k2.Fsa.from_str(new_arcs, acceptor=False)
        return fst

    def get_my_words(self, word_list, first_k=4):
        my_words = []
        for word in word_list:
            if len(my_words) == 0 or my_words[-1] != word:
                my_words.append(word)
        # print(my_words)
        if len(my_words) > 0:
            return_word = random.choice(my_words[:first_k])
            return_word = return_word.replace(' ', '')
            return return_word
        else:
            return None

    def get_proxies(self, word_list, nbest_scale=0.5, num_paths=100, first_k=4):
        kw = [w.upper() for w in word_list]  # Be careful! This has to be consistent with the bpe model
        kw = self.sp.encode(kw, out_type=int)

        kw_fst = k2.linear_fsa(kw)
        kw_fst.aux_labels = kw_fst.labels.clone()

        G1 = k2.arc_sort(kw_fst)
        # G2 = k2.arc_sort(self.proxy_fst)
        G2 = self.proxy_fst  # already arc-sorted
        G1G2 = k2.compose(G1, G2)
        G1G2 = k2.connect(G1G2)
        # G1G2.shape, G1G2.num_arcs

        G1G2.scores *= nbest_scale
        paths = k2.random_paths(
            G1G2, num_paths=num_paths, use_double_scores=False
        )

        token_ids = k2.ragged.index(G1G2.aux_labels, paths)
        my_candidates = [self.sp.decode([tks[:-1] for tks in token_ids[i].tolist()]) for i in range(len(word_list))]
        hyps_list = [self.get_my_words(my_candidates[i], first_k=first_k) for i in range(len(word_list))]
        hyps_list = [w2 if w2 is not None else w1 for w1, w2 in zip(word_list, hyps_list)]
        return hyps_list

    def perturb_texts(self, texts, common_words, prob=0.6) -> str:
        # p = np.random.rand(len(texts)) < prob

        all_rare_words = list()
        for text in texts:
            rare_words = [word for word in text.split() if word not in common_words]
            all_rare_words.append(rare_words)
        
        _all_rare_words = [w for rare_words in all_rare_words for w in rare_words if random.random() < prob]
        _all_rare_words_proxies = self.get_proxies(_all_rare_words, nbest_scale=0.8, num_paths=50, first_k=4)
        _all_rare_words = {w1: w2 for w1, w2 in zip(_all_rare_words, _all_rare_words_proxies)}  # a mapping of old => new

        new_texts = list()
        new_rare_words = list()
        for text in texts:
            text_split = text.split()
            new_texts.append(" ".join([_all_rare_words.get(w, w) for w in text_split]))
            new_rare_words.append([_all_rare_words[w] for w in text_split if w in _all_rare_words])

        return new_texts, new_rare_words


class TextPerturbator2:
    def __init__(self):
        rules = """
a aa
aa ar
a ae
a e
a o
a ei
a ay
a ey
ay ey
ai ay
ai ei
an en
an in
an on
ar er
ar er
ar or
ar our
ar ur
au oo
au u
au o
au aw
as es
at ad
al el
al ol
b p
b bh
c_ ck_
c_ k_
c s
ce se
ch_ ck_
ch_ k_
ch j
ch sh
ch tch
con com
d dh
t dh
d t
d tt
d dd
di dee
dis this
ear eer
em en
en an
e io
e eo
e eu
ew eu
aw au
ee i
ee ea
ee y
er ar
er ir
er ir
er ur
er or
es is
ew oo
ew io
ew iu
el ol
f ff
f ph
f v
g gg
ght_ t_
h hh
i e
i ea
i ie
i ey
ie ee
ie y
io eo
igh ie
igh y
ii i
in an
in een
in en
in en
in ing
ir ur
_j _g
j dj
k ck
la le
ll l
mm m
_n _kn
nn n
o oe
o oe
o ow
o ol
o ar
al ol
el ol
al el
oi oy
oo ew
oo ou
oo u
oo ue
or ar
or er
or ow
or ur
ou ow
ough uff
ow oa
ow ou
p b
p pp
ph f
q k
q ck
que_ k_
que_ ck_
r rh
re ri
s th
s ts
s tz
s z
s ss
s dz
sc xg
sa se
sh ch
sh j
stle_ so_
t d
t tt
th d
th_ f_
th_ s_
th_ z_
th v
tch ch
tch sh
tion shion
tion sion
tr dr
ue oo
ur ir
u iu
u eo
u ew
v w
w wh
x ks
x s
z ts
z tz
z zz
"""

        rules = rules.upper().strip().split("\n")  # upper case for librispeech
        rules = [rule.strip() for rule in rules]

        rules_dict = defaultdict(set)
        for r in rules:
            rules_dict[r.split()[0]].add(r.split()[1])
            rules_dict[r.split()[1]].add(r.split()[0])
        self.rules_dict = {k: list(v) for k, v in rules_dict.items()}
        self.rules_regex = "|".join(sorted(map(re.escape,  self.rules_dict.keys()), key=len, reverse=True))

        import nltk
        self.pos_tagger = nltk.tag.PerceptronTagger()

    def perturb_one_word0(self, s):
        s = f"_{s}_"
        appearances = [sub for sub in self.rules_dict.keys() if sub in s]
        if len(appearances) == 0:
            return s
        
        # This is to find `maximum` matched pattern
        max_len = max([len(sub) for sub in appearances])
        appearances = [sub for sub in appearances if len(sub) == max_len]

        # However, `maximal` matched pattern may be what we need, e.g., if "ie" is matched, we won't consider "i" or "e" anymore
        
        pattern = random.choice(appearances)
        target = random.choice(self.rules_dict[pattern])
        # n_occur = len(re.findall(f'(?={pattern})', s))
        # random.randint(0, n_occur-1)
        s = s.replace(pattern, target, 1)
        s = s[1:-1]
        return s

    def perturb_one_word(self, s):
        # This implementation uses re to find "maximal" matched patterns

        s = f"_{s}_"
        appearances = [(m.group(), m.start()) for m in re.finditer(self.rules_regex, s)]
        if len(appearances) == 0:
            return s

        # In regular expressions, the engine will always try to make the match as large as possible when using quantifiers like *, +, ?, and {m,n}. This is called "greedy" matching.
        # However, in the context of matching multiple different substrings with the | operator, the re module in Python uses a "first-come" strategy. It will return the first match it finds, even if a later substring would produce a longer match.
        # To ensure "maximal" matches in this context, you can sort the substrings by length in descending order before joining them into a regular expression. This way, longer substrings will be matched first.

        # test case:
        # substring_set = {"y", "o", "yo", "ng"}
        # s = "your long string"
        
        pattern, pos = random.choice(appearances)
        target = random.choice(self.rules_dict[pattern])
        s = s[:pos] + target + s[pos + len(pattern):]
        s = s[1:-1]
        return s

    def perturb_texts(self, texts, common_words=[], prob=0.6) -> str:
        # p = np.random.rand(len(texts)) < prob

        # TODO: <1> is this necessary?
        # pos_tags = [self.pos_tagger.tag(text.lower().split()) for text in texts]
        pos_tags = [[(w, "NN") for w in text.lower().split()] for text in texts]

        # all_rare_words = list()
        # for text in texts:
        #     rare_words = [word for word in text.split() if word not in common_words]
        #     all_rare_words.append(rare_words)

        all_rare_words = list()
        for pos_tag in pos_tags:
            rare_words = [word.upper() for word, tag in pos_tag if tag.startswith("NN") and word.upper() not in common_words]
            all_rare_words.append(rare_words)

        _all_rare_words = [w for rare_words in all_rare_words for w in rare_words if random.random() < prob]
        _all_rare_words_proxies = [self.perturb_one_word(w) for w in _all_rare_words]
        _all_rare_words = {w1: w2 for w1, w2 in zip(_all_rare_words, _all_rare_words_proxies)}  # a mapping of old => new

        new_texts = list()
        new_rare_words = list()
        for text in texts:
            text_split = text.split()
            new_texts.append(" ".join([_all_rare_words.get(w, w) for w in text_split]))
            new_rare_words.append([_all_rare_words[w] for w in text_split if w in _all_rare_words])

        return new_texts, new_rare_words

    def random_perturb_distractors(self, words, prob=0.05, replacement_set=string.ascii_lowercase):
        # We will synthesize some OOVs here
        # `words` is a list of words

        text = " ".join(words)
        text = list(text)
        num_replacements = int(len(words) * prob)
        indices = random.sample(range(len(text)), num_replacements)

        # Replace the chosen characters with random letters
        for index in indices:
            text[index] = random.choice(replacement_set)
        
        text = ''.join(text)
        return text.split()