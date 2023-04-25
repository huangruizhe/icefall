import torch
import random
from pathlib import Path
import sentencepiece as spm
from typing import List
import logging
import ast
import numpy as np
from itertools import chain
from word_encoder_bert import BertEncoder
from context_wfst import generate_context_graph_nfa

class ContextCollector(torch.utils.data.Dataset):
    def __init__(
        self, 
        path_rare_words: Path,
        sp: spm.SentencePieceProcessor,
        bert_encoder: BertEncoder = None,
        n_distractors: int = 100,
        ratio_distractors: int = None,
        is_predefined: bool = False,
        keep_ratio: float = 1.0,
        is_full_context: bool = False,
        backoff_id: int = None,
        slides: Path = None,
    ):
        self.sp = sp
        self.bert_encoder = bert_encoder
        self.path_rare_words = path_rare_words
        self.n_distractors = n_distractors
        self.ratio_distractors = ratio_distractors
        self.is_predefined = is_predefined
        self.keep_ratio = keep_ratio
        self.is_full_context = is_full_context   # use all words (rare or common) in the context
        # self.embedding_dim = self.bert_encoder.bert_model.config.hidden_size
        self.backoff_id = backoff_id
        self.slides = slides

        logging.info(f"""
            n_distractors={n_distractors},
            ratio_distractors={ratio_distractors},
            is_predefined={is_predefined},
            keep_ratio={keep_ratio},
            is_full_context={is_full_context},
            bert_encoder={bert_encoder.name if bert_encoder is not None else None},
            slides={slides},
        """)

        self.common_words = []
        self.rare_words = []
        self.all_words = []
        self.all_words2embeddings = None
        self.all_words2pieces = dict()
        self.ec53_biasing_list = None
        self.cached_ec53_wfst = dict()
        self.cached_ec53_embeddings = dict()

        with open(path_rare_words / "all_rare_words.txt", "r") as fin:
            self.rare_words = [l.strip() for l in fin if len(l) > 0]
        
        # with open(path_rare_words / "common_words_6k.txt", "r") as fin:
        with open(path_rare_words / "common_words_3k.txt", "r") as fin:
            self.common_words = [l.strip() for l in fin if len(l) > 0]
        
        self.all_words = self.rare_words + self.common_words  # sp needs a list of strings, can't be a set
        self.common_words = set(self.common_words)
        self.rare_words = set(self.rare_words)

        logging.info(f"Number of common words: {len(self.common_words)}. Examples: {random.sample(self.common_words, 5)}")
        logging.info(f"Number of rare words: {len(self.rare_words)}. Examples: {random.sample(self.rare_words, 5)}")
        logging.info(f"Number of all words: {len(self.all_words)}. Examples: {random.sample(self.all_words, 5)}")

        if is_predefined and slides is not None:
            self.ec53_biasing_list = dict()

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
            
            from glob import glob
            import pandas as pd
            import numpy as np

            new_words = set()                

            biasing_files = glob(f"{slides}/*.txt")
            for filename in biasing_files:
                biasing_list = read_ref_biasing_list(filename)
                biasing_list2 = {word: weight for word, weight in biasing_list.items() if word not in self.common_words}
                # biasing_list2 = {word: weight for word, weight in biasing_list.items() if word in self.all_words}  # only contain "known" words

                base_name = filename.split("/")[-1]
                base_name = base_name.replace(".txt", "")  # one ec

                logging.info(f"EC {base_name} biasing list size: {len(biasing_list)} -> keep rare only: {len(biasing_list2)}")
                self.ec53_biasing_list[base_name] = biasing_list2
                new_words.update(biasing_list2.keys())
            
            logging.info(f"Number of ECs in EC53 biasing lists: {len(self.ec53_biasing_list)}")
            
            df_describe = pd.DataFrame([len(l) for l in self.ec53_biasing_list.values()])
            logging.info(f"Biasing lists stats: {str(df_describe.describe())}")

            new_words = list(new_words)
            logging.info(f"Number of words from predefined biasing lists: {len(new_words)}, Example: {random.sample(new_words, 5)}")
            self.add_new_words(new_words)

            _all_words = set(self.all_words)
            oovs = [w for w in new_words if w not in _all_words]
            logging.info(f"Number of OOVs from predefined biasing lists: {len(oovs)}, Example: {random.sample(oovs, min(5, len(oovs)))}")

            self.word_analysis(oovs)

        if self.sp is not None:
            all_words2pieces = sp.encode(self.all_words, out_type=int)  # a list of list of int
            all_words2pieces = {w: pieces for w, pieces in zip(self.all_words, all_words2pieces)}
            self.all_words2pieces.update(all_words2pieces)
            logging.info(f"len(self.all_words2pieces)={len(self.all_words2pieces)}")

        self.temp_dict = None
        self.temp_rare_words_list = None

    def word_analysis(self, oovs):
        file_oovs = "/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/oovs.txt"
        with open(file_oovs, "w") as fout:
            for w in set(oovs):
                print(w, file=fout)
        logging.info(f"There are {len(oovs)} OOVs.")
        
        from lhotse import CutSet

        ec53_cuts_file = "/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_ec53_norm.jsonl.gz"
        logging.info(f"Loading cuts from: {ec53_cuts_file}")
        ec53_cuts = CutSet.from_file(ec53_cuts_file)
        gt_words = set()
        for c in ec53_cuts:
            gt_words.update(c.supervisions[0].text.split())
        logging.info(f"There are {len(gt_words)} unique GT words.")
        
        file_oovs_in_gt = "/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/oovs_in_gt.txt"
        cnt = 0
        with open(file_oovs_in_gt, "w") as fout:
            for w in set(oovs):
                if w in gt_words:
                    cnt += 1
                    print(w, file=fout)
        logging.info(f"There are {cnt} OOVs in GT.")

        context_words = set()
        for base_name, blist in self.ec53_biasing_list.items():
            context_words.update(blist.keys())
        cnt = 0
        cnt_oov = 0
        oovs = set(oovs)
        for w in context_words:
            if w in gt_words:
                cnt += 1
                if w in oovs:
                    cnt_oov += 1
        logging.info(f"There are {cnt} unique rare words in the slides spoken in GT, where {cnt_oov} are OOVs.")

    def add_new_words(self, new_words_list, return_dict=False, silent=False):
        if len(new_words_list) == 0:
            if return_dict is True:
                return dict()
            else:
                return
        
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
        
        # self.all_words.extend(new_words_list)
        # self.rare_words.update(new_words_list)

    def discard_some_common_words(words, keep_ratio):
        pass

    def _get_random_word_lists(self, batch):
        if "text" in batch["supervisions"]:
            texts = batch["supervisions"]["text"]  # For training
        else:
            texts = [cut.supervisions[0].text for cut in batch['supervisions']['cut']]  # For decoding ec53

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
                rare_words = random.sample(rare_words, int(len(rare_words) * self.keep_ratio))

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
            self.all_words,  # self.rare_words, 
            distractors_cnt
        )  # TODO: actually the context should contain both rare and common words
        # distractors = random.choices(  # random choices with replacement
        #     self.rare_words, 
        #     distractors_cnt,
        # )
        distractors_pos = 0
        for i, rare_words in enumerate(rare_words_list):
            rare_words.extend(distractors[distractors_pos: distractors_pos + n_distractors_each[i]])
            distractors_pos += n_distractors_each[i]
            # random.shuffle(rare_words)
            # logging.info(rare_words)
        assert distractors_pos == len(distractors)

        return rare_words_list

    def _uid_2_ecid(self, uid):
        ec_id = uid.split("_")[:-2]
        ec_id = "_".join(ec_id)
        return ec_id
    
    def biasing_list_downsample_oracle(self, biasing_list, text):
        if len(biasing_list) < 700:
            return biasing_list

        text = set(text.split())
        
        gt_words = [w for w in biasing_list if w in text]
        other_words = [w for w in biasing_list if w not in text]

        sample_size = random.randint(200, 700)
        distractors = random.sample(
            other_words,
            sample_size
        )
        return gt_words + distractors


    def _get_predefined_word_lists(self, batch):
        rare_words_list = []
        for cut in batch['supervisions']['cut']:
            uid = cut.supervisions[0].id
            ec_id = self._uid_2_ecid(uid)
            
            if ec_id in self.ec53_biasing_list:
                biasing_list = list(self.ec53_biasing_list[ec_id].keys())
                # biasing_list = self.biasing_list_downsample_oracle(biasing_list, cut.supervisions[0].text)  # TODO
                rare_words_list.append(biasing_list)
            else:
                rare_words_list.append([])
                logging.error(f"uid={uid} cannot find the predefined biasing list")
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
        self.temp_rare_words_list = rare_words_list
        
        if self.all_words2embeddings is None:
            # Use SentencePiece to encode the words
            rare_words_pieces_list = []
            max_pieces_len = 0
            for rare_words in rare_words_list:
                rare_words_pieces = [self.all_words2pieces[w] if w in self.all_words2pieces else self.temp_dict[w] for w in rare_words]
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

        uid_list = [cut.supervisions[0].id for cut in batch['supervisions']['cut']]
        ec_id_list = [self._uid_2_ecid(uid) for uid in uid_list]
        fsa_list = []
        fsa_sizes = []
        for ec_id, rare_words_pieces in zip(ec_id_list, rare_words_pieces_list):
            if ec_id in self.cached_ec53_wfst:
                fsa, fsa_size = self.cached_ec53_wfst[ec_id]
            else:
                fsa, fsa_size = generate_context_graph_nfa(
                    words_pieces_list = [rare_words_pieces], 
                    backoff_id = self.backoff_id, 
                    sp = self.sp,
                )
                logging.info(f"Cached contextual WFST for EC {ec_id}, size: {fsa_size}")
                fsa = fsa[0]
                fsa_size = fsa_size[0]
                self.cached_ec53_wfst[ec_id] = (fsa, fsa_size)
            fsa_list.append(fsa)
            fsa_sizes.append(fsa_size)

        return fsa_list, fsa_sizes, num_words_per_utt
