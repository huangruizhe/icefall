import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from collections import defaultdict
import pickle

import psutil
import torch.multiprocessing as mp
try:
    from tqdm_loggable.auto import tqdm
except:
    from tqdm import tqdm

import k2
import optim
import sentencepiece as spm
import torch
import torch.multiprocessing as mp
import torch.nn as nn

def get_uid_key(my_id):
    # /data/skhudan1/corpora/librispeech/CHAPTERS.TXT
    speaker_id, chapter_id, utterance_id, _ = my_id.split("-")
    speaker_id, chapter_id, utterance_id = int(speaker_id), int(chapter_id), int(utterance_id)
    return speaker_id, chapter_id, utterance_id


def convert_long_text_to_fst(items, sp, pid, results):
    libri_long_text_sp = dict()
    for k, text in tqdm(items, mininterval=2, desc=f"libri_long_text [{pid}]"):
        # libri_long_text_sp[k] = make_factor_transducer1(sp.encode(text, out_type=int), return_str=True, blank_penalty=0)
        # libri_long_text_sp[k] = make_factor_transducer2(sp.encode(text, out_type=int), return_str=True, blank_penalty=-12)
        # libri_long_text_sp[k] = make_factor_transducer3(sp.encode(text, out_type=int), word_start_symbols=[i for i in range(sp.vocab_size()) if sp.id_to_piece(i).startswith('▁')], return_str=True, blank_penalty=0)
        libri_long_text_sp[k] = make_factor_transducer4(sp.encode(text, out_type=int), word_start_symbols=[i for i in range(sp.vocab_size()) if sp.id_to_piece(i).startswith('▁')], return_str=True, blank_penalty=0)
    results[pid] = libri_long_text_sp


def get_long_text(cuts, sp=None, make_fst=False):
    logging.info(f"Getting long text from cuts ... ")  # len(cuts) = {len(cuts)}
    cuts_by_recoding = defaultdict(list)
    for i, c in enumerate(cuts):  # tqdm(cuts, miniters=1000, total=None):
        if "_sp" in c.id:
            continue
        cuts_by_recoding[tuple(get_uid_key(c.id)[:2])].append(c)

        # if i % 1e4 == 0:
        #     print(f"progress: {i}")

    libri_long_text = dict()
    for k, v in cuts_by_recoding.items():
        v.sort(key = lambda x: get_uid_key(x.id)[-1])
        text = " ".join([c.supervisions[0].text for c in v])
        libri_long_text[k] = text
    
    if sp is None:
        return libri_long_text, None

    logging.info(f"Converting long text to fst ... ")

    if not make_fst:
        libri_long_text_sp = dict()
        for k, text in libri_long_text.items():
            libri_long_text_sp[k] = sp.encode(text, out_type=int)
    else:
        processes = []
        manager = mp.Manager()
        # Fork processes
        n_process = 6
        items = list(libri_long_text.items())
        chunk_size = int(len(items) / n_process) + 1
        i_chunk = 0
        results = manager.list([0] * n_process)
        for i in range(0, len(items), chunk_size):
            chunk = items[i: i+chunk_size]
            fork = mp.Process(target=convert_long_text_to_fst,
                            args=(chunk, sp, i_chunk, results))
            fork.start()
            processes.append(fork)
            i_chunk += 1
        # Wait until all processes are finished
        for fork in processes:
            fork.join()
        
        libri_long_text_sp = dict()
        for rs in results:
            libri_long_text_sp.update(rs)
        for k, v in tqdm(libri_long_text_sp.items()):
            libri_long_text_sp[k] = k2.Fsa.from_str(v, acceptor=False)
    
    ram_info = psutil.virtual_memory()
    ram_used_mb = ram_info.used / (1024 ** 2)  # Convert bytes to megabytes
    ram_total_mb = ram_info.total / (1024 ** 2)  # Convert bytes to megabytes
    ram_usage = ram_info.percent
    logging.info(f"Current RAM Usage: {ram_used_mb:.2f} MB out of {ram_total_mb:.2f} MB ({ram_usage}%)")

    return libri_long_text, libri_long_text_sp


def make_factor_transducer1(word_id_list, return_str=False, blank_penalty=0):
    # This is the original, simplest factor transducer for a "linear" fst

    fst_graph = k2.ctc_graph([word_id_list], modified=False, device='cpu')[0]

    c_str = k2.to_str_simple(fst_graph)
    arcs = c_str.strip().split("\n")
    arcs = [x.strip() for x in arcs if len(x.strip()) > 0]
    final_state = int(arcs[-1])
    
    arcs = arcs[:-1]
    arcs = [tuple(map(int, a.split())) for a in arcs]
    # ss, ee, l1, l2, w = arc

    non_eps_nodes = set((arc[1], arc[3]) for arc in arcs if arc[3] > -1)   # if this node has a non-eps in-coming arc
    arcs += [(0, n, l, l, 0) for n, l in non_eps_nodes if n > 1]

    arcs += [(n, final_state, -1, -1, 0) for n in range(1, final_state - 2)]

    new_arcs = arcs
    new_arcs.append([final_state])

    new_arcs = sorted(new_arcs, key=lambda arc: arc[0])
    new_arcs = [[str(i) for i in arc] for arc in new_arcs]
    new_arcs = [" ".join(arc) for arc in new_arcs]
    new_arcs = "\n".join(new_arcs)

    if return_str:
        return new_arcs
    else:
        fst = k2.Fsa.from_str(new_arcs, acceptor=False)
        return fst


def make_factor_transducer2(word_id_list, return_str=False, blank_penalty=-1):
    # This is the factor transducer where blank symbols at the beginning and ending of the graph is penalized
    # Last resort: use a cheap alignment model to get a subgraph of the big graph first

    # blank_penalty should be negative

    fst_graph = k2.ctc_graph([word_id_list], modified=False, device='cpu')[0]

    c_str = k2.to_str_simple(fst_graph)
    arcs = c_str.strip().split("\n")
    arcs = [x.strip() for x in arcs if len(x.strip()) > 0]
    final_state = int(arcs[-1])
    
    arcs = arcs[:-1]
    arcs = [tuple(map(int, a.split())) for a in arcs]
    # ss, ee, l1, l2, w = arc

    arc0 = arcs[0]
    arcs_last = [a for a in arcs[-5:] if a[2] > 0]

    arcs = [(0, 0, 0, 0, blank_penalty)] + arcs[1:-5] + arcs_last

    non_eps_nodes = set((arc[1], arc[3]) for arc in arcs if arc[3] > 0)   # if this node has a non-eps in-coming arc
    arcs += [(0, n, l, l, 0) for n, l in non_eps_nodes if n > 1]

    # arcs += [(n, final_state, -1, -1, 0) for n in range(1, final_state - 2)]
    arcs += [(n, final_state - 1, 0, 0, blank_penalty) for n, l in non_eps_nodes]
    arcs += [(final_state - 1, final_state - 1, 0, 0, blank_penalty)]
    arcs += [(final_state - 1, final_state, -1, -1, 0)]

    new_arcs = arcs
    new_arcs.append([final_state])

    new_arcs = sorted(new_arcs, key=lambda arc: arc[0])
    new_arcs = [[str(i) for i in arc] for arc in new_arcs]
    new_arcs = [" ".join(arc) for arc in new_arcs]
    new_arcs = "\n".join(new_arcs)

    if return_str:
        return new_arcs
    else:
        fst = k2.Fsa.from_str(new_arcs, acceptor=False)
        return fst


def make_factor_transducer3(word_id_list, word_start_symbols, return_str=False, blank_penalty=0):
    # This is a modification of make_factor_transducer1, where the factors are on "word-level"
    # That is, the words always come as a whole

    fst_graph = k2.ctc_graph([word_id_list], modified=False, device='cpu')[0]

    c_str = k2.to_str_simple(fst_graph)
    arcs = c_str.strip().split("\n")
    arcs = [x.strip() for x in arcs if len(x.strip()) > 0]
    final_state = int(arcs[-1])
    
    arcs = arcs[:-1]
    arcs = [tuple(map(int, a.split())) for a in arcs]
    # ss, ee, l1, l2, w = arc

    non_eps_nodes = set((arc[1], arc[3]) for arc in arcs if arc[3] > 0 and arc[3] in word_start_symbols)   # if this node has a non-eps, word-start in-coming arc
    arcs += [(0, n, l, l, 0) for n, l in non_eps_nodes if n > 1]

    non_eps_nodes2 = set((arc[0], arc[3]) for arc in arcs if arc[3] > 0 and arc[3] in word_start_symbols)   # if this node has a non-eps, word-start out-going arc
    non_eps_nodes2 = [(n, l) for n, l in non_eps_nodes2 if 0 < n < final_state - 2]
    arcs += [(n, final_state, -1, -1, 0) for n, l in non_eps_nodes2]

    new_arcs = arcs
    new_arcs.append([final_state])

    new_arcs = sorted(new_arcs, key=lambda arc: arc[0])
    new_arcs = [[str(i) for i in arc] for arc in new_arcs]
    new_arcs = [" ".join(arc) for arc in new_arcs]
    new_arcs = "\n".join(new_arcs)

    if return_str:
        return new_arcs
    else:
        fst = k2.Fsa.from_str(new_arcs, acceptor=False)
        return fst


class MyCounter: 
    def __init__(self): self.counter1 = 0; self.counter2 = 1; self.counter2_ = 0
    # def __call__(self): self.counter += 1; return self.counter
    def f1(self): 
        self.counter1 += 1; return self.counter1
    def f2(self): 
        self.counter2_ += 1; 
        if self.counter2_ % 2 == 1: self.counter2 += 1; 
        return self.counter2


def make_factor_transducer4(word_id_list, word_start_symbols, return_str=False, blank_penalty=0):
    fst_graph = k2.ctc_graph([word_id_list], modified=False, device='cpu')[0]

    c_str = k2.to_str_simple(fst_graph)
    arcs = c_str.strip().split("\n")
    arcs = [x.strip() for x in arcs if len(x.strip()) > 0]
    final_state = int(arcs[-1])
    
    arcs = arcs[:-1]
    arcs = [tuple(map(int, a.split())) for a in arcs]
    # ss, ee, l1, l2, w = arc

    counter = MyCounter()

    non_eps_nodes1 = set((arc[1], arc[3]) for arc in arcs if arc[3] > 0 and arc[3] in word_start_symbols)   # if this node has a non-eps, word-start in-coming arc
    non_eps_nodes1 = sorted(non_eps_nodes1, key=lambda x: x[0])
    non_eps_nodes2 = list((arc[0], arc[1]) for arc in arcs if arc[3] < 0 or (arc[3] > 0 and arc[3] in word_start_symbols and arc[0] > 0))   # if this node has a non-eps, word-start out-going arc
    self_loops = {ss: l1 for ss, ee, l1, l2, w in arcs if ss == ee}

    arcs = [arcs[0]] + arcs[2:-5] + [a for a in arcs[-5:] if a[2] >= 0]
    arcs = [[ss, ee, l1, 0, w] for ss, ee, l1, l2, w in arcs]
    arcs += [(0, n, l, counter.f1(), 0) for n, l in non_eps_nodes1]
    arcs += [(n, final_state, self_loops[n], counter.f2(), 0) for n, l in non_eps_nodes2]
    arcs += [(final_state, final_state + 1, -1, -1, 0)]
    
    # non_eps_nodes2 = [(n, l) for n, l in non_eps_nodes2 if 0 < n < final_state - 2]
    
    new_arcs = arcs
    new_arcs.append([final_state + 1])

    new_arcs = sorted(new_arcs, key=lambda arc: arc[0])
    new_arcs = [[str(i) for i in arc] for arc in new_arcs]
    new_arcs = [" ".join(arc) for arc in new_arcs]
    new_arcs = "\n".join(new_arcs)

    # print(new_arcs)

    if return_str:
        return new_arcs
    else:
        fst = k2.Fsa.from_str(new_arcs, acceptor=False)
        return fst

def get_decoding_graphs(_texts):        
    fst_graph = k2.ctc_graph(sp.encode(_texts, out_type=int), modified=False, device='cpu')
    return [fst_graph[i] for i in range(fst_graph.shape[0])]

