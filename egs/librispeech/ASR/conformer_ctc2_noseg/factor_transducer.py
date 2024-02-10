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

import pywrapfst as openfst


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
        # libri_long_text_sp[k] = make_factor_transducer3(sp.encode(text, out_type=int), word_start_symbols={i for i in range(sp.vocab_size()) if sp.id_to_piece(i).startswith('▁')}, return_str=True, blank_penalty=0)
        # libri_long_text_sp[k] = make_factor_transducer4(sp.encode(text, out_type=int), word_start_symbols={i for i in range(sp.vocab_size()) if sp.id_to_piece(i).startswith('▁')}, return_str=True, blank_penalty=0)
        libri_long_text_sp[k] = make_factor_transducer4_skip(sp.encode(text, out_type=int), word_start_symbols={i for i in range(sp.vocab_size()) if sp.id_to_piece(i).startswith('▁')}, return_str=True, blank_penalty=0)
    results[pid] = libri_long_text_sp
    return libri_long_text_sp


def get_long_text(cuts, sp=None, make_fst=False, nj=6):
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
        if len(libri_long_text) > 100: 
            processes = []
            manager = mp.Manager()
            # Fork processes
            n_process = nj
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
        else:
            libri_long_text_sp = convert_long_text_to_fst(libri_long_text.items(), sp, 0, [None])

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


class WordCounter: 
    def __init__(self): 
        self.counter1 = 0 
        self.counter2 = 1; self.counter2_ = 0
        self.counter3 = 0
    # def __call__(self): self.counter += 1; return self.counter
    def f1(self): 
        self.counter1 += 1; return self.counter1
    def f2(self): 
        self.counter2_ += 1; 
        if self.counter2_ % 2 == 1: self.counter2 += 1; 
        return self.counter2
    def f3(self): 
        self.counter3 += 1; return self.counter3
    def c1(self): return self.counter1
    def c2(self): return self.counter2
    def c3(self): return self.counter3
    def reset(): 
        self.counter1 = 0
        self.counter2 = 1; self.counter2_ = 0
        self.counter3 = 0


def make_factor_transducer4(word_id_list, word_start_symbols, return_str=False, blank_penalty=0):
    # This is a modification of make_factor_transducer3, but we only output word indices instead of word symbols

    fst_graph = k2.ctc_graph([word_id_list], modified=False, device='cpu')[0]

    c_str = k2.to_str_simple(fst_graph)
    arcs = c_str.strip().split("\n")
    arcs = [x.strip() for x in arcs if len(x.strip()) > 0]
    final_state = int(arcs[-1])
    
    arcs = arcs[:-1]
    arcs = [tuple(map(int, a.split())) for a in arcs]
    # ss, ee, l1, l2, w = arc

    counter = WordCounter()

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


def make_factor_transducer4_bigram(word_id_list, word_start_symbols, return_str=False, blank_penalty=0):
    # This is a modification of make_factor_transducer4, but we just use a bigram graph instead of a factor transducer

    fst_graph = k2.ctc_graph([word_id_list], modified=False, device='cpu')[0]


def make_factor_transducer4_skip(word_id_list, word_start_symbols, return_str=False, blank_penalty=0):
    # This is a modification of make_factor_transducer4, but we allow skip arcs instead of a factor transducer

    fst_graph = k2.ctc_graph([word_id_list], modified=False, device='cpu')[0]

    c_str = k2.to_str_simple(fst_graph)
    arcs = c_str.strip().split("\n")
    arcs = [x.strip() for x in arcs if len(x.strip()) > 0]
    final_state = int(arcs[-1])

    arcs = arcs[:-1]
    arcs = [tuple(map(int, a.split())) for a in arcs]
    # ss, ee, l1, l2, w = arc

    counter = WordCounter()
    counter.f3()

    non_eps_nodes1 = set((arc[1], arc[3]) for arc in arcs if arc[3] > 0 and arc[3] in word_start_symbols)   # if this node has a non-eps, word-start in-coming arc
    non_eps_nodes1 = sorted(non_eps_nodes1, key=lambda x: x[0])
    non_eps_nodes2 = list((arc[0], arc[1]) for arc in arcs if arc[3] < 0 or (arc[3] > 0 and arc[3] in word_start_symbols and arc[0] > 0))   # if this node has a non-eps, word-start out-going arc
    self_loops = {ss: l1 for ss, ee, l1, l2, w in arcs if ss == ee}

    # eps_arcs1 = [(ss, ee) for ss, ee, l1, l2, w in arcs if ss != ee and l1 == 0]
    # token_starts = [ss for ss, ee, l1, l2, w in arcs if ss == ee and l1 > 0]
    eps_self_loops = [ss for ss, l1 in self_loops.items() if l1 == 0]
    eps_self_loops = eps_self_loops[1:-1]

    arcs = [arcs[0]] + arcs[2:-5] + [a for a in arcs[-5:] if a[2] >= 0]
    arcs = [[ss, ee, l1, 0, w] for ss, ee, l1, l2, w in arcs]

    # in-coming arcs
    arcs += [(0, n, l, counter.f1(), 0) for n, l in non_eps_nodes1]

    # out-going arcs
    arcs += [(n, final_state, self_loops[n], counter.f3(), 0) for n, l in non_eps_nodes2 if self_loops[n] > 0]
    arcs += [(final_state - 1, final_state, self_loops[final_state - 1], counter.counter3, 0)]

    # skip arcs
    # arcs += [(n, next_token, self_loops[next_token], 0, 0) for ns, next_token in zip(eps_arcs1, token_starts[2:]) for n in ns]
    arcs += [(n1, n2, 0, 0, 0) for n1, n2 in zip(eps_self_loops, eps_self_loops[1:])]

    arcs += [(final_state, final_state, 0, 0, 0)]
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


def get_decoding_graphs(_texts, sp):
    fst_graph = k2.ctc_graph(sp.encode(_texts, out_type=int), modified=False, device='cpu')
    return [fst_graph[i] for i in range(fst_graph.shape[0])]


def get_decoding_graphs_factor_transducer(token_ids):
    return [make_factor_transducer1(t) for t in token_ids]


def _make_factor_transducer5(fst_graphs_3, word_start_symbols, two_ends_bonus=1.0):
    # This is similar to make_factor_transducer3
    # We concat three graphs, and then add some bonus to the two ends of the graph to encourage alignment to the two ends
    # We will probably also do non-linear scaling

    # Compared to make_factor_transducer3, we remove some arcs from the graph
    # Also, we modify some arcs to add bonus to the two ends of the graph to encourage alignment to the two ends

    ############ Graph 1 ############
    fst_graph1 = fst_graphs_3[0]
    
    c_str = k2.to_str_simple(fst_graph1)
    arcs = c_str.strip().split("\n")
    arcs = [x.strip() for x in arcs if len(x.strip()) > 0]
    final_state = int(arcs[-1])

    arcs = arcs[:-1]
    arcs = [tuple(map(int, a.split())) for a in arcs]
    # ss, ee, l1, l2, w = arc

    non_eps_nodes = set((arc[1], arc[3]) for arc in arcs if arc[3] > 0 and arc[3] in word_start_symbols)   # if this node has a non-eps, word-start in-coming arc
    arcs += [(0, n, l, l, 0) for n, l in non_eps_nodes if n > 1]

    # boost the weights
    arcs = [(ss, ee, l1, l2, w + two_ends_bonus) if (ss>0 and l1>0) else (ss, ee, l1, l2, w + two_ends_bonus * 5) if (ee>0 and l1>0) else (ss, ee, l1, l2, w) for ss, ee, l1, l2, w in arcs]

    arcs += [(0, final_state, -1, -1, 0)]

    new_arcs = arcs
    new_arcs.append([final_state])

    new_arcs = sorted(new_arcs, key=lambda arc: arc[0])
    new_arcs = [[str(i) for i in arc] for arc in new_arcs]
    new_arcs = [" ".join(arc) for arc in new_arcs]
    new_arcs = "\n".join(new_arcs)

    fst_graph1 = k2.Fsa.from_str(new_arcs, acceptor=False)

    ############ Graph 2 ############
    fst_graph2 = fst_graphs_3[1]

    ############ Graph 3 ############
    fst_graph3 = fst_graphs_3[2]

    c_str = k2.to_str_simple(fst_graph3)
    arcs = c_str.strip().split("\n")
    arcs = [x.strip() for x in arcs if len(x.strip()) > 0]
    final_state = int(arcs[-1])

    arcs = arcs[:-1]
    arcs = [tuple(map(int, a.split())) for a in arcs]
    # ss, ee, l1, l2, w = arc

    non_eps_nodes2 = set((arc[0], arc[3]) for arc in arcs if arc[3] > 0 and arc[3] in word_start_symbols)   # if this node has a non-eps, word-start out-going arc
    non_eps_nodes2 = [(n, l) for n, l in non_eps_nodes2 if 0 < n < final_state - 2]
    arcs += [(n, final_state, -1, -1, 0) for n, l in non_eps_nodes2]

    # boost the weights
    arcs = [(ss, ee, l1, l2, w + two_ends_bonus) if (ee!=final_state and l1>0) else (ss, ee, l1, l2, w + two_ends_bonus*5) if l1>0 else (ss, ee, l1, l2, w) for ss, ee, l1, l2, w in arcs]

    arcs += [(0, final_state, -1, -1, 0)]

    new_arcs = arcs
    new_arcs.append([final_state])

    new_arcs = sorted(new_arcs, key=lambda arc: arc[0])
    new_arcs = [[str(i) for i in arc] for arc in new_arcs]
    new_arcs = [" ".join(arc) for arc in new_arcs]
    new_arcs = "\n".join(new_arcs)

    fst_graph3 = k2.Fsa.from_str(new_arcs, acceptor=False)

    # concatenate the three graphs: don't be fooled by `k2.cat`
    compiler = openfst.Compiler()
    compiler.write(k2.to_str_simple(fst_graph1, openfst=True))
    f1 = compiler.compile()
    compiler.write(k2.to_str_simple(fst_graph2, openfst=True))
    f2 = compiler.compile()
    compiler.write(k2.to_str_simple(fst_graph3, openfst=True))
    f3 = compiler.compile()
    f1.concat(f2)
    f1.concat(f3)
    fst = k2.Fsa.from_openfst(str(f1), acceptor=False)
    fst = k2.arc_sort(fst)
    return fst


def make_factor_transducer5(libri_long_text_str, cut_ids, text_ranges, sp, extension=10, two_ends_bonus=1.0):
    # This factor transducer enforce some specific factors to be present in the graph

    text_ranges = [(max(rg[0]-1, 0), rg[-1]-1) for cid, rg in zip(cut_ids, text_ranges)]
    text_ranges = [
        (
            (max(rg[0]-extension, 0), rg[0]),  # left extension
            (rg[0], rg[-1]),                    # alignment results
            (rg[-1], rg[-1]+extension)           # right extension
        ) for cid, rg in zip(cut_ids, text_ranges)
    ]
    text_keys = [tuple(get_uid_key(cid)[:2]) for cid in cut_ids]
    texts = [" ".join(libri_long_text_str[tk][rg[0]: rg[-1]]) for tk, rg3 in zip(text_keys, text_ranges) for rg in rg3]
    token_ids = sp.encode(texts, out_type=int)
    fst_graphs = k2.ctc_graph(token_ids, modified=False, device='cpu')

    # token_lens = [(len(left), len(mid), len(right)) for left, mid, right in zip(token_ids[::3], token_ids[1::3], token_ids[2::3])]
    # word_id_lists = [left + mid + right for left, mid, right in zip(token_ids[::3], token_ids[1::3], token_ids[2::3])]

    # breakpoint()

    fst_graphs = [(fst_graphs[i], fst_graphs[i+1], fst_graphs[i+2]) for i in range(0, fst_graphs.shape[0], 3)]
    word_start_symbols = {i for i in range(sp.vocab_size()) if sp.id_to_piece(i).startswith('▁')}
    fst_graphs = [_make_factor_transducer5(fst_graphs_3, word_start_symbols, two_ends_bonus=two_ends_bonus) for fst_graphs_3 in fst_graphs]
    
    return fst_graphs
