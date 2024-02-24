import k2
import sentencepiece as spm
import torch
# import pywrapfst as openfst
# from icefall.decode import get_lattice, one_best_decoding
# from icefall.utils import get_alignments, get_texts, get_texts_with_timestamp
from data_long_dataset import *
from pathlib import Path
import itertools
import logging
import lis


def get_linear_fst(word_ids_list, blank_id=0, max_symbol_id=1000, is_left=True, return_str=False):
    graph = k2.linear_fst(labels=[word_ids_list], aux_labels=[word_ids_list])[0]

    max_symbol_id += 1

    c_str = k2.to_str_simple(graph)
    arcs = c_str.strip().split("\n")
    arcs = [x.strip() for x in arcs if len(x.strip()) > 0]
    final_state = int(arcs[-1])

    arcs = arcs[:-1]
    arcs = [tuple(map(int, a.split())) for a in arcs]

    max_symbol_id = max(max_symbol_id, len(arcs) + 10)

    new_arcs = []
    # left: ground-truth (from arange)
    # right: hypothesis (from alignment)
    for i, (ss, ee, l1, l2, w) in enumerate(arcs):
        if l1 > 0:
            # substitution
            if is_left:
                new_arcs.append([ss, ee, l1, max_symbol_id, -1])
            else:
                new_arcs.append([ss, ee, max_symbol_id, i+1, -1])
            
            # deletion
            new_arcs.append([ss, ee, l1, blank_id, -2])
        
        # insertion
        if is_left:
            new_arcs.append([ss, ss, blank_id, max_symbol_id, -2])
        else:
            new_arcs.append([ss, ss, max_symbol_id, blank_id, -2])
    
    if not is_left:
        arcs = [(ss, ee, l1, i+1 if l2 >= 0 else l2, w) for i, (ss, ee, l1, l2, w) in enumerate(arcs)]

    new_arcs = arcs + new_arcs
    new_arcs.append([final_state])

    new_arcs = sorted(new_arcs, key=lambda arc: arc[0])
    new_arcs = [[str(i) for i in arc] for arc in new_arcs]
    new_arcs = [" ".join(arc) for arc in new_arcs]
    new_arcs = "\n".join(new_arcs)

    if return_str:
        return new_arcs
    else:
        fst = k2.Fsa.from_str(new_arcs, acceptor=False)
        fst = k2.arc_sort(fst)
        return fst


def get_range_without_outliers(my_list, scan_range=100, outlier_threshold=60):
    # remove outliers
    # given a list of integers in my_list in ascending order, find the range without outliers
    # outliers: a number that is outlier_threshold smaller/larger than its neighbors

    # my_list = [150] + list(range(200,1000)) + [1105]
    # print(my_list)
    # get_range_without_outliers(my_list)

    if len(my_list) <= 10:
        return my_list[0], my_list[-1]

    scan_range = min(scan_range, int(len(my_list)/2) - 1)
    
    left = [i+1 for i in range(0, scan_range) if my_list[i+1] - my_list[i] > outlier_threshold]
    right = [i-1 for i in range(-scan_range, 0) if my_list[i] - my_list[i-1] > outlier_threshold]
    left = left[-1] if len(left) > 0 else 0
    right = right[0] if len(right) > 0 else -1
    left = my_list[left]
    right = my_list[right]
    return left, right


def remove_outliers(my_list, max_symbol_id, scan_range=100, outlier_threshold=60):
    # Given a list of integers in my_list in ascending order, remove outliers
    # such that there is a gap more than outlier_threshold

    while my_list[0] == max_symbol_id:
        my_list = my_list[1:]
    while my_list[-1] == max_symbol_id:
        my_list = my_list[:-1]

    if len(my_list) <= 10:
        return my_list
    
    scan_range = min(scan_range, int(len(my_list)/2) - 1)
    left = [i+1 for i in range(0, scan_range) if my_list[i+1] - my_list[i] > outlier_threshold]
    right = [i-1 for i in range(-scan_range, 0) if my_list[i] - my_list[i-1] > outlier_threshold]
    left = left[-1] if len(left) > 0 else 0
    right = right[0]+1 if len(right) > 0 else None

    return my_list[left: right]


def get_lis_alignment(hyp_list, lis_result, max_symbol_id):
    # The last integer from lis_result may be wrong
    # lis_result = lis_result[:-1]
    # We have removed the outliers already
    
    midpoint = len(lis_result) // 2

    indices_in_segment = [dict()]
    last_boundary_i = 0
    j = 0  # i: index in hyp_list, j: index in lis_result
    for i in range(len(hyp_list)):
        if j < midpoint:
            if hyp_list[i] == lis_result[j]:
                indices_in_segment[-1][lis_result[j]] = i - last_boundary_i
                j += 1
                if j == len(lis_result):
                    break
        else:
            if hyp_list[i] == lis_result[j] and lis_result[j] not in indices_in_segment[-1]:
                indices_in_segment[-1][lis_result[j]] = i - last_boundary_i
                j += 1
                if j == len(lis_result):
                    break
        
        if hyp_list[i] == max_symbol_id:
            last_boundary_i = i + 1
            indices_in_segment.append(dict())
    
    # here, i and j still match; let's increase i by a certain range and 
    # see if there's something that can extend to the alignment
    range_thres = 50
    gap_thres = 10  # within 10 words
    last_lis_result = lis_result[-1]
    for k in range(range_thres):
        i += 1
        if i == len(hyp_list):
            break
        if 0 < hyp_list[i] - last_lis_result < gap_thres and hyp_list[i] not in indices_in_segment[-1]:
            indices_in_segment[-1][hyp_list[i]] = i - last_boundary_i
            last_lis_result = hyp_list[i]

    return indices_in_segment


class WordCounter: 
    def __init__(self, val=0): 
        self.counter1 = val
    def f1(self): 
        self.counter1 += 1; return self.counter1


def reduce_long_list1(ids_list1, ids_list2):
    # break it into parts and compute overlap

    T = 2000
    start = 0
    end = None

    for i in range(0, len(ids_list1), T):
        ids_list1_tmp = ids_list1[i: i+T]
        nonoverlap = set(ids_list1_tmp) - set(ids_list2)
        if len(nonoverlap) / len(ids_list1_tmp) < 0.3:
            start = max(0, i-1)
            break
    
    for i in reversed(range(0, len(ids_list1), T)):
        ids_list1_tmp = ids_list1[i: i+T]
        if len(ids_list1_tmp) < 0.8 * T:
            continue

        nonoverlap = set(ids_list1_tmp) - set(ids_list2)
        if len(nonoverlap) / len(ids_list1_tmp) < 0.3:
            end = i + 1 + 1
            break
    
    return ids_list1[start: end]



def get_aligned_list(hyps_list, my_hyps_min=None, my_hyps_max=None, device='cpu'):
    my_hyps_ids = sorted([w for hyp in hyps_list for w in hyp])

    _my_hyps_min, _my_hyps_max = get_range_without_outliers(my_hyps_ids, scan_range=100, outlier_threshold=60)
    my_hyps_min = _my_hyps_min if my_hyps_min is None else max(my_hyps_min - 10, 0)
    my_hyps_max = _my_hyps_max if my_hyps_max is None else my_hyps_max + 10
        
    max_symbol_id = max(my_hyps_ids) + 100   # use this symbol to mark the chunk boundaries
    ids_list1 = list(range(my_hyps_min, my_hyps_max + 1))
    ids_list2 = [item for sublist in hyps_list for item in ([max_symbol_id] + sublist)]
    ids_list2 = ids_list2[1:]

    list1_len_thres = 40000
    if len(ids_list1) > list1_len_thres:
        logging.warning(f"Long ids_list1 ({len(ids_list1)}) in range {my_hyps_min} and {my_hyps_max} vs. list2 ({len(ids_list2)})")
        ids_list1 = reduce_long_list1(ids_list1, ids_list2)
        logging.warning(f"Reduced to ({len(ids_list1)})")

    graph1 = get_linear_fst(ids_list1, max_symbol_id=max_symbol_id+1, blank_id=0, is_left=True, return_str=False)
    graph2 = get_linear_fst(ids_list2, max_symbol_id=max_symbol_id+1, blank_id=0, is_left=False, return_str=False)

    # graph1 = graph1.to(device)
    # graph2 = graph2.to(device)

    rs = k2.compose(graph1, graph2, treat_epsilons_specially=True)
    rs = k2.connect(rs)
    rs = k2.remove_epsilon_self_loops(rs)
    # rs = k2.arc_sort(rs)
    rs = k2.top_sort(rs)  # top-sort is needed for shortest path: https://github.com/k2-fsa/k2/issues/746#issuecomment-856503238
    # print("Composed graph size: ", rs.shape, rs.num_arcs)

    rs_vec = k2.create_fsa_vec([rs])
    best_paths = k2.shortest_path(rs_vec, use_double_scores=True)
    # best_paths.shape, best_paths.num_arcs

    best_paths = k2.top_sort(best_paths)
    best_paths = k2.arc_sort(best_paths)

    rs_list1 = best_paths[0].labels.tolist()
    rs_list2 = best_paths[0].aux_labels.tolist()
    rs_list2_ = [ids_list2[i-1] if i > 0 else None for i in rs_list2]
    for l1, l2 in zip(rs_list1, rs_list2_):
        if l1 == l2:
            break
    rs_my_hyps_min = l1
    
    for l1, l2 in zip(reversed(rs_list1), reversed(rs_list2_)):
        if l1 == l2 and l1 > 0:
            break
    rs_my_hyps_max = l1

    # Adjust rs_list2 (which are the indices in ids_list2) for each chunk
    # We need to convert it to the indices in the chunk
    word_counter = WordCounter(-1)
    i_mapping = {word_counter.f1() : i - 1 for sublist in hyps_list for i, item in enumerate([max_symbol_id] + sublist)}
    rs_list2 = [i_mapping[i] if i in i_mapping else None for i in rs_list2]

    # print("Best path range: ", rs_my_hyps_min, rs_my_hyps_max)

    return best_paths, rs_list1, rs_list2, rs_list2_, max_symbol_id, rs_my_hyps_min, rs_my_hyps_max


def handle_failed_groups(no_need_to_realign, alignment_results):
    # Just simply and linearly makeup the timestamp (frame index in the output frames)

    if len(no_need_to_realign) > 0:
        for group in no_need_to_realign:
            ss = group[0]   # this is aligned left end point
            ee = group[-1]  # this is aligned right end point
            num_gaps = ee - ss
            tt1 = alignment_results[ss]
            tt2 = alignment_results[ee]
            for j, i in enumerate(range(ss + 1, ee)):
                alignment_results[i] = int((tt2 - tt1) / num_gaps * (j + 1) + tt1)
    return

# TODO: we may need to do a two pass alignment
def align_long_text(rs, num_segments_per_chunk=5, neighbor_threshold=5, device='cpu'):
    # The task here is to find the "reliable" aligned parts from the alignment results `rs`
    # Since the alignment results are actually "indices" in the long text, we hope to find the
    # longest increasing subsubsequence from the alignment results.

    # solution1: just wfst(k2) to compute edit distance (shortest path from the pruned graph)
    # solution2: just use python's difflib: https://docs.python.org/3/library/difflib.html#differ-example
    #            - https://github.com/lowerquality/gentle/blob/master/gentle/diff_align.py
    # solution3: https://unix.stackexchange.com/questions/2150/diffing-two-big-text-files
    # solution4: Longest Increasing Subsequence (LIS)
    #            - https://en.wikipedia.org/wiki/Longest_increasing_subsequence
    #            - https://www.reddit.com/r/algorithms/comments/c5rerp/given_a_list_of_unsorted_integers_find_the/?onetap_auto=true
    #            - https://leetcode.com/problems/longest-increasing-subsequence/description/
    #            - https://algo.monster/liteproblems/673
    #            - https://python.plainenglish.io/longest-increasing-subsequence-python-e75d028cef7a
    #            - https://www.youtube.com/watch?v=66w10xKzbRM&t=0s
    # solution5: vimdiff: vimdiff <(tr ' ' '\n' <download/LibriSpeechAligned/LibriSpeech/books/ascii/2981/2981.txt) <(tr ' ' '\n' <download/LibriSpeechAligned/LibriSpeech/books/ascii/3600/3600.txt)

    hyps = rs['hyps']
    timestamps = rs['timestamps']
    output_frame_offset = rs['output_frame_offset'].tolist()
    meta_data = rs['meta_data']

    alignment_results = dict()

    # find the longest increasing subsequence
    hyp_list = [i for hyp in hyps for i in hyp]
    max_symbol_id = max(hyp_list) + 100   # use this symbol to mark the chunk boundaries
    hyp_list = [i for hyp in hyps for i in [max_symbol_id] + hyp]
    hyp_list = hyp_list[1:]
    lis_result = lis.longestIncreasingSubsequence(hyp_list)
    lis_result = remove_outliers(lis_result, max_symbol_id, scan_range=100, outlier_threshold=60)
    if len(lis_result) == 0:
        return dict(), None
    indices_in_segment = get_lis_alignment(hyp_list, lis_result, max_symbol_id)  # hyp_list and lis_result are both word indices in the long text
    
    for idx, ts, ofs in zip(indices_in_segment, timestamps, output_frame_offset):
        for k, v in idx.items():
            alignment_results[k] = ts[v] + ofs
    
    # Post-process: remove isolatedly aligned words
    # Each aligned word should have a neighborhood of at least neighbor_threshold words
    rg_min = min(alignment_results.keys())
    rg_max = max(alignment_results.keys())
    aligned_flag = [i in alignment_results for i in range(rg_min, rg_max + 1)]
    rg_min_tt = alignment_results[rg_min]  # just in case the first or last word got removed from the alignment due to the heursitic below
    rg_max_tt = alignment_results[rg_max]
    for i in range(rg_min, rg_max + 1):
        i = i - rg_min
        if aligned_flag[i]:
            sub_list = aligned_flag[max(0, i-neighbor_threshold): i + neighbor_threshold]
            if sum(sub_list) < 0.5 * len(sub_list):  # only less than 50% of the words in the neighborhood are aligned
                del alignment_results[i + rg_min]
                aligned_flag[i] = False
    
    # Find the aligned parts
    alignment_results[rg_min] = rg_min_tt  # dirty solution -- I still need to put them here to provide boundary information. They will still be re-aligned
    alignment_results[rg_max] = rg_max_tt
    to_realign, no_need_to_realign = find_unaligned(aligned_flag, rg_min, alignment_results)

    # For some unaligned parts, we don't need to realign them cos they are too short
    handle_failed_groups(no_need_to_realign, alignment_results)

    return alignment_results, to_realign


def merge_segments(segments, threshold, is_sorted=True):
    merged = []
    if not is_sorted:
        segments = sorted(segments, key=lambda x: x[0]) 
    for start, end in segments:
        if merged and start - merged[-1][1] <= threshold:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def get_dur_group(group, rg_min, alignment_results):
    ss = rg_min + group[0] - 1
    ee = rg_min + group[-1] + 1
    return alignment_results[ee] - alignment_results[ss]


def get_neighbor_aligned_word(idx, alignment_results, neighbor_range):
    if neighbor_range > 0:
        step = -1
    else:
        step = 1

    idx += neighbor_range
    while idx != 0:
        if idx in alignment_results:
            return idx
        idx += step
    
    return idx


def find_unaligned(aligned_flag, rg_min, alignment_results, no_need_to_realign_thres1=2, no_need_to_realign_thres2=10):
    # assert aligned_flag[0] is True
    # assert aligned_flag[-1] is True

    # Find the indices of all consecutive "False" segments
    to_realign = [[i for i, _ in group] for key, group in itertools.groupby(enumerate(aligned_flag), key=lambda x: x[1]) if not key]

    if False:
        # Too few words, e.g., 2 words
        no_need_to_realign1 = [group for group in to_realign if len(group) <= no_need_to_realign_thres1]
        no_need_to_realign1 = [(rg_min + group[0] - 1, rg_min + group[-1] + 1) for group in no_need_to_realign1]
        to_realign = [group for group in to_realign if len(group) > no_need_to_realign_thres1]

        # Too short time, e.g., 10 frames in the final output is 0.4 sec
        no_need_to_realign2 = [group for group in to_realign if get_dur_group(group, rg_min, alignment_results) <= no_need_to_realign_thres2]
        no_need_to_realign2 = [(rg_min + group[0] - 1, rg_min + group[-1] + 1) for group in no_need_to_realign2]
        to_realign = [group for group in to_realign if get_dur_group(group, rg_min, alignment_results) > no_need_to_realign_thres2]

        no_need_to_realign = no_need_to_realign1 + no_need_to_realign2
        no_need_to_realign = sorted(no_need_to_realign, key=lambda x: x[0])
    else:
        # Ok, just add more padding to the segments
        no_need_to_realign = []
        pass

    # Ok, these are needed to be realigned
    to_realign = [(group[0], group[-1]) for group in to_realign]

    # Merge the unaligned segments if they are close to each other
    to_realign = merge_segments(to_realign, threshold=3, is_sorted=True)
    neighbor_range = 3  # add 3 already aligned words to the left and right
    to_realign = [
        (
            get_neighbor_aligned_word(rg_min + group[0], alignment_results, -neighbor_range),
            get_neighbor_aligned_word(rg_min + group[-1] + 1, alignment_results, neighbor_range)
        ) for group in to_realign
    ]  # each start/end will be an aligned word
    return to_realign, no_need_to_realign


def to_audacity_label_format(params, frame_rate, alignment_results, text):
    # To audacity: https://manual.audacityteam.org/man/importing_and_exporting_labels.html
    text = text.split()
    alignment_results_ = [(text[k], v*frame_rate*params.subsampling_factor) for k, v in sorted(alignment_results.items())] 

    audacity_labels_str = "\n".join([f"{t:.2f}\t{t:.2f}\t{label}" for label, t in alignment_results_])
    # print(audacity_labels)
    
    # with open("audacity_labels.txt", "w") as fout:
    #     print(audacity_labels_str, file=fout)

    # str(Path("audacity_labels.txt").absolute())
    return audacity_labels_str