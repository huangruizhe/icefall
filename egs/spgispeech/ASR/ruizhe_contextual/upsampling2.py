from egs.spgispeech.ASR.pruned_transducer_stateless2_context.context_collector import ContextCollector
# export PYTHONPATH=/export/fs04/a12/rhuang/icefall_align2/:$PYTHONPATH
# export PYTHONPATH=/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/pruned_transducer_stateless7_context:$PYTHONPATH

import logging
import argparse
from pathlib import Path
import sentencepiece as spm
from itertools import chain
import ast
from dataclasses import replace

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
        "--N",
        type=int,
        default=5,
        help="The ratio to upsample sentence containing rare words ",
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

def upsample_utterance_rare(params, context_collector):
    from lhotse import CutSet
    import numpy as np
    from tqdm import tqdm
    import random

    in_file_name = '/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_train_shuf.jsonl.gz'
    out_file_name = f'/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_train_shuf_upsampled_{params.N}.jsonl.gz'
    cuts = CutSet.from_file(in_file_name)
    logging.info(f'len(cuts) = {len(cuts)}')
    
    cuts_common = []
    cuts_rare = []
    cnt = 0
    for c in tqdm(cuts, mininterval=2):
        assert len(c.supervisions) == 1
        text = c.supervisions[0].text
        text = text.split()
        rare = [w for w in text if w in context_collector.rare_words]

        if len(rare) > 0:
            cuts_rare.append(c)
        else:
            cuts_common.append(c)
        
        # cnt += 1
        # if cnt > 10000:
        #     break

    logging.info(f"len(cuts_rare) = {len(cuts_rare)}")
    logging.info(f"len(cuts_common) = {len(cuts_common)}")

    upsampled_cuts = [] + cuts_common + cuts_rare * params.N
    random.shuffle(upsampled_cuts)
    # import pdb; pdb.set_trace()
    for i in tqdm(range(len(upsampled_cuts)), mininterval=2):
        upsampled_cuts[i] = replace(upsampled_cuts[i], id=upsampled_cuts[i].id + f"_up{i}")
    upsampled_cuts = CutSet.from_cuts(upsampled_cuts)
    upsampled_cuts.describe()

    logging.info(f"Saving to: {out_file_name}")
    upsampled_cuts.to_file(out_file_name)
    logging.info(f'Done: {out_file_name}')


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
    
    upsample_utterance_rare(params, context_collector)

if __name__ == '__main__':
    opts = parse_opts()

    main(opts)


