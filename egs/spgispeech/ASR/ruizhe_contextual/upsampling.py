from lhotse import CutSet
import string
from tqdm import tqdm
from dataclasses import replace
import json
import pickle
import gzip

from egs.spgispeech.ASR.pruned_transducer_stateless7_context.context_collector import ContextCollector
# export PYTHONPATH=/export/fs04/a12/rhuang/icefall_align2/:$PYTHONPATH
# export PYTHONPATH=/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/pruned_transducer_stateless7_context:$PYTHONPATH


# https://github.com/lhotse-speech/lhotse/blob/172e9f7a09e70cee4ee977fa9d288f96616cf7ec/lhotse/recipes/spgispeech.py
def normalize(text: str) -> str:
    # Remove all punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Convert all upper case to lower case
    text = text.lower()
    return text

N = 5  # ratio of upsamped sentence containing rare words 

in_file_name = "/exp/rhuang/icefall_latest/egs/spgispeech/ASR/data/manifests/cuts_train_shuf.jsonl.gz"
out_file_name = f"/exp/rhuang/icefall_latest/egs/spgispeech/ASR/data/manifests/cuts_train_shuf_upsample_{N}.jsonl.gz"

print(f"Loading cuts from: {in_file_name}")
with gzip.open(in_file_name, 'r') as fin:
    cuts = [json.loads(line) for line in fin]



print(f"len(cuts) = {len(cuts)}")  # len(cuts) = 5886320
for c in tqdm(cuts):
    # for s in c.supervisions:
    #     s.text = normalize(s.text)
    assert len(c['supervisions']) == 1
    s = c['supervisions'][0]
    text = normalize(s['text'])
    s['text'] = text

print(f"Saving cuts to: {out_file_name}")
# cuts = cuts.trim_to_supervisions(keep_overlapping=False).to_eager()
# cuts = cuts.sort_by_duration()
with gzip.open(out_file_name, 'wt') as fout:
    for c in cuts:
        c_str = json.dumps(c)
        print(c_str, file=fout)
print("Done.")
