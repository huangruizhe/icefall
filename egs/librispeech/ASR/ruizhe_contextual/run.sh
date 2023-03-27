# extract a subset of libri100
cd /export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR
ipython

import lhotse
from lhotse import load_manifest, CutSet
cuts = load_manifest("data/fbank/librispeech_cuts_train-clean-100.jsonl.gz")
random_sample = cuts.sample(n_cuts=int(len(cuts) * 0.4))
cuts_ = [c for c in cuts if c in random_sample]
print(len(cuts_))
cuts_ = CutSet.from_items(cuts_)
cuts_.to_file("data/fbank/librispeech_cuts_train-clean-100-0.4.jsonl.gz")

# In [6]: random_sample.describe()
# Cut statistics:
# ╒═══════════════════════════╤═══════════╕
# │ Cuts count:               │ 34246     │
# ├───────────────────────────┼───────────┤
# │ Total duration (hh:mm:ss) │ 121:35:57 │
# ├───────────────────────────┼───────────┤
# │ mean                      │ 12.8      │
# ├───────────────────────────┼───────────┤
# │ std                       │ 3.8       │
# ├───────────────────────────┼───────────┤
# │ min                       │ 1.6       │
# ├───────────────────────────┼───────────┤
# │ 25%                       │ 11.4      │
# ├───────────────────────────┼───────────┤
# │ 50%                       │ 13.8      │
# ├───────────────────────────┼───────────┤
# │ 75%                       │ 15.3      │
# ├───────────────────────────┼───────────┤
# │ 99%                       │ 18.1      │
# ├───────────────────────────┼───────────┤
# │ 99.5%                     │ 18.4      │
# ├───────────────────────────┼───────────┤
# │ 99.9%                     │ 18.9      │
# ├───────────────────────────┼───────────┤
# │ max                       │ 20.3      │
# ├───────────────────────────┼───────────┤
# │ Recordings available:     │ 34246     │
# ├───────────────────────────┼───────────┤
# │ Features available:       │ 34246     │
# ├───────────────────────────┼───────────┤
# │ Supervisions available:   │ 34246     │
# ╘═══════════════════════════╧═══════════╛
# Speech duration statistics:
# ╒══════════════════════════════╤═══════════╤══════════════════════╕
# │ Total speech duration        │ 121:35:57 │ 100.00% of recording │
# ├──────────────────────────────┼───────────┼──────────────────────┤
# │ Total speaking time duration │ 121:35:57 │ 100.00% of recording │
# ├──────────────────────────────┼───────────┼──────────────────────┤
# │ Total silence duration       │ 00:00:00  │ 0.00% of recording   │
# ╘══════════════════════════════╧═══════════╧══════════════════════╛

cd /exp/rhuang/icefall_latest/egs/librispeech/ASR
scp -r rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/pruned_transducer_stateless7/train.py pruned_transducer_stateless7/train.py
scp -r rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/tdnn_lstm_ctc/asr_datamodule.py tdnn_lstm_ctc/asr_datamodule.py
scp -r rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/ruizhe_contextual/ .
scp -r rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/data/fbank/librispeech_cuts_train-clean-100-0.4.jsonl.gz data/fbank/.


cd /export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/data
git clone git@github.com:facebookresearch/fbai-speech.git
cd -

