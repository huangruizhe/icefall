cd /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/

mamba activate /home/rhuang/mambaforge/envs/efrat2
export PYTHONPATH=/export/fs04/a12/rhuang/k2/k2/python:$PYTHONPATH # for `import k2`
export PYTHONPATH=/export/fs04/a12/rhuang/k2/build/temp.linux-x86_64-cpython-38/lib/:$PYTHONPATH # for `import _k2`
export PYTHONPATH=/export/fs04/a12/rhuang/icefall_align2/:$PYTHONPATH

export PYTHONPATH=/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/pruned_transducer_stateless7_context:$PYTHONPATH

N=5
python /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/upsampling2.py --N $N

scp -r /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_train_shuf_upsampled_$N.jsonl.gz \
  rhuang@test1.hltcoe.jhu.edu:/exp/rhuang/icefall_latest/egs/spgispeech/ASR/pretrained/icefall-asr-spgispeech-pruned-transducer-stateless2/data/manifests/cuts_train_shuf_upsampled_$N.jsonl.gz


# 2023-04-30 01:10:14,230 - INFO - upsample_utterance_rare:100 - len(cuts_rare) = 4135690
# 2023-04-30 01:10:14,230 - INFO - upsample_utterance_rare:101 - len(cuts_common) = 1750630
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 14157700/14157700 [02:42<00:00, 87373.23it/s]
# Cut statistics:
# ╒═══════════════════════════╤═════════════╕
# │ Cuts count:               │ 14157700    │
# ├───────────────────────────┼─────────────┤
# │ Total duration (hh:mm:ss) │ 37183:31:09 │
# ├───────────────────────────┼─────────────┤
# │ mean                      │ 9.5         │
# ├───────────────────────────┼─────────────┤
# │ std                       │ 2.8         │
# ├───────────────────────────┼─────────────┤
# │ min                       │ 4.6         │
# ├───────────────────────────┼─────────────┤
# │ 25%                       │ 7.2         │
# ├───────────────────────────┼─────────────┤
# │ 50%                       │ 9.2         │
# ├───────────────────────────┼─────────────┤
# │ 75%                       │ 11.5        │
# ├───────────────────────────┼─────────────┤
# │ 99%                       │ 16.1        │
# ├───────────────────────────┼─────────────┤
# │ 99.5%                     │ 16.4        │
# ├───────────────────────────┼─────────────┤
# │ 99.9%                     │ 16.6        │
# ├───────────────────────────┼─────────────┤
# │ max                       │ 16.7        │
# ├───────────────────────────┼─────────────┤
# │ Recordings available:     │ 14157700    │
# ├───────────────────────────┼─────────────┤
# │ Features available:       │ 14157700    │
# ├───────────────────────────┼─────────────┤
# │ Supervisions available:   │ 14157700    │
# ╘═══════════════════════════╧═════════════╛
# Speech duration statistics:
# ╒══════════════════════════════╤═════════════╤══════════════════════╕
# │ Total speech duration        │ 37183:31:09 │ 100.00% of recording │
# ├──────────────────────────────┼─────────────┼──────────────────────┤
# │ Total speaking time duration │ 37183:31:09 │ 100.00% of recording │
# ├──────────────────────────────┼─────────────┼──────────────────────┤
# │ Total silence duration       │ 00:00:00    │ 0.00% of recording   │
# ╘══════════════════════════════╧═════════════╧══════════════════════╛
# 2023-04-30 01:20:07,480 - INFO - upsample_utterance_rare:111 - Saving to: /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_train_shuf_upsampled_3.jsonl.gz
# 2023-04-30 01:55:34,607 - INFO - upsample_utterance_rare:113 - Done: /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_train_shuf_upsampled_3.jsonl.gz

