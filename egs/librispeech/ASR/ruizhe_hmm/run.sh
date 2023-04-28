mamba activate /home/rhuang/mambaforge/envs/efrat2
export PYTHONPATH=/export/fs04/a12/rhuang/k2/k2/python:$PYTHONPATH # for `import k2`
export PYTHONPATH=/export/fs04/a12/rhuang/k2/build/temp.linux-x86_64-cpython-38/lib/:$PYTHONPATH # for `import _k2`
# export PYTHONPATH=/export/fs04/a12/rhuang/icefall_align/:$PYTHONPATH
# export PYTHONPATH=/export/fs04/a12/rhuang/icefall/:$PYTHONPATH
export PYTHONPATH=/export/fs04/a12/rhuang/icefall_align2/:$PYTHONPATH

cd /export/fs04/a12/rhuang/icefall/egs/librispeech/ASR

# Models to run:
# - zipformer-mmi (wp), with hmm topology, on libri100
# - with or w/o sil:
#   - tdnn-lstm-ctc (phone), on libri100
#   - tdnn-lstm-ctc (wp), on libri100
#   - tdnn-lstm-ctc (wp), with hmm topology, on libri100

cd /exp/rhuang/icefall_latest/egs/librispeech/ASR
scp -r rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/tdnn_lstm_ctc/decode.py tdnn_lstm_ctc/.

b 649

transcript_fsa.labels_sym = self.lexicon.token_table
transcript_fsa[0].draw("aaa.svg")

decoding_graph.labels_sym = self.lexicon.token_table
decoding_graph.aux_labels_sym = self.lexicon.word_table
decoding_graph[0].draw("aaa1.svg")

supervisions['text'][0]
decoding_graph.labels_sym = ctc_graph_compiler.lexicon.token_table
decoding_graph.aux_labels_sym = ctc_graph_compiler.lexicon.word_table
decoding_graph[0].shape, decoding_graph[0].num_arcs
decoding_graph[0].draw("aaa1.svg")

texts[0] = "I CRYRY I LAUGH AND"
word_ids_list = ctc_graph_compiler.texts_to_ids(texts)
decoding_graph = ctc_graph_compiler.compile(word_ids_list, modified=True)

word_fsa.labels_sym = self.lexicon.word_table
word_fsa[0].draw("aaa2.svg")

lexicon = Lexicon(params.lang_dir)
supervisions['text'][0]
decoding_graph.labels_sym = lexicon.token_table
decoding_graph.aux_labels_sym = lexicon.token_table
decoding_graph[0].shape, decoding_graph[0].num_arcs
decoding_graph[0].draw("aaa1.svg")

ctc_graph_compiler.remove_intra_word_blk_flag = True

# Generate lexicon that contains sil
bash ruizhe/prepare_sil.sh \
  --stage 0 --stop-stage 0

# CTC vs hmm
/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/ruizhe_hmm/log/train-3629925.out  # hmm
/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/ruizhe_hmm/log/train-3629926.out  # ctc
  - /export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/zipformer_mmi_hmm/exp/exp_libri_100_ml/tensorboard1
  - https://tensorboard.dev/experiment/hGZIGjQ5Q8adH2Rp7BkRhg/

tensorboard dev upload --logdir . --description "`pwd`"

/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/ruizhe_hmm/log/train-3630391.out  # hmm with added blk at the tail
/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/ruizhe_hmm/log/train-3630390.out  # continue training hmm from ctc

/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/ruizhe_hmm/log/train-3630765.out  # zipformer_ctc model

# small sample
/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/ruizhe_hmm/log/train-3631059.out  # zipformer_ctc model
/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/ruizhe_hmm/log/train-3631061.out  # zipformer_hmm model
/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/ruizhe_hmm/log/train-3631100.out  # zipformer_hmm model + sort cuts in an accending order
  - /export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/zipformer_mmi/exp/exp_libri_100_ml/tensorboard1
  - https://tensorboard.dev/experiment/C4cJMpdzSGCQRrGO5xA7Fw/#scalars

# libri100
/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/ruizhe_hmm/log/train-3631157.out  # zipformer_ctc model
/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/ruizhe_hmm/log/train-3631156.out  # zipformer_hmm model
/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/ruizhe_hmm/log/train-3631148.out  # zipformer_hmm model + sort cuts in an accending order
/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/ruizhe_hmm/log/train-3631179.out  # zipformer_hmm model + continue training from ctc model

