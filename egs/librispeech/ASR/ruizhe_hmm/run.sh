mamba activate /home/rhuang/mambaforge/envs/efrat
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

word_fsa.labels_sym = self.lexicon.word_table
word_fsa[0].draw("aaa2.svg")


# Generate lexicon that contains sil
bash ruizhe/prepare_sil.sh \
  --stage 0 --stop-stage 0


