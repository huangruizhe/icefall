#!/usr/bin/env bash

set -eou pipefail

nj=15
stage=-1
stop_stage=100

dl_dir=$PWD/download

. shared/parse_options.sh || exit 1

# vocab size for sentence piece models.
# It will generate data/lang_bpe_xxx,
# data/lang_bpe_yyy if the array contains xxx, yyy
vocab_sizes=(
#   5000
#   2000
#   1000
  500
)

# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Prepare new BPE based lang with <sil> token from the standard BPE based lang"

  for vocab_size in ${vocab_sizes[@]}; do
    old_lang_dir=data/lang_bpe_${vocab_size}
    new_lang_dir=data/lang_bpe_${vocab_size}_sil

    mkdir -p $new_lang_dir
    # cp $old_lang_dir/{tokens.txt,words.txt,lexicon.txt} $new_lang_dir/.
    # cp $old_lang_dir/{tokens.txt,words.txt} $new_lang_dir/.

    # [[ -z $(grep "<sil>" $new_lang_dir/tokens.txt) ]] && echo "<sil> $(wc -l $new_lang_dir/tokens.txt | cut -d' ' -f2)" >> $new_lang_dir/tokens.txt
    # [[ -z $(grep "!SIL" $new_lang_dir/words.txt) ]] && echo "!SIL $(wc -l $new_lang_dir/words.txt | cut -d' ' -f2)" >> $new_lang_dir/words.txt
    # [[ -z $(grep "!SIL <sil>" $new_lang_dir/lexicon.txt) ]] && echo "!SIL <sil>" >> $new_lang_dir/lexicon.txt
    # [[ -z $(grep "!SIL <sil>" $new_lang_dir/lexicon_disambig.txt) ]] && echo "!SIL <sil>" >> $new_lang_dir/lexicon_disambig.txt

    # (echo '!SIL <sil>'; ) |
    # printf "!SIL <sil>\n" |
    #   cat - $old_lang_dir/lexicon.txt |
    #   sort | uniq > $new_lang_dir/lexicon.txt

    # if [ ! -f $new_lang_dir/L_disambig.pt ]; then
    #   ./local/prepare_lang.py --lang-dir $new_lang_dir \
    #     --sil "<sil>" \
    #     --sil-prob 0.5
    
    if [ ! -f $new_lang_dir/L.pt ]; then
      ./local/prepare_lang_bpe_sil.py \
        --lang-dir $new_lang_dir \
        --old-lang-dir $old_lang_dir \
        --sil-word "!SIL" \
        --sil-token "<sil>"

      # [[ -z $(grep "<sos/eos>" $new_lang_dir/tokens.txt) ]] && echo "<sos/eos> $(wc -l $new_lang_dir/tokens.txt | cut -d' ' -f1)" >> $new_lang_dir/tokens.txt
    fi
  done
fi

exit 0

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  log "Stage 7: Prepare bigram token-level P for MMI training"

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}

    if [ ! -f $lang_dir/transcript_tokens.txt ]; then
      ./local/convert_transcript_words_to_tokens.py \
        --lexicon $lang_dir/lexicon.txt \
        --transcript $lang_dir/transcript_words.txt \
        --oov "<UNK>" \
        > $lang_dir/transcript_tokens.txt
    fi

    if [ ! -f $lang_dir/P.arpa ]; then
      ./shared/make_kn_lm.py \
        -ngram-order 2 \
        -text $lang_dir/transcript_tokens.txt \
        -lm $lang_dir/P.arpa
    fi

    if [ ! -f $lang_dir/P.fst.txt ]; then
      python3 -m kaldilm \
        --read-symbol-table="$lang_dir/tokens.txt" \
        --disambig-symbol='#0' \
        --max-order=2 \
        $lang_dir/P.arpa > $lang_dir/P.fst.txt
    fi
  done
fi

# Because of the mismatched words.txt, I need to compile a new G.fst
if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  log "Stage 8: Prepare G"
  # We assume you have install kaldilm, if not, please install
  # it using: pip install kaldilm

  mkdir -p data/lm2
  if [ ! -f data/lm/G_3_gram.fst.txt ]; then
    # It is used in building HLG
    python3 -m kaldilm \
      --read-symbol-table="data/lang_bpe_500_sil/words.txt" \
      --disambig-symbol='#0' \
      --max-order=3 \
      $dl_dir/lm/3-gram.pruned.1e-7.arpa > data/lm2/G_3_gram.fst.txt
  fi

  if [ ! -f data/lm/G_4_gram.fst.txt ]; then
    # It is used for LM rescoring
    python3 -m kaldilm \
      --read-symbol-table="data/lang_bpe_500_sil/words.txt" \
      --disambig-symbol='#0' \
      --max-order=4 \
      $dl_dir/lm/4-gram.arpa > data/lm2/G_4_gram.fst.txt
  fi
fi

# https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/prepare.sh#L273
# You may need to run it on a b* machine, as it will be killed on c* machines due to large memory usage
if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
  log "Stage 9: Compile HLG"
  # ./local/compile_hlg.py --lang-dir data/lang_phone

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}_sil
    ./local/compile_hlg.py --lang-dir $lang_dir
  done

  lang_dir=data/lang_bpe_500_sil
  python tdnn_lstm_hmm/compile_hlg.py --lang-dir $lang_dir --lm-dir data/lm2 --sil-token "<sil>"
fi

# Compile LG for RNN-T fast_beam_search decoding
if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
  log "Stage 10: Compile LG"
  # ./local/compile_lg.py --lang-dir data/lang_phone

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}_sil
    ./local/compile_lg.py --lang-dir $lang_dir
  done
fi


# cd /export/fs04/a12/rhuang/icefall_align/egs/librispeech/ASR
# conda activate /export/fs04/a12/rhuang/anaconda/anaconda3/envs/espnet_gpu
# export PYTHONPATH=/export/fs04/a12/rhuang/icefall_align/:$PYTHONPATH

