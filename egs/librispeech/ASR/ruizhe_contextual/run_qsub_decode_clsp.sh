#!/usr/bin/env bash
#$ -wd /export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR
#$ -V
#$ -N decode
#$ -j y -o ruizhe_contextual/log/$JOB_NAME-$JOB_ID.out
#$ -M ruizhe@jhu.edu
#$ -m e
#$ -l ram_free=16G,mem_free=16G,gpu=1,hostname=!b*&!c18*&!c04*
#$ -q g.q

# &!octopod*

#### Activate dev environments and call programs
# mamba activate /home/rhuang/mambaforge/envs/efrat
mamba activate /home/rhuang/mambaforge/envs/efrat2
export PYTHONPATH=/export/fs04/a12/rhuang/k2/k2/python:$PYTHONPATH # for `import k2`
export PYTHONPATH=/export/fs04/a12/rhuang/k2/build/temp.linux-x86_64-cpython-38/lib/:$PYTHONPATH # for `import _k2`
# export PYTHONPATH=/export/fs04/a12/rhuang/icefall_align/:$PYTHONPATH
# export PYTHONPATH=/export/fs04/a12/rhuang/icefall/:$PYTHONPATH
export PYTHONPATH=/export/fs04/a12/rhuang/icefall_align2/:$PYTHONPATH

echo "python: `which python`"

#### Assign a free-GPU to your program (make sure -n matches the requested number of GPUs above)
source /home/gqin2/scripts/acquire-gpu 1

# ngpus=4 # num GPUs for multiple GPUs training within a single node; should match those in $free_gpu
# free_gpu= # comma-separated available GPU ids, eg., "0" or "0,1"; automatically assigned if on CLSP grid
# [ -z "$free_gpu" ] && [[ $(hostname -f) == *.clsp.jhu.edu ]] && free_gpu=$(free-gpu -n $ngpus) || \
# echo "Unable to get $ngpus GPUs"
# [ -z "$free_gpu" ] && echo "$0: please specify --free-gpu" && exit 1;
# [ $(echo $free_gpu | sed 's/,/ /g' | awk '{print NF}') -ne "$ngpus" ] && \
#  echo "number of GPU ids in --free-gpu=$free_gpu does not match --ngpus=$ngpus" && exit 1;
# export CUDA_VISIBLE_DEVICES="$free_gpu"
echo $CUDA_VISIBLE_DEVICES

#### Test running qsub
hostname
python3 -c "import torch; print(torch.__version__)"
nvidia-smi

####################################
# modified_beam_search
####################################

n_distractors=100
# n_distractors=-1
exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c${n_distractors}_continue3
# exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_wronglower/
# exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c100_bert_stage1
# exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c-1_continue4

# exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c-1_stage1
# exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c-1_stage2_10pt/
exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c-1_stage2
# exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c-1_no_stage1


epochs=17
epochs=30
avgs=1
use_averaged_model=$([ "$avgs" = 1 ] && echo "false" || echo "true")

stage=1
stop_stage=1
echo "Stage: $stage"

# download model from coe
# mkdir -p $exp_dir
# scp -r rhuang@test1.hltcoe.jhu.edu:/exp/rhuang/icefall_latest/egs/librispeech/ASR/pruned_transducer_stateless7_context/exp/exp_libri_full_c100_continue/epoch-7.pt \
#   /export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/${exp_dir}/.

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  # greedy_search fast_beam_search
  for m in modified_beam_search ; do
    for epoch in $epochs; do
      for avg in $avgs; do
        # python -m pdb -c continue
        ./pruned_transducer_stateless7_context/decode.py \
            --epoch $epoch \
            --avg $avg \
            --use-averaged-model $use_averaged_model \
            --exp-dir $exp_dir \
            --feedforward-dims  "1024,1024,2048,2048,1024" \
            --max-duration 600 \
            --decoding-method $m \
            --context-dir "data/fbai-speech/is21_deep_bias/" \
            --n-distractors $n_distractors \
            --keep-ratio 1.0 --is-predefined true --n-distractors 100
        # --is-full-context true
        # --n-distractors 0
        # --no-encoder-biasing true --no-decoder-biasing true
        # --is-predefined true
        # --is-pretrained-context-encoder true
        # --no-wfst-lm-biasing false --biased-lm-scale 9
        # --is-predefined true --no-wfst-lm-biasing false --biased-lm-scale 9 --no-encoder-biasing true --no-decoder-biasing true
      done
    done
  done

  # On CLSP grid:
  # cd /export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR
  # for m in modified_beam_search ; do   for epoch in 10; do     for avg in 1; do       ./pruned_transducer_stateless7_context/decode.py           --epoch $epoch           --avg $avg           --use-averaged-model 0           --exp-dir ./pruned_transducer_stateless7_context/exp/exp_libri_full           --feedforward-dims  "1024,1024,2048,2048,1024"           --max-duration 400           --decoding-method $m;     done;   done; done
  # /export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/pruned_transducer_stateless7_context/exp/exp_libri_full/modified_beam_search/log-decode-epoch-10-avg-1-modified_beam_search-beam-size-4-2023-03-27-17-25-08

  # # eval
  # for part in test-clean test-other; do
  #   echo
  #   echo `date` "part=$part"
  #   hyp_in=$exp_dir/modified_beam_search/recogs-${part}-epoch-$epochs-avg-$avgs-modified_beam_search-beam-size-4.txt
  #   hyp=$exp_dir/modified_beam_search/recogs-${part}-epoch-$epochs-avg-$avgs-modified_beam_search-beam-size-4.hyp.txt
  #   ref=data/fbai-speech/is21_deep_bias/ref/${part}.biasing_${n_distractors}.tsv
    
  #   python ruizhe_contextual/recogs_to_text.py \
  #     --cuts data/fbank/librispeech_cuts_${part}.jsonl.gz \
  #     --input $hyp_in \
  #     --out $hyp

  #   wc $ref $hyp
  #   python data/fbai-speech/is21_deep_bias/score.py \
  #     --refs $ref \
  #     --hyps $hyp
  # done
fi

####################################
# LODR
####################################

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  # path_to_pretrained_asr_model="/export/fs04/a12/rhuang/deep_smoothing/data_librispeech/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11/"
  # ln -s $path_to_pretrained_asr_model/exp/pretrained.pt $path_to_pretrained_asr_model/exp/epoch-999.pt

  path_to_pretrained_nnlm="/export/fs04/a12/rhuang/deep_smoothing/data_librispeech/icefall-librispeech-rnn-lm/"
  # ln -s $path_to_pretrained_nnlm/exp/pretrained.pt $path_to_pretrained_nnlm/exp/epoch-999.pt
  lm_type="rnn"

  # path_to_pretrained_nnlm="/export/fs04/a12/rhuang/deep_smoothing/data_librispeech/icefall-librispeech-transformer-lm"
  # # ln -s $path_to_pretrained_nnlm/exp/pretrained.pt $path_to_pretrained_nnlm/exp/epoch-999.pt
  # lm_type="transformer"

  tokens_ngram_order=2
  for m in modified_beam_search_LODR ; do
    for epoch in $epochs; do
      for avg in $avgs; do
        ./pruned_transducer_stateless7_context/decode.py \
            --epoch $epoch \
            --avg $avg \
            --use-averaged-model $use_averaged_model \
            --exp-dir $exp_dir \
            --lang-dir data/lang_bpe_500 \
            --max-duration 600 \
            --decoding-method $m \
            --beam-size 4 \
            --max-contexts 4 \
            --use-shallow-fusion true \
            --lm-type $lm_type \
            --lm-exp-dir $path_to_pretrained_nnlm/exp \
            --lm-epoch 999 \
            --lm-scale 0.4 \
            --lm-avg 1 \
            --rnn-lm-num-layers 3 \
            --rnn-lm-tie-weights 1 \
            --tokens-ngram $tokens_ngram_order \
            --ngram-lm-scale -0.16 \
            --context-dir "data/fbai-speech/is21_deep_bias/" \
            --n-distractors $n_distractors \
            --keep-ratio 1.0 --no-wfst-lm-biasing false --biased-lm-scale 9 --is-predefined true --n-distractors 100
        # --is-predefined true
        # --no-encoder-biasing true --no-decoder-biasing true
        # --no-wfst-lm-biasing false --biased-lm-scale 10
        # all: --is-predefined true --n-distractors 500 --no-wfst-lm-biasing false --biased-lm-scale 11
      done
    done
  done

  # # eval
  # for part in test-clean test-other; do
  #   echo 
  #   echo `date` "part=$part"
  #   hyp_in=$exp_dir/modified_beam_search_LODR/recogs-${part}-epoch-17-avg-1-modified_beam_search_LODR-beam-size-4-${lm_type}-lm-scale-0.4-LODR-2gram-scale--0.16.txt
  #   hyp=$exp_dir/modified_beam_search_LODR/recogs-${part}-epoch-17-avg-1-modified_beam_search_LODR-beam-size-4-${lm_type}-lm-scale-0.4-LODR-2gram-scale--0.16.hyp.txt
  #   ref=data/fbai-speech/is21_deep_bias/ref/${part}.biasing_${n_distractors}.tsv

  #   python ruizhe_contextual/recogs_to_text.py \
  #     --cuts data/fbank/librispeech_cuts_${part}.jsonl.gz \
  #     --input $hyp_in \
  #     --out $hyp

  #   wc $ref $hyp
  #   python data/fbai-speech/is21_deep_bias/score.py \
  #     --refs $ref \
  #     --hyps $hyp
  # done
fi

####################################
# Export averaged models 
# https://icefall.readthedocs.io/en/latest/model-export/export-model-state-dict.html#how-to-export
####################################

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  python pruned_transducer_stateless7_context/export.py \
    --exp-dir $exp_dir \
    --bpe-model data/lang_bpe_500/bpe.model \
    --epoch $epochs \
    --avg $avgs

  mv $exp_dir/pretrained.pt $exp_dir/stage1.pt
fi
