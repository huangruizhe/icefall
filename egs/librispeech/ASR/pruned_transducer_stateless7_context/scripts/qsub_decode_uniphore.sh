#!/usr/bin/env bash
#$ -wd /exp/rhuang/meta/icefall/egs/librispeech/ASR/
#$ -V
#$ -N decode_context
#$ -j y -o /exp/rhuang/meta/icefall/egs/librispeech/ASR/log/log-$JOB_NAME-$JOB_ID.out
#$ -M ruizhe@jhu.edu
#$ -m e
#$ -l mem_free=20G,h_rt=600:00:00,gpu=1,hostname=!r7n07*
#$ -q gpu.q@@rtx

# #$ -q gpu.q@@v100
# #$ -q gpu.q@@rtx

# #$ -l ram_free=300G,mem_free=300G,gpu=0,hostname=b*

# hostname=b19
# hostname=!c04*&!b*&!octopod*
# hostname
# nvidia-smi

# conda activate aligner5
export PATH="/home/hltcoe/rhuang/mambaforge/envs/aligner5/bin/":$PATH
module load cuda11.7/toolkit
module load cudnn/8.5.0.96_cuda11.x
module load nccl/2.13.4-1_cuda11.7
module load gcc/7.2.0
module load intel/mkl/64/2019/5.281

which python
nvcc --version
nvidia-smi
date

# k2
K2_ROOT=/exp/rhuang/meta/k2/
export PYTHONPATH=$K2_ROOT/k2/python:$PYTHONPATH # for `import k2`
export PYTHONPATH=$K2_ROOT/temp.linux-x86_64-cpython-310/lib:$PYTHONPATH # for `import _k2`
export PYTHONPATH=/exp/rhuang/meta/icefall:$PYTHONPATH

# # torchaudio recipe
# cd /exp/rhuang/meta/audio
# cd examples/asr/librispeech_conformer_ctc

# To verify SGE_HGR_gpu and CUDA_VISIBLE_DEVICES match for GPU jobs.
env | grep SGE_HGR_gpu
env | grep CUDA_VISIBLE_DEVICES
echo "hostname: `hostname`"
echo "current path:" `pwd`

# export PYTHONPATH=/exp/rhuang/meta/audio/examples/asr/librispeech_conformer_ctc2:$PYTHONPATH

exp_dir=/exp/rhuang/icefall_latest/egs/librispeech/ASR/pruned_transducer_stateless7_context/exp/exp_libri_full_c-1_stage2/

echo
echo "exp_dir:" $exp_dir
echo

####################################
# decode ctc
####################################
# for m in ctc-decoding 1best nbest nbest-rescoring whole-lattice-rescoring; do
#   python pruned_transducer_stateless7_context/ctc_decode.py \
#       --epoch 40 \
#       --avg 16 \
#       --exp-dir $exp_dir \
#       --use-transducer false \
#       --use-ctc true \
#       --max-duration 300 \
#       --causal 0 \
#       --num-paths 100 \
#       --nbest-scale 1.0 \
#       --hlg-scale 0.6 \
#       --decoding-method $m
# done


####################################
# decode transducer
####################################
# epochs=30
# avgs=1
# use_averaged_model=$([ "$avgs" = 1 ] && echo "false" || echo "true")

# for m in modified_beam_search; do
#   if [ "$m" = "modified_beam_search" ]; then
#     ./pruned_transducer_stateless7_context/decode.py \
#       --epoch $epochs \
#       --avg $avgs \
#       --use-averaged-model $use_averaged_model \
#       --exp-dir $exp_dir \
#       --max-duration 600 \
#       --decoding-method modified_beam_search \
#       --beam-size 4
#   elif [ "$m" = "fast_beam_search" ]; then
#     ./pruned_transducer_stateless7_context/decode.py \
#       --epoch 40 \
#       --avg 16 \
#       --exp-dir $exp_dir \
#       --max-duration 600 \
#       --decoding-method fast_beam_search \
#       --beam 20.0 \
#       --max-contexts 8 \
#       --max-states 64
#   elif [ "$m" = "fast_beam_search_nbest" ]; then
#     ./pruned_transducer_stateless7_context/decode.py \
#       --epoch 40 \
#       --avg 16 \
#       --exp-dir $exp_dir \
#       --max-duration 600 \
#       --decoding-method fast_beam_search_nbest \
#       --beam 20.0 \
#       --max-contexts 8 \
#       --max-states 64 \
#       --num-paths 200 \
#       --nbest-scale 0.5
#   fi
# done


####################################
# decode transducer with contexts
####################################
n_distractors=100
epochs=30
avgs=1
use_averaged_model=$([ "$avgs" = 1 ] && echo "false" || echo "true")
manifest=/exp/draj/jsalt2023/icefall/egs/librispeech/ASR/data/manifests/

stage=1
stop_stage=1
echo "Stage: $stage"

cuda_id=`env | grep CUDA_VISIBLE_DEVICES | cut -d'=' -f2`

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
        # ./pruned_transducer_stateless7_context/decode.py \
        ./pruned_transducer_stateless7_context/decode_uniphore.py \
            --epoch $epoch \
            --avg $avg \
            --use-averaged-model $use_averaged_model \
            --exp-dir $exp_dir \
            --manifest-dir $manifest \
            --feedforward-dims  "1024,1024,2048,2048,1024" \
            --max-duration 600 \
            --decoding-method $m \
            --context-dir "data/uniphore_contexts/" \
            --n-distractors $n_distractors \
            --is-predefined false --n-distractors 500 --no-encoder-biasing true --no-decoder-biasing true --no-wfst-lm-biasing false --biased-lm-scale 26
            # --keep-ratio 1.0 --is-predefined true --n-distractors 100 --is-reused-context-encoder true
        # --is-full-context true
        # --n-distractors 0
        # --no-encoder-biasing true --no-decoder-biasing true
        # --is-predefined true
        # --is-pretrained-context-encoder true
        # --no-wfst-lm-biasing false --biased-lm-scale 9
        # --is-predefined true --no-wfst-lm-biasing false --biased-lm-scale 9 --no-encoder-biasing true --no-decoder-biasing true
        # --cuda-id $cuda_id \
        #
        # LM exp: --is-predefined true --n-distractors 500 --no-encoder-biasing true --no-decoder-biasing true --no-wfst-lm-biasing false --biased-lm-scale 45
      done
    done
  done
fi

####################################
# tensorboard
####################################
# tensorboard dev upload --logdir /exp/rhuang/meta/icefall/egs/librispeech/ASR/$exp_dir/tensorboard --description `pwd`

####################################
# Evaluation (Rare WER)
####################################

# # conda activate /home/hltcoe/rhuang/mambaforge/envs/aligner5
# export PATH="/home/hltcoe/rhuang/mambaforge/envs/aligner5/bin/":$PATH
# export PYTHONPATH=/exp/rhuang/meta/icefall:$PYTHONPATH
# per_utt=/exp/rhuang/meta/audio_ruizhe/whisper/exp/uniphore/whisper_tiny/recogs-banking-whisper_tiny-ruizhe.txt
# contexts=/exp/rhuang/icefall_latest/egs/spgispeech/ASR/data/rare_words/
# contexts=/exp/rhuang/icefall_latest/egs/spgispeech/ASR/data/swbd_rare_words/
# contexts=/exp/rhuang/icefall_latest/egs/spgispeech/ASR/data/libri_rare_words/
# cd /exp/rhuang/meta/icefall/egs/spgispeech/ASR/pruned_transducer_stateless7_context
# python ./scripts/compute_rare_wer.py   --context-dir $contexts   --lang-dir ../data/lang_bpe_500   --per-utt $per_utt
