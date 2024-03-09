#!/usr/bin/env bash
#$ -wd /exp/rhuang/meta/icefall/egs/spgispeech/ASR/
#$ -V
#$ -N decode_context
#$ -j y -o log/log-$JOB_NAME-$JOB_ID.out
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

# conda activate /home/hltcoe/rhuang/mambaforge/envs/aligner5
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

# exp_dir=/exp/rhuang/icefall_latest/egs/spgispeech/ASR/pruned_transducer_stateless7_context/exp/exp_libri_full_c-1_stage1/  # epochs=21
# exp_dir=/exp/rhuang/icefall_latest/egs/spgispeech/ASR/pruned_transducer_stateless7_context/exp/exp_libri_full_c-1_stage1_unfrozen_joiner_only/  # epochs=40

exp_dir=/exp/rhuang/meta/icefall/egs/spgispeech/ASR/pruned_transducer_stateless7_context_proxy_all_layers/exp/exp_all_layers_proxy  

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
epochs=15
epochs=21
# epochs=40
avgs=1
use_averaged_model=$([ "$avgs" = 1 ] && echo "false" || echo "true")
manifest=data/manifests
# manifest=/exp/draj/jsalt2023/icefall/egs/librispeech/ASR/data/manifests/

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
        # ./pruned_transducer_stateless7_context/decode.py \
        # ./pruned_transducer_stateless7_context/decode_uniphore.py \
        # ./pruned_transducer_stateless7_context/decode_ec21.py \
        # ./pruned_transducer_stateless7_context/decode_ec21_slides.py \
        ./pruned_transducer_stateless7_context_proxy_all_layers/decode_ec21_slides.py \
            --epoch $epoch \
            --avg $avg \
            --use-averaged-model $use_averaged_model \
            --exp-dir $exp_dir \
            --manifest-dir $manifest \
            --bpe-model "/exp/rhuang/meta/icefall/egs/spgispeech/ASR/data/lang_bpe_500/bpe.model" \
            --max-duration 400 \
            --decoding-method $m \
            --beam-size 4 \
            --context-dir "/exp/rhuang/meta/icefall/egs/spgispeech/ASR/data/uniphore_contexts/" \
            --n-distractors $n_distractors \
            --is-predefined true --n-distractors 0 --no-encoder-biasing false --no-decoder-biasing false --no-wfst-lm-biasing true --biased-lm-scale 7 --slides "/exp/rhuang/meta/audio_ruizhe/ec21/data/earnings21_slides_context2_names" --is-bi-context-encoder false 
            # --is-predefined false --n-distractors 0 --no-encoder-biasing false --no-decoder-biasing false --no-wfst-lm-biasing true --biased-lm-scale 12 --slides "/exp/rhuang/meta/audio_ruizhe/ec21/data/earnings21_slides_context_names" --is-bi-context-encoder true
            # --is-predefined false --n-distractors 500 --no-encoder-biasing true --no-decoder-biasing true --no-wfst-lm-biasing false --biased-lm-scale 12 --is-bi-context-encoder true
            # --is-predefined false --n-distractors 500 --no-encoder-biasing true --no-decoder-biasing true --no-wfst-lm-biasing false --biased-lm-scale 12 --slides "/exp/rhuang/meta/audio_ruizhe/ec21/data/earnings21_slides_context_names" --is-bi-context-encoder true

            # --is-predefined false --n-distractors 500 --no-encoder-biasing false --no-decoder-biasing false --no-wfst-lm-biasing true --biased-lm-scale 12 --slides "/exp/rhuang/meta/audio_ruizhe/ec21/data/earnings21_slides_context_names" --is-predefined true --is-bi-context-encoder false
            # --is-predefined false --n-distractors 500 --no-encoder-biasing true --no-decoder-biasing true --no-wfst-lm-biasing false --biased-lm-scale 18 --is-bi-context-encoder true
            # --is-predefined false --n-distractors 500 --no-encoder-biasing false --no-decoder-biasing false --no-wfst-lm-biasing false --biased-lm-scale 12 --slides "/exp/rhuang/meta/audio_ruizhe/ec21/data/earnings21_slides_context_names" --is-predefined true --is-bi-context-encoder true
            # --is-predefined false --n-distractors 500 --no-encoder-biasing true --no-decoder-biasing true --no-wfst-lm-biasing false --biased-lm-scale 12 --slides "/exp/rhuang/meta/audio_ruizhe/ec21/data/earnings21_slides_context_names" --is-predefined true

            # --context-dir "/exp/rhuang/meta/icefall/egs/spgispeech/ASR/data/uniphore_contexts/" \
            # --is-full-context true

            # --is-predefined false --n-distractors 500 --no-encoder-biasing true --no-decoder-biasing true --no-wfst-lm-biasing false --biased-lm-scale 14
            # --slides "/exp/rhuang/meta/audio_ruizhe/ec21/data/earnings21_slides_context" --is-predefined true
            # --slides "/exp/rhuang/meta/audio_ruizhe/ec21/data/earnings21_slides_context_names" --is-predefined true
            # --slides "/exp/rhuang/meta/audio_ruizhe/ec21/data/earnings21_slides_context2_names" --is-predefined true
            # --slides "/exp/rhuang/meta/audio_ruizhe/ec21/data/earnings21_biasing_list/distractor_list" --is-predefined true
            # --slides "/exp/rhuang/meta/audio_ruizhe/ec21/data/earnings21_biasing_list/oracle_list" --is-predefined true
            # --is-bi-context-encoder true
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
