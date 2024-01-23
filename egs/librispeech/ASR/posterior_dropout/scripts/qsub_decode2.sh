#!/usr/bin/env bash
#$ -wd /exp/rhuang/meta/icefall/egs/librispeech/ASR/
#$ -V
#$ -N decode_posterior_dropout
#$ -j y -o /exp/rhuang/meta/icefall/egs/librispeech/ASR/log/log-$JOB_NAME-$JOB_ID.out
#$ -M ruizhe@jhu.edu
#$ -m e
#$ -l mem_free=20G,h_rt=600:00:00,gpu=1
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

# exp_dir=posterior_dropout/exp-ctc
# exp_dir=posterior_dropout/exp-transducer
# exp_dir=posterior_dropout/exp-transducer-dp0.3-chng0.8-libri100
# exp_dir=posterior_dropout/exp-transducer-dp0.5-chng0.8-libri100
exp_dir=posterior_dropout/exp-transducer-libri100-rd-s1.0-p0.5/

echo
echo "exp_dir:" $exp_dir
echo

####################################
# decode ctc
####################################
# for m in ctc-decoding 1best nbest nbest-rescoring whole-lattice-rescoring; do
#   python posterior_dropout/ctc_decode.py \
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
# train transducer
####################################
# fast_beam_search fast_beam_search_nbest 
for m in fast_beam_search fast_beam_search_nbest modified_beam_search; do
  if [ "$m" = "modified_beam_search" ]; then
    ./posterior_dropout/decode.py \
      --epoch 30 \
      --avg 10 \
      --exp-dir $exp_dir \
      --max-duration 600 \
      --decoding-method modified_beam_search \
      --beam-size 8
  elif [ "$m" = "fast_beam_search" ]; then
    ./posterior_dropout/decode.py \
      --epoch 30 \
      --avg 10 \
      --exp-dir $exp_dir \
      --max-duration 600 \
      --decoding-method fast_beam_search \
      --beam 20.0 \
      --max-contexts 8 \
      --max-states 64
  elif [ "$m" = "fast_beam_search_nbest" ]; then
    ./posterior_dropout/decode.py \
      --epoch 30 \
      --avg 10 \
      --exp-dir $exp_dir \
      --max-duration 600 \
      --decoding-method fast_beam_search_nbest \
      --beam 20.0 \
      --max-contexts 8 \
      --max-states 64 \
      --num-paths 200 \
      --nbest-scale 0.5
  fi
done


####################################
# tensorboard
####################################
# tensorboard dev upload --logdir /exp/rhuang/meta/icefall/egs/librispeech/ASR/$exp_dir/tensorboard --description `pwd`
