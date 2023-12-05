#!/usr/bin/env bash
#$ -wd /exp/rhuang/meta/icefall/egs/spgispeech/ASR/
#$ -V
#$ -N train_context
#$ -j y -o log/log-$JOB_NAME-$JOB_ID.out
#$ -M ruizhe@jhu.edu
#$ -m e
#$ -l mem_free=20G,h_rt=600:00:00,gpu=4,hostname=!r7n07*
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

exp_dir=/exp/rhuang/icefall_latest/egs/spgispeech/ASR/pruned_transducer_stateless7_context/exp/exp_libri_full_c-1_stage1_overfitting/
mkdir -p $exp_dir

echo
echo "exp_dir:" $exp_dir
echo

if [ ! -f $exp_dir/epoch-21.pt ]; then
  ln -s /exp/rhuang/icefall_latest/egs/spgispeech/ASR/pruned_transducer_stateless7_context/exp/exp_libri_full_c-1_stage1/epoch-21.pt $exp_dir/epoch-21.pt
fi

####################################
# train for overfitting experiments
####################################

# if false; then
#    echo "True"
# else
#    echo "False"
# fi

if false; then
    n_distractors=0
    max_duration=1200
    python /exp/rhuang/meta/icefall/egs/spgispeech/ASR/pruned_transducer_stateless7_context/train_overfitting_exp.py \
      --world-size 1 \
      --use-fp16 true \
      --max-duration $max_duration \
      --exp-dir $exp_dir \
      --bpe-model "data/lang_bpe_500/bpe.model" \
      --prune-range 5 \
      --use-fp16 true \
      --context-dir "data/uniphore_contexts/" \
      --keep-ratio 1.0 \
      --start-epoch 22 \
      --num-epochs 30 \
      --is-bi-context-encoder true \
      --n-distractors $n_distractors \
      --is-predefined true --slides "/exp/rhuang/meta/audio_ruizhe/ec21/data/earnings21_slides_context_names"

    n_distractors=100
    max_duration=800
    python /exp/rhuang/meta/icefall/egs/spgispeech/ASR/pruned_transducer_stateless7_context/train_overfitting_exp.py \
      --world-size 1 \
      --use-fp16 true \
      --max-duration $max_duration \
      --exp-dir $exp_dir \
      --bpe-model "data/lang_bpe_500/bpe.model" \
      --prune-range 5 \
      --use-fp16 true \
      --context-dir "data/uniphore_contexts/" \
      --keep-ratio 1.0 \
      --start-epoch 22 \
      --num-epochs 30 \
      --is-bi-context-encoder true \
      --n-distractors $n_distractors \
      --is-predefined false
fi

####################################
# tensorboard
####################################
# tensorboard dev upload --logdir /exp/rhuang/meta/icefall/egs/librispeech/ASR/$exp_dir/tensorboard --description `pwd`
# wandb sync $exp_dir/tensorboard

