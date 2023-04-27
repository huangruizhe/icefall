#!/usr/bin/env bash
#$ -wd /exp/rhuang/icefall_latest/egs/spgispeech/ASR/
#$ -V
#$ -N train
#$ -j y -o ruizhe_contextual/log/log-$JOB_NAME-$JOB_ID.out
#$ -M ruizhe@jhu.edu
#$ -m e
#$ -l mem_free=20G,h_rt=600:00:00,gpu=4
#$ -q gpu.q@@v100

# #$ -q gpu.q@@v100
# #$ -q gpu.q@@rtx

# #$ -l ram_free=300G,mem_free=300G,gpu=0,hostname=b*

# hostname=b19
# hostname=!c04*&!b*&!octopod*
hostname
nvidia-smi

export PATH="/exp/rhuang/mambaforge/envs/icefall2/bin/":$PATH
which python
export PATH="/exp/rhuang/gcc-7.2.0/mybuild/bin":$PATH
export PATH=/exp/rhuang/cuda-10.2/bin/:$PATH
which nvcc
export LD_LIBRARY_PATH=/exp/rhuang/cuda-10.2/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/exp/rhuang/gcc-7.2.0/mybuild/lib64:$LD_LIBRARY_PATH

# k2
K2_ROOT=/exp/rhuang/k2/
export PYTHONPATH=$K2_ROOT/k2/python:$PYTHONPATH # for `import k2`
export PYTHONPATH=$K2_ROOT/build_debug_cuda10.2/lib:$PYTHONPATH # for `import _k2`

# kaldifeat
export PYTHONPATH=/exp/rhuang/kaldifeat/build/lib:/exp/rhuang/kaldifeat/kaldifeat/python:$PYTHONPATH

# icefall
# export PYTHONPATH=/exp/rhuang/icefall/:$PYTHONPATH
export PYTHONPATH=/exp/rhuang/icefall_latest/:$PYTHONPATH
# export PYTHONPATH=/exp/rhuang/icefall/icefall/transformer_lm/:$PYTHONPATH

# To verify SGE_HGR_gpu and CUDA_VISIBLE_DEVICES match for GPU jobs.
env | grep SGE_HGR_gpu
env | grep CUDA_VISIBLE_DEVICES
echo "hostname: `hostname`"

# python pruned_transducer_stateless7/train.py \
#   --world-size 4 \
#   --num-epochs 30 \
#   --full-libri false \
#   --use-fp16 true \
#   --max-duration 750 \
#   --exp-dir pruned_transducer_stateless7/exp/exp_libri_100 \
#   --feedforward-dims  "1024,1024,2048,2048,1024" \
#   --master-port 12535

n_distractors=100
n_distractors=0
max_duration=700
max_duration=900
max_duration=1200
n_distractors=-1
# max_duration=700
# n_distractors=500
# max_duration=100


# Stage1: Continue training from pretrained.pt
# path_to_pretrained_asr_model="/exp/rhuang/icefall_latest/egs/spgispeech/ASR/pretrained/icefall-asr-spgispeech-pruned-transducer-stateless2"
# exp_dir=pruned_transducer_stateless2_context/exp/exp_libri_full_c${n_distractors}_stage1
# mkdir -p $exp_dir
# if [ ! -f $exp_dir/epoch-1.pt ]; then
#   ln -s $path_to_pretrained_asr_model/exp/pretrained.pt $exp_dir/epoch-1.pt
# fi

# Stage2
path_to_pretrained_asr_model="/exp/rhuang/icefall_latest/egs/spgispeech/ASR/pruned_transducer_stateless2_context/exp/exp_libri_full_c-1_stage1/"
exp_dir=pruned_transducer_stateless2_context/exp/exp_libri_full_c${n_distractors}_stage2
mkdir -p $exp_dir
if [ ! -f $exp_dir/epoch-1.pt ]; then
  ln -s $path_to_pretrained_asr_model/checkpoint-72000.pt $exp_dir/epoch-1.pt
fi


# ./pruned_transducer_stateless2/train.py \
#   --world-size 8 \
#   --num-epochs 20 \
#   --start-epoch 0 \
#   --exp-dir pruned_transducer_stateless2/exp \
#   --max-duration 200 \
#   --prune-range 5 \
#   --lr-factor 5 \
#   --lm-scale 0.25 \
#   --use-fp16 True

python pruned_transducer_stateless2_context/train.py \
  --world-size 4 \
  --use-fp16 true \
  --max-duration $max_duration \
  --exp-dir $exp_dir \
  --prune-range 5 \
  --use-fp16 true \
  --context-dir "data/rare_words/" \
  --keep-ratio 1.0 \
  --start-epoch 2 \
  --num-epochs 30 \
  --n-distractors $n_distractors

# Stage1: --n-distractors 0 --is-full-context true
# --start-batch 
# --is-pretrained-context-encoder true


# Initial run: max_duration=700 is feasible (Stage1)
# /exp/rhuang/icefall_latest/egs/spgispeech/ASR/ruizhe_contextual/log/log-train-10587871.out
# /exp/rhuang/icefall_latest/egs/spgispeech/ASR/ruizhe_contextual/log/log-train-10587888.out
#    - https://tensorboard.dev/experiment/ZglzzMlsRouPL44pGk3AfA/
# => wrong! 
# data/manifests/cuts_train_shuf.jsonl.gz was not normalized!!
# => moved to: pruned_transducer_stateless2_context/exp/exp_libri_full_c-1_stage1_nonorm

# Stage1:
# /exp/rhuang/icefall_latest/egs/spgispeech/ASR/ruizhe_contextual/log/log-train-10588317.out <= max_duration=1200
#    - https://tensorboard.dev/experiment/uM39UVDdTXyUvJvSwjF9rw/
# /exp/rhuang/icefall_latest/egs/spgispeech/ASR/ruizhe_contextual/log/log-train-10588318.out <= max_duration=900

# Stage2 (common words 6k):
# /exp/rhuang/icefall_latest/egs/spgispeech/ASR/ruizhe_contextual/log/log-train-10588562.out
# /exp/rhuang/icefall_latest/egs/spgispeech/ASR/ruizhe_contextual/log/log-train-10588574.out
#    - https://tensorboard.dev/experiment/L4wlmd9NTy2jFw1Sj7KHWg/
# mv pruned_transducer_stateless2_context/exp/exp_libri_full_c-1_stage2 pruned_transducer_stateless2_context/exp/exp_libri_full_c-1_stage2_6k

# Stage2 (common words 3k):
# /exp/rhuang/icefall_latest/egs/spgispeech/ASR/ruizhe_contextual/log/log-train-10588947.out
#    - https://tensorboard.dev/experiment/f4U3S1lYR0KaAFeEP2xOrg/
# ls pruned_transducer_stateless2_context/exp/exp_libri_full_c-1_stage2

# Stage3 (common words 3k + keep_ratio = 0.3~0.8):
