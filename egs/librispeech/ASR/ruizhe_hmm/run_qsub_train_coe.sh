#!/usr/bin/env bash
#$ -wd /exp/rhuang/icefall_latest/egs/librispeech/ASR
#$ -V
#$ -N train
#$ -j y -o ruizhe/log/log-$JOB_NAME-$JOB_ID.out
#$ -M ruizhe@jhu.edu
#$ -m e
#$ -l mem_free=20G,h_rt=600:00:00,gpu=4
#$ -q gpu.q@@rtx

# #$ -q gpu.q@@v100

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
# export PYTHONPATH=/exp/rhuang/icefall_latest/:$PYTHONPATH
export PYTHONPATH=/exp/rhuang/icefall_align2/:$PYTHONPATH
# export PYTHONPATH=/exp/rhuang/icefall/icefall/transformer_lm/:$PYTHONPATH

# To verify SGE_HGR_gpu and CUDA_VISIBLE_DEVICES match for GPU jobs.
env | grep SGE_HGR_gpu
env | grep CUDA_VISIBLE_DEVICES
echo "hostname: `hostname`"

# full librispeech
# python tdnn_lstm_ctc/train.py \
#     --world-size 4 \
#     --full-libri true \
#     --max-duration 200 \
#     --num-epochs 30 \
#     --exp-dir "tdnn_lstm_ctc/exp/exp_libri_full"

# libri100
# python tdnn_lstm_ctc/train.py \
#     --world-size 4 \
#     --full-libri false \
#     --max-duration 400 \
#     --num-epochs 30 \
#     --valid-interval 100 \
#     --exp-dir "tdnn_lstm_ctc/exp/exp_libri_100"
#
# /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe/log/log-decode-10565878.out
# lm_scale_0.9    16.18   best for test-clean
# lm_scale_0.9    43.78   best for test-other

# # zipformer_ctc on libri100
# ./zipformer_ctc/train.py \
#   --world-size 4 \
#   --master-port 12345 \
#   --num-epochs 30 \
#   --start-epoch 1 \
#   --lang-dir data/lang_bpe_500 \
#   --exp-dir zipformer_ctc/exp/exp_libri_100  \
#   --max-duration 500 \
#   --full-libri false \
#   --use-fp16 true

# zipformer_mmi on libri100
# ./zipformer_mmi/train.py \
#   --world-size 4 \
#   --master-port 12345 \
#   --num-epochs 30 \
#   --start-epoch 1 \
#   --lang-dir data/lang_bpe_500 \
#   --exp-dir zipformer_mmi/exp/exp_libri_100  \
#   --max-duration 500 \
#   --full-libri false \
#   --use-fp16 true

# zipformer_hmm_mmi on libri100
# ./zipformer_mmi_hmm/train.py \
#   --world-size 4 \
#   --master-port 12345 \
#   --num-epochs 30 \
#   --start-epoch 1 \
#   --lang-dir data/lang_bpe_500 \
#   --exp-dir zipformer_mmi_hmm/exp/exp_libri_100  \
#   --max-duration 500 \
#   --full-libri false \
#   --use-fp16 true \
#   --valid-interval 200

# zipformer_hmm_ml on libri100
# ./zipformer_mmi_hmm/train.py \
#   --world-size 4 \
#   --master-port 12345 \
#   --num-epochs 30 \
#   --start-epoch 1 \
#   --lang-dir data/lang_bpe_500 \
#   --exp-dir zipformer_mmi_hmm/exp/exp_libri_100_ml  \
#   --max-duration 500 \
#   --full-libri false \
#   --use-fp16 true \
#   --warm-step 90000000000 \
#   --num-workers 6 \
#   --ctc-beam-size 15 \
#   --sil-modeling true

  # --enable-spec-aug false \

ngpus=4
python zipformer_mmi/train.py \
  --world-size $ngpus \
  --master-port 12346 \
  --num-epochs 30 \
  --start-epoch 1 \
  --lang-dir data/lang_bpe_500 \
  --exp-dir zipformer_mmi/exp/exp_libri_100_hmm  \
  --max-duration 1000 \
  --use-fp16 true \
  --save-every-n 20000 --exp-dir zipformer_mmi/exp/exp_ctc_mmi --topo-type "ctc" --warm-step 2000 --start-epoch 3 # --shuffle false --bucketing-sampler false --curriculum true
# --warm-step 90000000000
# --shuffle false --bucketing-sampler false
# --start-epoch 3
# --full-libri false
# --base-lr 0.025






