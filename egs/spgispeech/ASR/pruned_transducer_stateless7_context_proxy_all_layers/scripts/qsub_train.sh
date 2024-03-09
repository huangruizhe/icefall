#!/usr/bin/env bash
#$ -wd /exp/rhuang/meta/icefall/egs/spgispeech/ASR/
#$ -V
#$ -N train_context
#$ -j y -o log/log-$JOB_NAME-$JOB_ID.out
#$ -M ruizhe@jhu.edu
#$ -m e
#$ -l mem_free=32G,h_rt=600:00:00,gpu=4,hostname=!r7n07*
#$ -q gpu.q@@v100

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

# exp_dir=/exp/rhuang/meta/icefall/egs/spgispeech/ASR/pruned_transducer_stateless7_context_proxy_all_layers/exp/exp_all_layers  # 11170650
exp_dir=/exp/rhuang/meta/icefall/egs/spgispeech/ASR/pruned_transducer_stateless7_context_proxy_all_layers/exp/exp_all_layers_proxy  
# stage1: 11170651, 11171399 (pruned_scale:10.0), 11171555 (pruned_scale:2.0)
# stage2: 11171772 (full context), 11172050 (0.6)

mkdir -p $exp_dir

echo
echo "exp_dir:" $exp_dir
echo

if [ ! -f $exp_dir/epoch-1.pt ]; then
  ln -s /exp/rhuang/icefall_latest/egs/spgispeech/ASR/pruned_transducer_stateless7/exp_500_norm/pretrained.pt $exp_dir/epoch-1.pt
fi

####################################
# train
####################################

# if false; then
#    echo "True"
# else
#    echo "False"
# fi

if true; then
    # ####  stage1:
    # n_distractors=0
    # max_duration=1200
    # is_full_context=true

    ####  stage2:
    n_distractors=100
    max_duration=1200
    is_full_context=false

    python /exp/rhuang/meta/icefall/egs/spgispeech/ASR/pruned_transducer_stateless7_context_proxy_all_layers/train.py \
      --world-size 4 \
      --use-fp16 true \
      --max-duration $max_duration \
      --exp-dir $exp_dir \
      --bpe-model "data/lang_bpe_500/bpe.model" \
      --prune-range 5 \
      --use-fp16 true \
      --context-dir "data/uniphore_contexts/" \
      --keep-ratio 1.0 \
      --start-epoch 2 \
      --num-epochs 30 \
      --is-bi-context-encoder false \
      --is-pretrained-context-encoder false \
      --is-full-context $is_full_context \
      --n-distractors $n_distractors --master-port 12378 --proxy-prob 0.2 --throwaway-prob 0.7 --early-layers 3  --start-epoch 11 #--start-batch 128000
fi

####################################
# tensorboard
####################################
# tensorboard dev upload --logdir /exp/rhuang/meta/icefall/egs/librispeech/ASR/$exp_dir/tensorboard --description `pwd`
# wandb sync $exp_dir/tensorboard

# https://github.com/k2-fsa/icefall/issues/1298
# python -c "import wandb; wandb.init(project='icefall-asr-gigaspeech-zipformer-2023-10-20')"
# wandb sync zipformer/exp/tensorboard -p icefall-asr-gigaspeech-zipformer-2023-10-20

# https://stackoverflow.com/questions/37987839/how-can-i-run-tensorboard-on-a-remote-server
# ssh -L 16006:127.0.0.1:6006 rhuang@test1.hltcoe.jhu.edu
# tensorboard --logdir $exp_dir/tensorboard --port 6006
# http://localhost:16006 

# no-biasing: /exp/rhuang/icefall_latest/egs/spgispeech/ASR/pruned_transducer_stateless7/exp_500_norm/tensorboard/
# bi_enc: /exp/rhuang/icefall_latest/egs/spgispeech/ASR/pruned_transducer_stateless7_context/exp/exp_libri_full_c-1_stage1/
# single_enc: /exp/rhuang/icefall_latest/egs/spgispeech/ASR/pruned_transducer_stateless7_context/exp/exp_libri_full_c-1_stage1_single_enc


# exp_dir=pruned_transducer_stateless7_context_ali/exp
# n_distractors=0
# max_duration=1200
# python /exp/rhuang/meta/icefall/egs/spgispeech/ASR/pruned_transducer_stateless7_context_ali/train.py       --world-size 1       --use-fp16 true  --max-duration $max_duration       --exp-dir $exp_dir       --bpe-model "data/lang_bpe_500/bpe.model"       --prune-range 5       --use-fp16 true       --context-dir "data/uniphore_contexts/"       --keep-ratio 1.0       --start-epoch 2       --num-epochs 30       --is-bi-context-encoder false       --is-pretrained-context-encoder false       --is-full-context true       --n-distractors $n_distractors