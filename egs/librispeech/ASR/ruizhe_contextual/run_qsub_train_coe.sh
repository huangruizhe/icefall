#!/usr/bin/env bash
#$ -wd /exp/rhuang/icefall_latest/egs/librispeech/ASR
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

# max_duration=1400
# max_duration=1200
# max_duration=1000
# n_distractors=100
# n_distractors=0
# n_distractors=-1
# max_duration=700
n_distractors=-1
max_duration=1000
# n_distractors=500
# max_duration=100

full_libri=true
full_libri_name=$([ "$full_libri" = true ] && echo "full" || echo "100")


# Continue training from epoch-30.pt -- this is not optimal!
# path_to_pretrained_asr_model=/exp/rhuang/librispeech/pretrained2/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11/
# exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c${n_distractors}
# mkdir -p $exp_dir
# if [ ! -f $exp_dir/epoch-1.pt ]; then
#   ln -s $path_to_pretrained_asr_model/exp/epoch-30.pt $exp_dir/epoch-1.pt
# fi

# Continue training from pretrained.pt
path_to_pretrained_asr_model=/exp/rhuang/librispeech/pretrained2/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11/
# exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c${n_distractors}_continue3
# exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c${n_distractors}_bert_stage1
# exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c${n_distractors}_fasttext_stage1
# exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c${n_distractors}_continue4
# exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c${n_distractors}_stage1
# exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c${n_distractors}_stage2
# exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c${n_distractors}_no_stage1
# exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c${n_distractors}_stage1.4
exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c0_reuse_decoder_stage1
exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c100_reuse_decoder_stage2
# exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c-1_reuse_decoder_stage2
mkdir -p $exp_dir

# # From pretrained ASR model
# if [ ! -f $exp_dir/epoch-1.pt ]; then
#   ln -s $path_to_pretrained_asr_model/exp/pretrained.pt $exp_dir/epoch-1.pt
# fi

# # Epoch 10/30 from stage 1:
# if [ ! -f $exp_dir/epoch-1.pt ]; then
#   ln -s /exp/rhuang/icefall_latest/egs/librispeech/ASR/pruned_transducer_stateless7_context/exp/exp_libri_full_c-1_stage1/epoch-30.pt $exp_dir/epoch-1.pt
# fi

# Continue training from the wrong model
# exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c${n_distractors}_continue3
# mkdir -p $exp_dir
# ln -s /exp/rhuang/icefall_latest/egs/librispeech/ASR/pruned_transducer_stateless7_context/exp/exp_libri_full_lowerwrong/epoch-30.pt \
#   $exp_dir/epoch-1.pt

python pruned_transducer_stateless7_context/train.py \
  --world-size 4 \
  --full-libri $full_libri \
  --use-fp16 true \
  --max-duration $max_duration \
  --exp-dir $exp_dir \
  --feedforward-dims  "1024,1024,2048,2048,1024" \
  --master-port 12535 \
  --num-workers 10 \
  --base-lr 0.1 \
  --context-dir "data/fbai-speech/is21_deep_bias/" \
  --keep-ratio 1.0 \
  --start-epoch 2 \
  --num-epochs 30 \
  --n-distractors $n_distractors --n-distractors 100 --is-full-context false --is-reused-context-encoder true

# Stage1: --n-distractors 0 --is-full-context true
# Stage1.4: --n-distractors 0 --is-full-context true --keep-ratio 0.9
# Stage2: --n-distractors 100 --is-full-context false --start-epoch
# --start-batch 
# --is-pretrained-context-encoder true
# --is-reused-context-encoder true



# _continue: continue with full context from epoch-30.pt
# _continue2: continue with random context from _continue
# _continue3: continue with full context from pretrained.pt

# context size 100 (wrong due to lowercased biasing list):
# /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe_contextual/log/log-train-10576003.out
# /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe_contextual/log/log-train-10577155.out
# /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe_contextual/log/log-train-10578161.out

# context size 500 (wrong due to lowercased biasing list):
# /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe_contextual/log/log-train-10579396.out
# /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe_contextual/log/log-train-10579398.out

# context size 100
# /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe_contextual/log/log-train-10579501.out

# context size 100 (trained from the wrong model)
# /export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/ruizhe_contextual/log/train-3611947.out
# /export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/ruizhe_contextual/log/train-3611979.out
# /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe_contextual/log/log-train-10579608.out  # continue training from the full context model
# /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe_contextual/log/log-train-10580874.out  # continue2, but use context_generator matching fbai_is_21 biasing list
# /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe_contextual/log/log-train-10581036.out  # continue3, trained from pretrained.pt
# /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe_contextual/log/log-train-10581683.out

# continue3 (train from averaged pretrained model instead of epoch-30.pt):
# - Stage1:   --start-epoch 2 --num-epochs 30 --n-distractors 0 --is-full-context true
#             --base-lr 0.15 => 0.1 => 0.05
#           https://tensorboard.dev/experiment/7PzL4cpxTgGup6Vp15FqMA/#scalars
#           /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe_contextual/log/log-train-10581036.out
#           
# - Stage2:   --start-epoch 2 --num-epochs 30 --base-lr 0.1=>0.06 --n-distractors 100
#           /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe_contextual/log/log-train-10583140.out
#           https://tensorboard.dev/experiment/9lnyyTJDR9mlf0fwCLL12w/#scalars
#
# - Stage2:   --start-epoch 2 --num-epochs 30 --base-lr 0.1=>0.06 --n-distractors -1 (80,1000)
#           /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe_contextual/log/log-train-10583446.out
#           https://tensorboard.dev/experiment/zyVJCxbDQrm2kG9pwbqVWQ/

# continue4: train stage2 directly from the pretrained ASR model, skipping stage1
# - Stage2:   --start-epoch 2 --num-epochs 30 --base-lr 0.12 --n-distractors -1
#             --base-lr 0.12: 
#             /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe_contextual/log/log-train-10584053.out
#             https://tensorboard.dev/experiment/h1aLWotQQymmRHjJ2s5t6Q/
#             --base-lr 0.08: => actually, base-lr did not take effect...
#             /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe_contextual/log/log-train-10584057.out
#             https://tensorboard.dev/experiment/HnzjFPJ6Qeq6V63XpvLKjw/
#             https://tensorboard.dev/experiment/MNMWFd0aQsKxuDBpPktqkw/
#             continue beyond 30 epochs:
#             /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe_contextual/log/log-train-10585306.out
#             /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe_contextual/log/log-train-10585321.out
#             https://tensorboard.dev/experiment/DJwVlmGgSJG6ztXU4eOL9Q/
#             The validation loss has reduced to 0.1286 but WER is not good: 
#             rd: 3.01(2.06/10.76), 4.57(3.38/15.01)
#             pd: 3.11(2.16/10.81), 4.68(3.45/15.48)

# BERT: with dropout=0.3 and relu layer, no full context
# max_duration=120, world_size=8
# /export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/ruizhe_contextual/log/train-3613914.out
# https://tensorboard.dev/experiment/fp13QHZaRZqPZxv38u2ZOg/
# resume from epoch 3 if you want to run it again

# BERT: stage1
# /export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/ruizhe_contextual/log/train-3613915.out
# https://tensorboard.dev/experiment/NGRw0MYMS5WU5NPMyBYUEA/
# https://tensorboard.dev/experiment/0Gl8ijfDS9eAECYqLxmP2w/ => clsp grid 8 GPUs
#
# /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe_contextual/log/log-train-10584971.out
# https://tensorboard.dev/experiment/PAXLxp3nS5mJWQdE2sMGaA/ => COE grid, [with non-linear activation and dropout]=>was commented out
#
# The model did not learn anything.
# WER with full context and no distractors (epoch 16 on clsp grid): test-clean 2.24(1.36/9.39); test-other 5.22(3.41/21.16)
# with 100 distractors: 2.59(1.68/10.00); 5.93(4.08/22.15)

# Train with the "predefined/random bug" fixed:
# Stage1:
# COE: /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe_contextual/log/log-train-10585355.out
#      /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe_contextual/log/log-train-10585355.out => increased learning rate
# CLSP (context_dim=256, drop_out=0.1): /export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/ruizhe_contextual/log/train-3614171.out
#
# Stage2:
# COE: /exp/draj/mini_scale_2022/icefall/egs/librispeech/ASR/train_ruizhe.sh
#      /exp/draj/mini_scale_2022/icefall/egs/librispeech/ASR/pruned_transducer_stateless7_context/exp/log/log-train-10585484.out => stage2 from epoch-10.pt of stage1
# COE: /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe_contextual/log/log-train-10586081.out => stage2 from epoch-30.pt of stage1
#
# Stage2 without stage1:
# COE: /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe_contextual/log/log-train-10587396.out
#
# Results:
# exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c-1_stage1
# exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c-1_stage2
# exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c-1_stage2_10pt/
# exp_dir=pruned_transducer_stateless7_context/exp/exp_libri_full_c-1_no_stage1
#


# BERT: 
# stage1
# /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe_contextual/log/log-train-10600123.out
# /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe_contextual/log/log-train-10600362.out
#
# go stage2 directly:
# /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe_contextual/log/log-train-10600438.out
# /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe_contextual/log/log-train-10602901.out


# Tensorboard:
# ln -s /exp/rhuang/librispeech/pretrained2/icefall-asr-librispeech-pruned-transducer-stateless7-2022-11-11/exp/tensorboard/events.out.tfevents.1667739880.de-74279-k2-train-2-0602201035-5fb6d86964-mclm7.2931411.0 pruned_transducer_stateless7_context/exp/exp_libri_full_c100_reuse_decoder_nostage1/tensorboard/.