#!/usr/bin/env bash
#$ -wd /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR
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

vocab_size=500

# python pruned_transducer_stateless7/decode.py \
#   --epoch 25 \
#   --avg 8 \
#   --use-averaged-model true \
#   --exp-dir pruned_transducer_stateless7/exp_${vocab_size}_norm \
#   --max-duration 400 \
#   --decoding-method modified_beam_search \
#   --beam-size 4 \
#   --bpe-model data/lang_bpe_${vocab_size}/bpe.model

# --epoch 25 \
# --iter 598000 \

# Check out: /export/fs04/a12/rhuang/contextualizedASR/lm/LM/run_rnnlm.sh
# rnnlm_dir="/export/fs04/a12/rhuang/icefall_align/egs/spgispeech/LM/my-rnnlm-exp/"   # --rnn-lm-num-layers 2 --rnn-lm-embedding-dim 800 --rnn-lm-hidden-dim 200 --rnn-lm-tie-weights false --lm-epoch 13 --lm-avg 1 
rnnlm_dir="/export/fs04/a12/rhuang/contextualizedASR/lm/LM/my-rnnlm-exp-1024-3-tied"
# rnnlm_dir="/export/fs04/a12/rhuang/contextualizedASR/lm/LM/my-rnnlm-exp-1024-2-tied"  # --rnn-lm-num-layers 2 --lm-epoch 14 --lm-avg 5
lm_type=rnn
lang_dir="/export/fs04/a12/rhuang/contextualizedASR/lm/LM/my-ngram-exp/bpe"
# rnn_lm_scale=0.2
# ngram_lm_scale=-0.1
rnn_lm_scale=0.35
ngram_lm_scale=-0.3
tokens_ngram_order=2
python pruned_transducer_stateless7/decode.py \
    --epoch 25 \
    --avg 5 \
    --use-averaged-model true \
    --exp-dir "pruned_transducer_stateless7/exp_${vocab_size}_norm" \
    --bpe-model "data/lang_bpe_500/bpe.model" \
    --max-duration 300 \
    --lang-dir $lang_dir \
    --decoding-method modified_beam_search_LODR \
    --beam 4 \
    --max-contexts 4 \
    --use-shallow-fusion true \
    --lm-type $lm_type \
    --lm-exp-dir $rnnlm_dir \
    --lm-epoch 7 \
    --lm-avg 5 \
    --lm-scale $rnn_lm_scale \
    --rnn-lm-embedding-dim 1024 \
    --rnn-lm-hidden-dim 1024 \
    --rnn-lm-num-layers 3 \
    --rnn-lm-tie-weights true \
    --tokens-ngram $tokens_ngram_order \
    --ngram-lm-scale $ngram_lm_scale


# 3627958 0.50067 decode     rhuang       r     04/25/2023 02:01:05 g.q@c24.clsp.jhu.edu               1  598000-10  2.2/2.13 1.25h <-- rnnlm experiment
# 3627967 0.50066 decode     rhuang       r     04/25/2023 02:16:43 g.q@octopod.clsp.jhu.edu           1  598000-5   2.21/2.14
# 3627968 0.50066 decode     rhuang       r     04/25/2023 02:16:57 g.q@c22.clsp.jhu.edu               1  598000-1   2.23/2.18
# 3627969 0.50066 decode     rhuang       r     04/25/2023 02:17:37 g.q@octopod.clsp.jhu.edu           1  598000-15  2.17/2.12
# 3627970 0.50065 decode     rhuang       r     04/25/2023 02:18:17 g.q@c10.clsp.jhu.edu               1  25-3       2.14/2.11
# 3627971 0.50065 decode     rhuang       r     04/25/2023 02:18:23 g.q@octopod.clsp.jhu.edu           1  25-5       2.13/2.1  <--best: /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/log/decode-3627971.out
# 3627972 0.50065 decode     rhuang       r     04/25/2023 02:18:27 g.q@octopod.clsp.jhu.edu           1  25-10      2.17/2.12
# 3627975 0.50064 decode     rhuang       r     04/25/2023 02:28:15 g.q@c24.clsp.jhu.edu               1  25-2       2.15/2.11
# 3627976 0.50064 decode     rhuang       r     04/25/2023 02:28:21 g.q@c21.clsp.jhu.edu               1  25-3       2.14/2.11
# 3627977 0.50064 decode     rhuang       r     04/25/2023 02:28:25 g.q@c17.clsp.jhu.edu               1  25-4       2.15/2.11
# 3628233 0.50061 decode     rhuang       r     04/25/2023 10:08:59 g.q@octopod.clsp.jhu.edu           1  25-6       2.14/2.1
# 3628234 0.50061 decode     rhuang       r     04/25/2023 10:09:07 g.q@c20.clsp.jhu.edu               1  25-7       2.14/2.11
# 3628235 0.50061 decode     rhuang       r     04/25/2023 10:09:11 g.q@octopod.clsp.jhu.edu           1  25-8       2.16/2.11

# Rnnlm exp with 598000-10 
# 3627958 0.50067 decode     rhuang       r     04/25/2023 02:01:05 g.q@c24.clsp.jhu.edu               1  598000-10  2.2/2.13 (baseline)
# 3628257 0.50062 decode     rhuang       r     04/25/2023 10:42:29 g.q@c10.clsp.jhu.edu               1  0.05-0.0   2.16/2.12
# 3628258 0.50062 decode     rhuang       r     04/25/2023 10:42:59 g.q@octopod.clsp.jhu.edu           1  0.1-0.0    2.14/2.13
# 3628260 0.50062 decode     rhuang       r     04/25/2023 10:43:05 g.q@octopod.clsp.jhu.edu           1  0.1-0.05   2.13/2.12 
# 3628261 0.50062 decode     rhuang       r     04/25/2023 10:43:15 g.q@octopod.clsp.jhu.edu           1  0.15-0.0   2.11/2.15
# 3628262 0.50062 decode     rhuang       r     04/25/2023 10:43:19 g.q@octopod.clsp.jhu.edu           1  0.15-0.05  2.1/2.13
# 3628263 0.50062 decode     rhuang       r     04/25/2023 10:43:23 g.q@octopod.clsp.jhu.edu           1  0.15-0.1   2.12/2.12
# 3628264 0.50062 decode     rhuang       r     04/25/2023 10:43:29 g.q@c25.clsp.jhu.edu               1  0.2-0.0    2.1/2.2
# 3628265 0.50062 decode     rhuang       r     04/25/2023 10:43:35 g.q@c17.clsp.jhu.edu               1  0.2-0.05   2.09/2.15
# 3628266 0.50062 decode     rhuang       r     04/25/2023 10:43:39 g.q@c01.clsp.jhu.edu               1  0.2-0.1    2.1/2.14
# 3628267 0.50062 decode     rhuang       qw    04/25/2023 10:43:42                                    1  0.2-0.15   2.1/2.14
# 3628268 0.50062 decode     rhuang       qw    04/25/2023 10:43:47                                    1  0.25-0     2.16/2.26
# 3628269 0.50062 decode     rhuang       qw    04/25/2023 10:43:52                                    1  0.25-0.05  2.1/2.2
# 3628270 0.50062 decode     rhuang       qw    04/25/2023 10:43:57                                    1  0.25-0.1   2.08/2.17
# 3628271 0.50062 decode     rhuang       qw    04/25/2023 10:43:59                                    1  0.25-0.15  2.1/2.15
# 3628272 0.50062 decode     rhuang       qw    04/25/2023 10:44:10                                    1  0.3-0.0    2.24/2.36
# 3628273 0.50062 decode     rhuang       qw    04/25/2023 10:44:15                                    1  0.3-0.05   2.14/2.25
# 3628274 0.50062 decode     rhuang       qw    04/25/2023 10:44:18                                    1  0.3-0.1    2.09/2.2
# 3628275 0.50062 decode     rhuang       qw    04/25/2023 10:44:21                                    1  0.3-0.15   2.1/2.18
# 3628276 0.50062 decode     rhuang       qw    04/25/2023 10:44:25                                    1  0.3-0.2    2.09/2.16
# 3628277 0.50062 decode     rhuang       qw    04/25/2023 10:44:30                                    1  -
# 3628278 0.50062 decode     rhuang       qw    04/25/2023 10:44:34                                    1  0.35-0.05  2.22/2.35
# 3628279 0.50062 decode     rhuang       qw    04/25/2023 10:44:37                                    1  0.35-0.1   2.14/2.25
# 3628280 0.50062 decode     rhuang       qw    04/25/2023 10:44:40                                    1  0.35-0.15  2.11/2.21
# 3628281 0.50062 decode     rhuang       qw    04/25/2023 10:44:44                                    1  0.35-0.2   2.11/2.19
# 3628282 0.50062 decode     rhuang       qw    04/25/2023 10:44:48                                    1  0.35-0.25  2.1/2.19
# 3628283 0.50062 decode     rhuang       qw    04/25/2023 10:44:57                                    1  0.4-0.05   2.31/2.48
# 3628284 0.50062 decode     rhuang       qw    04/25/2023 10:45:01                                    1  0.4-0.1    2.2/2.34
# 3628285 0.50062 decode     rhuang       qw    04/25/2023 10:45:04                                    1  0.4-0.15   2.15/2.26
# 3628286 0.50062 decode     rhuang       qw    04/25/2023 10:45:07                                    1  0.4-0.2    2.13/2.23
# 3628287 0.50062 decode     rhuang       qw    04/25/2023 10:45:10                                    1  0.4-0.25   2.11/2.22
# 3628288 0.50062 decode     rhuang       qw    04/25/2023 10:45:22                                    1  0.45-0.1   2.29/2.47
# 3628289 0.50062 decode     rhuang       qw    04/25/2023 10:45:25                                    1  0.45-0.15  2.19/-
# 3628290 0.50062 decode     rhuang       qw    04/25/2023 10:45:30                                    1  0.45-0.2   2.16/2.28
# 3628291 0.50062 decode     rhuang       qw    04/25/2023 10:45:33                                    1  0.45-0.25  2.14/2.25
# 3628292 0.50062 decode     rhuang       qw    04/25/2023 10:45:40                                    1  0.45-0.3   2.14/2.24
# 3628293 0.50062 decode     rhuang       qw    04/25/2023 10:45:47                                    1  0.5-0.1    2.4/-
# 3628294 0.50062 decode     rhuang       qw    04/25/2023 10:45:50                                    1  0.5-0.15   2.26/-
# 3628295 0.50062 decode     rhuang       qw    04/25/2023 10:45:53                                    1  0.5-0.2    2.2/-
# 3628296 0.50062 decode     rhuang       qw    04/25/2023 10:45:56                                    1  -
# 3628297 0.50062 decode     rhuang       qw    04/25/2023 10:45:59                                    1  0.5-0.3    2.17/-
# -----
# Conclusion: the difference between $rnn_lm_scale and $ngram_lm_scale should be 0.05 or 0.1

# Run RNNLM rescoring on --epoch 25 --avg 5 with 1024-3 model:
# 3627971 0.50065 decode     rhuang       r     04/25/2023 02:18:23 g.q@octopod.clsp.jhu.edu           1  25-5       2.13/2.1 (baseline)
# 3628604 0.50062 decode     rhuang       r     04/25/2023 19:47:55 g.q@octopod.clsp.jhu.edu           1  0.05-0     2.1/2.1
# 3628605 0.50062 decode     rhuang       r     04/25/2023 19:48:11 g.q@octopod.clsp.jhu.edu           1  0.1-0      2.09/2.11
# 3628606 0.50062 decode     rhuang       r     04/25/2023 19:48:15 g.q@octopod.clsp.jhu.edu           1  0.1-0.05   2.09/2.1
# 3628607 0.50062 decode     rhuang       r     04/25/2023 19:48:27 g.q@octopod.clsp.jhu.edu           1  0.15-0.05  2.07/2.11
# 3628638 0.50062 decode     rhuang       qw    04/25/2023 19:48:29                                    1  0.15-0.1   2.08/2.1  <--best
# 3628639 0.50062 decode     rhuang       qw    04/25/2023 19:48:41                                    1  0.2-0.1    2.06/2.12 <--best on dev
# 3628640 0.50061 decode     rhuang       qw    04/25/2023 19:48:45                                    1  0.2-0.15   2.08/2.11
# 3628641 0.50061 decode     rhuang       qw    04/25/2023 19:48:54                                    1  0.25-0.15  2.08/2.13
# 3628612 0.50061 decode     rhuang       qw    04/25/2023 19:48:58                                    1  0.25-0.2   2.09/2.13
# 3628613 0.50061 decode     rhuang       qw    04/25/2023 19:49:01                                    1  0.3-0.2    2.09/2.15
# 3628614 0.50061 decode     rhuang       qw    04/25/2023 19:49:06                                    1  0.3-0.25   2.1/2.16
# 3628615 0.50061 decode     rhuang       qw    04/25/2023 19:49:11                                    1  0.35-0.25  2.08/2.17
# 3628616 0.50061 decode     rhuang       qw    04/25/2023 19:49:15                                    1  0.35-0.3   2.14/2.19

# Run RNNLM rescoring on --epoch 25 --avg 5 with 1024-2 model:
# 3628649 0.50061 decode     rhuang       qw    04/25/2023 20:26:55                                    1  0.05-0     2.12/2.1
# 3628650 0.50061 decode     rhuang       qw    04/25/2023 20:27:02                                    1  0.1-0      2.09/2.11
# 3628651 0.50061 decode     rhuang       qw    04/25/2023 20:27:06                                    1  0.1-0.05   2.1/2.09   <--best on val
# 3628652 0.50061 decode     rhuang       qw    04/25/2023 20:27:11                                    1  0.15-0.05  2.09/2.11
# 3628653 0.50061 decode     rhuang       qw    04/25/2023 20:27:15                                    1  0.15-0.1   2.08/2.1
# 3628654 0.50061 decode     rhuang       qw    04/25/2023 20:27:19                                    1  0.2-0.1    2.08/2.12
# 3628655 0.50061 decode     rhuang       qw    04/25/2023 20:27:23                                    1  0.2-0.15   2.08/2.12
# 3628656 0.50061 decode     rhuang       qw    04/25/2023 20:27:26                                    1  0.25-0.15  2.08/2.13
# 3628657 0.50061 decode     rhuang       qw    04/25/2023 20:27:30                                    1  0.25-0.2   2.07/2.14
# 3628658 0.50061 decode     rhuang       qw    04/25/2023 20:27:33                                    1  0.3-0.2    2.07/2.16
# 3628659 0.50061 decode     rhuang       qw    04/25/2023 20:27:36                                    1  0.3-0.25   2.08/2.16
# 3628660 0.50061 decode     rhuang       qw    04/25/2023 20:27:40                                    1  0.35-0.25  2.08/2.18
# 3628661 0.50061 decode     rhuang       qw    04/25/2023 20:27:44                                    1  0.35-0.3   2.11/2.2

# Run RNNLM rescoring on --epoch 25 --avg 5 with previous 800-200 model:
# 3628662 0.50061 decode     rhuang       qw    04/25/2023 20:29:05                                    1  0.05-0     2.14/2.1
# 3628663 0.50061 decode     rhuang       qw    04/25/2023 20:29:11                                    1  0.1-0      2.14/2.12
# 3628664 0.50061 decode     rhuang       qw    04/25/2023 20:29:16                                    1  0.1-0.05   2.14/2.11
# 3628665 0.50061 decode     rhuang       qw    04/25/2023 20:29:19                                    1  0.15-0.05  2.17/2.13
# 3628666 0.50061 decode     rhuang       qw    04/25/2023 20:29:22                                    1  0.15-0.1   2.17/2.12
# 3628667 0.50061 decode     rhuang       qw    04/25/2023 20:29:25                                    1  0.2-0.1    2.17/2.15
# 3628668 0.50061 decode     rhuang       qw    04/25/2023 20:29:28                                    1  0.2-0.15   2.17/2.14
# 3628669 0.50061 decode     rhuang       qw    04/25/2023 20:29:32                                    1  0.25-0.15  2.2/2.16
# 3628670 0.50061 decode     rhuang       qw    04/25/2023 20:29:34                                    1  0.25-0.2   2.2/2.16
# 3628671 0.50061 decode     rhuang       qw    04/25/2023 20:29:38                                    1  0.3-0.2    2.22/2.19
# 3628672 0.50061 decode     rhuang       qw    04/25/2023 20:29:41                                    1  0.3-0.25   2.24/2.19
# 3628673 0.50061 decode     rhuang       qw    04/25/2023 20:29:44                                    1  0.35-0.25  2.24/2.21
# 3628674 0.50061 decode     rhuang       qw    04/25/2023 20:29:47                                    1  0.35-0.3   2.25/2.22

