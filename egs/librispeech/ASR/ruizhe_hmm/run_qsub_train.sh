#!/usr/bin/env bash
#$ -wd /export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR
#$ -V
#$ -N train
#$ -j y -o ruizhe_hmm/log/$JOB_NAME-$JOB_ID.out
#$ -M ruizhe@jhu.edu
#$ -m e
#$ -l ram_free=8G,mem_free=8G,gpu=4,hostname=!b*&!c18*
#$ -q g.q

# ,hostname=!b*&!c18*
# &!octopod*
# -q 4gpu.q
# -q g.q
ngpus=4

# Check how many GPUs are available:
# qstat -F gpu -q g.q | grep -A2 '.*@\(c\).* *lx-amd64 *[aA ]$'

#### Activate dev environments and call programs
mamba activate /home/rhuang/mambaforge/envs/efrat2
export PYTHONPATH=/export/fs04/a12/rhuang/k2/k2/python:$PYTHONPATH # for `import k2`
export PYTHONPATH=/export/fs04/a12/rhuang/k2/build/temp.linux-x86_64-cpython-38/lib/:$PYTHONPATH # for `import _k2`
# export PYTHONPATH=/export/fs04/a12/rhuang/icefall_align/:$PYTHONPATH
# export PYTHONPATH=/export/fs04/a12/rhuang/icefall/:$PYTHONPATH
export PYTHONPATH=/export/fs04/a12/rhuang/icefall_align2/:$PYTHONPATH

hostname
nvidia-smi
echo "python: `which python`"

#### Assign a free-GPU to your program (make sure -n matches the requested number of GPUs above)
# source /home/gqin2/scripts/acquire-gpu 4

ngpus=$ngpus # num GPUs for multiple GPUs training within a single node; should match those in $free_gpu
free_gpu= # comma-separated available GPU ids, eg., "0" or "0,1"; automatically assigned if on CLSP grid
[ -z "$free_gpu" ] && [[ $(hostname -f) == *.clsp.jhu.edu ]] && free_gpu=$(free-gpu -n $ngpus) || \
echo "Unable to get $ngpus GPUs"
[ -z "$free_gpu" ] && echo "$0: please specify --free-gpu" && exit 1;
[ $(echo $free_gpu | sed 's/,/ /g' | awk '{print NF}') -ne "$ngpus" ] && \
 echo "number of GPU ids in --free-gpu=$free_gpu does not match --ngpus=$ngpus" && exit 1;
export CUDA_VISIBLE_DEVICES="$free_gpu"
echo $CUDA_VISIBLE_DEVICES

#### Test running qsub
python3 -c "import torch; print(torch.__version__)"

#### Your script
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# python3 tdnn_lstm_ctc/train.py --world-size 4
# python3 tdnn_lstm_ctc/train.py --world-size 4 --master-port 12355
# python3 tdnn_lstm_ctc2/train.py --world-size 4

# python3 tdnn_lstm_ctc/train_bpe.py --world-size 4 

# full librispeech
# python tdnn_lstm_ctc/train.py \
#     --world-size 4 \
#     --full-libri true \
#     --max-duration 200 \
#     --num-epochs 30 \
#     --exp-dir "tdnn_lstm_ctc/exp/exp_libri_full"

# # libri100
# python tdnn_lstm_ctc/train.py \
#     --world-size 4 \
#     --full-libri false \
#     --max-duration 200 \
#     --num-epochs 30 \
#     --valid-interval 100 \
#     --exp-dir "tdnn_lstm_ctc/exp/exp_libri_100"

# # # zipformer_ctc on libri100
# ./zipformer_ctc/train.py \
#   --world-size 4 \
#   --master-port 12345 \
#   --num-epochs 30 \
#   --start-epoch 1 \
#   --lang-dir data/lang_bpe_500 \
#   --exp-dir zipformer_ctc/exp/exp_libri_100  \
#   --max-duration 200 \
#   --full-libri false \
#   --use-fp16 true

# # zipformer_hmm_ml on libri100
# ./zipformer_mmi_hmm/train.py \
#   --world-size $ngpus \
#   --master-port 12345 \
#   --num-epochs 30 \
#   --start-epoch 1 \
#   --lang-dir data/lang_bpe_500 \
#   --exp-dir zipformer_mmi_hmm/exp/exp_libri_100_ml  \
#   --max-duration 200 \
#   --full-libri false \
#   --use-fp16 true \
#   --warm-step 90000000000 \
#   --num-workers 6 \
#   --ctc-beam-size 15 \
#   --sil-modeling false # --start-epoch 13


./zipformer_mmi/train.py \
  --world-size $ngpus \
  --master-port 12346 \
  --num-epochs 30 \
  --start-epoch 1 \
  --lang-dir data/lang_bpe_500 \
  --exp-dir zipformer_mmi/exp/exp_libri_100_ctc  \
  --max-duration 200 \
  --full-libri false \
  --use-fp16 true \
  --save-every-n 20000 --exp-dir zipformer_mmi/exp/exp_libri_100_ctc_mmi --start-epoch 4 --topo-type "ctc" # --base-lr 0.03 # --shuffle false --bucketing-sampler false --start-epoch 11
# --shuffle false --bucketing-sampler false
# --start-epoch 3

# ASR results: 
# ctc_ml    7.53/19.39  # /exp/rhuang/icefall_latest/egs/librispeech/ASR/ruizhe/log/log-decode-10571540.out
# ctc_mmi      6/16.65  # 
# hmm_ml   13.91/31.68  # ep30,avg1  # /export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/zipformer_mmi/exp/exp_libri_100_ml/nbest-rescoring-LG/log-decode-epoch-30-avg-1-2023-05-06-18-33-36 # /export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/zipformer_mmi/exp/exp_libri_100_ml/nbest-rescoring-LG/log-decode-epoch-30-avg-1-2023-05-06-18-53-59
# hmm_ml   12.16/28.09  # ep30,avg10 
# hmm_ml   12.72/28.98  # ep30,avg10,decode with ctc topo
# hmm_mmi       /       # ep30,avg10

# Xiaohui's paper:
# ctc_ml     4.6/11.5
# hmm_ml     7.2/17.3

# mkdir -p zipformer_mmi/exp/exp_libri_100_ctc_mmi
# ln -s /export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/zipformer_mmi/exp/exp_libri_100_ctc/epoch-3.pt zipformer_mmi/exp/exp_libri_100_ctc_mmi/.
# --exp-dir zipformer_mmi/exp/exp_libri_100_ctc_mmi --start-epoch 4 --topo-type "ctc"

# mkdir -p zipformer_mmi/exp/exp_libri_100_hmm_mmi
# ln -s /export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/zipformer_mmi/exp/exp_libri_100_ml/epoch-10.pt zipformer_mmi/exp/exp_libri_100_hmm_mmi/.
# --exp-dir zipformer_mmi/exp/exp_libri_100_hmm_mmi --start-epoch 11 --topo-type "hmm"

# hmm_ml  # 3638099 0.50262 train      rhuang       r     05/06/2023 09:29:21 g.q@octopod.clsp.jhu.edu           1        
# ctc_ml  # 3638112 0.50252 train      rhuang       r     05/06/2023 11:03:45 g.q@octopod.clsp.jhu.edu           1        
# hmm_mmi # 3638140 0.50158 train      rhuang       r     05/06/2023 13:47:51 g.q@c10.clsp.jhu.edu               1        
# ctc_mmi # 3638253 0.50237 train      rhuang       r     05/06/2023 22:24:37 g.q@octopod.clsp.jhu.edu           1        
# /export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/ruizhe_hmm/log/train-3638140.out

# hmm vs. hmm_mmi:
# https://tensorboard.dev/experiment/QOXqNxmbS1i8WLBdwYtAGw/#scalars&runSelectionState=eyIwLjAyX3NvcnRlZF9jdXRzIjpmYWxzZSwiMC4wMl9zb3J0ZWRfY3V0c19teXRvcG8iOmZhbHNlLCIwLjAzX215dG9wbyI6dHJ1ZSwiMC4wNSI6ZmFsc2UsImN0YyI6ZmFsc2UsIjAuMDEiOmZhbHNlLCIwLjAwMSI6ZmFsc2UsIjAuMDMiOmZhbHNlLCJjdGMyIjp0cnVlfQ%3D%3D
# https://tensorboard.dev/experiment/NDJjysghQl69AQRup2l3Cg/

# ctc vs. ctc_mmi:
# https://tensorboard.dev/experiment/aRhb7xjtRLqaLD5AOzM7XA/#scalars

