# https://docs.google.com/document/d/1UbIesSYxOpy5Gy7LRcayaij9slSUN0XwXM-LlQ4GVN8/edit

cp -r /data/skhudan1/extras/* .

cd /home/rhuang25/work/icefall/egs/librispeech/ASR/zipformer
mkdir download
ln -s /data/skhudan1/corpora/librispeech download/LibriSpeech
ln -s /data/skhudan1/corpora/musan download/musan

mkdir data
# on coe grid:
scp -r /exp/rhuang/librispeech/data2/{fbai-speech,fbank,lang_bpe_500,lm,manifests} \
  rhuang25@rfdtn1.rockfish.jhu.edu:/home/rhuang25/work/icefall/egs/librispeech/ASR/data/.

rsync -a --ignore-existing --progress /exp/rhuang/librispeech/data2/{fbai-speech,fbank,lang_bpe_500,lm,manifests} \
  rhuang25@rfdtn1.rockfish.jhu.edu:/home/rhuang25/work/icefall/egs/librispeech/ASR/data/.

cd scripts
sbatch sbatch.sh

interact  -usage
interact -p defq -c 6 -t 12:00:00
interact -p a100 -c 6 -t 12:00:00

salloc -J interact -N 1 -n 12 --time=120 --mem=48g -p defq srun --pty bash

# sbatch example scripts:
# https://www.arch.jhu.edu/guide/#pp-toc__heading-anchor-15:~:text=my%2Dinput%2Dfile-,Example%20Scripts,-Basic%20openMP%20Job

# This works!
# a100/ica100
salloc --job-name=test \
  --partition=a100 \
  --gpus=4 \
  --gres=gpu:4 \
  --gpus-per-node=4 \
  --nodes=1 \
  --gpus-per-task=1 \
  --ntasks-per-node=4 \
  --cpus-per-task=4 \
  --time=72:00:00 \
  --account=skhudan1_gpu \
  --qos=qos_gpu 

# -w gpu08
# --cpus-per-task=4

# This works!
# a100/ica100
srun -p a100 -t 72:00:00 --gpus=4 --gpus-per-node=4 --ntasks-per-node=4 --cpus-per-task=4 --account=skhudan1_gpu --qos=qos_gpu --pty bash

# CPU machine, this works!
salloc --job-name=test_cpu \
  --partition=parallel \
  --nodes=1 \
  --ntasks-per-node=8 \
  --cpus-per-task=4 \
  --time=72:00:00 \
  --account=skhudan1_gpu
# or: just run `salloc`

squeue | grep a100
squeue | grep ica100


# Apply for only one GPU
salloc --job-name=test \
  --partition=a100 \
  --gpus=1 \
  --gres=gpu:1 \
  --gpus-per-node=1 \
  --nodes=1 \
  --gpus-per-task=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=4 \
  --time=72:00:00 \
  --account=skhudan1_gpu \
  --qos=qos_gpu 

