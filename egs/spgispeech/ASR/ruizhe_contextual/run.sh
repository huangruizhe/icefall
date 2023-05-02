cd /exp/rhuang/icefall_latest/egs/spgispeech/ASR/
# export PATH=/export/fs04/a12/rhuang/git-lfs-3.2.0/:$PATH
git lfs install
git lfs version
mkdir -p pretrained
cd pretrained; git clone https://huggingface.co/desh2608/icefall-asr-spgispeech-pruned-transducer-stateless2; cd ..

path_to_pretrained_asr_model="/exp/rhuang/icefall_latest/egs/spgispeech/ASR/pretrained/icefall-asr-spgispeech-pruned-transducer-stateless2"
# ln -s $path_to_pretrained_asr_model/exp/pretrained.pt $path_to_pretrained_asr_model/exp/epoch-1.pt
lang=$path_to_pretrained_asr_model/data/lang_bpe_500/

scp -r rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/pruned_transducer_stateless7_context/*.* pruned_transducer_stateless7_context/.
scp -r rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/ruizhe_contextual/*.* ruizhe_contextual/.
scp -r rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/librispeech/ASR/pruned_transducer_stateless2/beam_search.py pruned_transducer_stateless2/.
scp -r rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/pruned_transducer_stateless2/*.* pruned_transducer_stateless2/.
scp -r rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/pruned_transducer_stateless2_context/*.* pruned_transducer_stateless2_context/.
scp -r rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/*.* ruizhe_contextual/.

scp -r rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/pruned_transducer_stateless7_context/*.* pruned_transducer_stateless7_context/.
scp -r rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/*.* ruizhe_contextual/.

scp -r rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/icefall/*.* /exp/rhuang/icefall_latest/icefall/.

#### re-use fbank for un-normalized text
# cd /exp/rhuang/icefall_latest/egs/spgispeech/ASR/
# mamba activate /exp/rhuang/mambaforge/envs/icefall2
python ruizhe_contextual/get_train_cuts.py  # This seems not working!
python ruizhe_contextual/get_train_cuts2.py

for f in /exp/rhuang/icefall/egs/spgispeech/ASR/data/fbank_no_norm/feats_train_*; do
    ln -s $(realpath $f) data/fbank/.
done

#### prepare data from scratch -- it has to be done on GPU!
# egs/spgispeech/ASR/prepare.sh
python local/compute_fbank_spgispeech.py --train --num-splits 20 --start 2
# --start
# --stop

# egs/spgispeech/ASR/local/compute_fbank_spgispeech.py
# Set: output_dir = Path("data/fbank_temp")

scp /exp/rhuang/icefall_latest/egs/spgispeech/ASR/data/manifests/cuts_{dev,val}.jsonl.gz \
  rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/.

# https://lhotse.readthedocs.io/en/latest/_modules/lhotse/audio.html
# class AudioSource:
# https://stackoverflow.com/questions/49908399/replace-attributes-in-data-class-objects
cd /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR
part="dev"
part="val"
python -c """
from lhotse import CutSet
from dataclasses import replace

file_name = '/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_${part}.jsonl.gz'
cuts = CutSet.from_file(file_name)
print(f'len(cuts) = {len(cuts)}')
s1 = '/exp/rhuang/icefall_latest/egs/spgispeech/ASR/download/spgispeech/'
s2 = '/export/c01/corpora6/spgispeech/spgispeech_recovered_uncomplete/'
for c in cuts:
    # for r in c.recording.sources:
    #     r.source = r.source.replace(s1, s2)
    assert len(c.recording.sources) == 1
    ss = c.recording.sources[0].source.replace(s1, s2)
    c.recording.sources[0] = replace(c.recording.sources[0], source=ss)

file_name = '/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_${part}_.jsonl.gz'
cuts.to_file(file_name)
print(f'Done: {file_name}')
"""
# It seems the above doesn't work.
# Use the following instead
s1='/exp/rhuang/icefall_latest/egs/spgispeech/ASR/download/spgispeech/'
s2='/export/c01/corpora6/spgispeech/spgispeech_recovered_uncomplete/'
zcat /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_${part}.jsonl.gz |\
  sed "s%$s1%$s2%g" | gzip \
> /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_${part}_.jsonl.gz
mv /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_${part}_.jsonl.gz \
  /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_${part}.jsonl.gz

# kaldifeat cannot be installed correctly...
if False:
    # https://github.com/search?q=repo%3Acsukuangfj%2Fkaldifeat%20CMAKE_ARGS&type=code
    # https://github.com/csukuangfj/kaldifeat/blob/master/doc/source/installation.rst

    CUDNN_LIBRARY_PATH=/home/smielke/cuda-cudnn/lib64
    CUDNN_INCLUDE_PATH=/home/smielke/cuda-cudnn/include
    CUDA_TOOLKIT_DIR=/usr/local/cuda

    export KALDIFEAT_MAKE_ARGS="-j4"
    export KALDIFEAT_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DCUDNN_LIBRARY_PATH=$CUDNN_LIBRARY_PATH/libcudnn.so -DCUDNN_INCLUDE_PATH=$CUDNN_INCLUDE_PATH"
    # pip install --verbose kaldifeat
    cd /export/fs04/a12/rhuang/kaldifeat/
    python setup.py install

    # https://csukuangfj.github.io/kaldifeat/installation.html#install-kaldifeat-from-conda-only-for-linux
    mamba install -c kaldifeat -c pytorch -c conda-forge kaldifeat python=3.8 cudatoolkit=10.2 pytorch=1.12.1
    mamba install -c kaldifeat -c pytorch cpuonly kaldifeat python=3.8 pytorch=1.12.1

    python3 -c "import kaldifeat; print(kaldifeat.__version__)"

    # still have
    conda activate /export/fs04/a12/rhuang/anaconda/anaconda3/envs/espnet_gpu
    mamba activate /home/rhuang/mambaforge/envs/efrat2

    export PYTHONPATH=/export/fs04/a12/rhuang/kaldifeat/build/lib:/export/fs04/a12/rhuang/kaldifeat/kaldifeat/python:$PYTHONPATH

mkdir -p data/fbank
python local/compute_fbank_spgispeech.py --test

# Don't use kaldifeat!!!
# Checkout other recipes of how they compute fbank

python -c """
from pathlib import Path
output_dir = Path("data/fbank")

sampling_rate = 16000
num_mel_bins = 80
"""


/export/fs04/a12/rhuang/contextualizedASR/data/ec53_kaldi_sp_gentle/20220129/cuts3.jsonl.gz
python ruizhe_contextual/get_train_cuts2.py

# move zipformer model from coe grid to clsp grid
scp -r /exp/rhuang/icefall_latest/egs/spgispeech/ASR/pruned_transducer_stateless7/exp_500_norm \
  rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/pruned_transducer_stateless7/.

scp -r /exp/rhuang/icefall_latest/egs/spgispeech/ASR/data/lang_bpe_500/ \
  rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/.

# scp while preserving soft linkes
rsync -Wav --progress /exp/rhuang/icefall_latest/egs/spgispeech/ASR/pruned_transducer_stateless7/*.* \
  rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/pruned_transducer_stateless7/.

# setup egs/spgispeech/ASR/pruned_transducer_stateless7_context
cd /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/pruned_transducer_stateless7_context/
for f in ../pruned_transducer_stateless7/*.*; do
    ln -s $f .
done
rm asr_datamodule.py train.py decode.py model.py
for f in ../pruned_transducer_stateless2_context/{asr_datamodule.py,context_*.*,word_encoder_*.*,biasing_module.py,score.py}; do
    ln -s $f .
done
for f in ../pruned_transducer_stateless7/{train.py,decode.py,model.py}; do
    cp $(realpath $f) .
done
for f in ../../../librispeech/ASR/pruned_transducer_stateless7_context/{model.py,}; do
    ln -s $f .
done
# TODO: modify these files train.py, decode.py, decode_ec53.py


# export/average the zipformer model
# https://icefall.readthedocs.io/en/latest/recipes/Non-streaming-ASR/librispeech/pruned_transducer_stateless.html#export-model
qrsh -q "gpu.q@@rtx" -l 'gpu=1,mem_free=32G,h_rt=600:00:00,hostname=!r3n01*'  # Actually, no need to login to gpu nodes
cd /exp/rhuang/icefall_latest/egs/spgispeech/ASR
# Go to the following file to set up envs:
# /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/run_decode_zipformer.sh
vocab_size=500
python pruned_transducer_stateless7/export.py \
  --exp-dir pruned_transducer_stateless7/exp_${vocab_size}_norm \
  --bpe-model data/lang_bpe_${vocab_size}/bpe.model \
  --epoch 25 \
  --avg 5

# mv manifests and fbanks for spgi dev/val to clsp grid
cd /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR
part="dev"
part="val"
s1='/exp/rhuang/icefall_latest/egs/spgispeech/ASR/download/spgispeech/'
s2='/export/c01/corpora6/spgispeech/spgispeech_recovered_uncomplete/'
zcat /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_${part}.jsonl.gz |\
  sed "s%$s1%$s2%g" | gzip \
> /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_${part}_.jsonl.gz
mv /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_${part}_.jsonl.gz \
  /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_${part}.jsonl.gz

scp -r /exp/rhuang/icefall_latest/egs/spgispeech/ASR/data/fbank/feats_{dev,val}.lca \
  rhuang@login.clsp.jhu.edu:/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/fbank/.

##### Rare word WER instead of entity WER
# set up envs in: /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/upsampling.sh
part="val"  # dev  # val
python /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/context_collector_test.py \
  --cuts-file-name "/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_${part}.jsonl.gz" \
> data/rare_words/ref/biasing_list_${part}.txt

part="ec53"
ec53_cuts="/export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/data/manifests/cuts_ec53_norm.jsonl.gz"
python /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/ruizhe_contextual/context_collector_test.py \
  --cuts-file-name $ec53_cuts \
> data/rare_words/ref/biasing_list_${part}.txt

cd /export/fs04/a12/rhuang/icefall_align2/egs/spgispeech/ASR/

# hyp_in=pruned_transducer_stateless7_context/exp/exp_libri_full_c-1_stage1/modified_beam_search/recogs-ec53-epoch-21-avg-1-modified_beam_search-beam-size-4-encoder-biasing-decoder-biasing-20230430-051101.txt
# hyp=pruned_transducer_stateless7_context/exp/exp_libri_full_c-1_stage1/modified_beam_search/recogs-ec53-epoch-21-avg-1-modified_beam_search-beam-size-4-encoder-biasing-decoder-biasing-20230430-051101.hyp.txt
hyp_in=pruned_transducer_stateless7_context/exp/exp_libri_full_c-1_stage1/modified_beam_search/recogs-ec53-epoch-1-avg-9-modified_beam_search-beam-size-4-20230426-121352.txt
hyp=${hyp_in%".txt"}.hyp.txt
part="ec53"
ref=data/rare_words/ref/biasing_list_${part}.txt
python ruizhe_contextual/recogs_to_text.py \
  --cuts $ec53_cuts \
  --input $hyp_in \
  --out $hyp

wc $ref $hyp
python pruned_transducer_stateless7_context/score.py \
  --refs $ref \
  --hyps $hyp

# no biasing: 10.59(9.22/23.24)
# slides:     11.09(9.57/25.16)