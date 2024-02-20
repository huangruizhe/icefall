## Generate a clearned version of dataset

from data_long_dataset import *
import torch.multiprocessing as mp
from tqdm import tqdm


def clean_one_data(dataset, i_start, chunk_size, rank):
    for i in tqdm(range(i_start, min(i_start+chunk_size, len(dataset))), desc=f"rank={rank}"):
        audio_path, text_path = dataset.manifest[i]

        new_text_path = text_path.replace("LibriSpeechOriginal/LibriSpeech/", "LibriSpeechAligned/LibriSpeech/")
        new_text_path = f"{dataset.root}/{new_text_path}"
        new_text_path = new_text_path.replace(".utf-8", "")
        if Path(new_text_path).exists():
            continue

        ds = dataset[i]
        waveform, sample_rate, text, speaker_id, audio_id, meta_data = ds
        if text is None:
            print(f"Problematic book: {i}/{i_start+chunk_size}", meta_data['audio_path'], meta_data['text_path'])
            continue

        Path(new_text_path).parent.mkdir(parents=True, exist_ok=True)
        with open(new_text_path, 'w') as fout:
            print(text, file=fout)


long_dataset = LibrispeechLongAudioDataset(
    root = "/exp/rhuang/meta/icefall/egs/librispeech/ASR/download/",
    skip_loading_audio = True,
    skip_text_normalization = False,
    manifest_file = "/exp/rhuang/librispeech/download2/LibriSpeechOriginal/chapter_manifest.txt",
)
print(f"len(long_dataset) = {len(long_dataset)}")

############################
# single thread
############################

# clean_one_data(long_dataset, 0, len(long_dataset), 0)
# exit(0)

############################
# parallel processing
############################
processes = []
manager = mp.Manager()
# Fork processes
n_process = 32
chunk_size = int(len(long_dataset) / n_process) + 1
i_chunk = 0
for i in range(0, len(long_dataset), chunk_size):
    fork = mp.Process(
        target=clean_one_data,
        args=(long_dataset, i, chunk_size, i_chunk)
    )
    fork.start()
    processes.append(fork)
    i_chunk += 1
# Wait until all processes are finished
for fork in processes:
    fork.join()