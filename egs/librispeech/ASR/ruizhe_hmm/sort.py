from lhotse import CutSet
from dataclasses import replace
import logging

logging.basicConfig(
    format = "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
    level = 10
)

in_file_name = "/fsx/users/huangruizhe/icefall_align2/egs/librispeech/ASR/data/fbank/librispeech_cuts_train-all-shuf.jsonl.gz"
out_file_name = "/fsx/users/huangruizhe/icefall_align2/egs/librispeech/ASR/data/fbank/librispeech_cuts_train-all-sorted.jsonl.gz"

logging.info(f"Loading from: {in_file_name}")
cuts = CutSet.from_file(in_file_name)

logging.info("Sorting ...")
sorted_cuts = cuts.sort_by_duration(ascending=True)

logging.info(f"Saving to: {out_file_name}")
sorted_cuts.to_file(out_file_name)
logging.info(f'Done: {out_file_name}')
