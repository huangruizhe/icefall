# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# However, we are losing a lot of features by using a simple for loop to iterate over the data. In particular, we are missing out on:
# - Batching the data
# - Shuffling the data
# - Load the data in parallel using multiprocessing workers.
# torch.utils.data.DataLoader is an iterator which provides all these features. Parameters used below should be clear. One parameter of interest is collate_fn. You can specify how exactly the samples need to be batched using collate_fn. However, default collate should work fine for most use cases.


# https://stackoverflow.com/questions/38378310/reading-data-in-parallel-with-multiprocess



# import multiprocessing
# import pandas as pd

# def load_data(filename):
#     df = pd.read_csv(filename)
#     return df

# if __name__ == '__main__':
#     filenames = ['data1.csv', 'data2.csv', 'data3.csv']

#     # Create a multiprocessing pool with 4 workers
#     pool = multiprocessing.Pool(4)

#     # Load the data in parallel
#     results = pool.map(load_data, filenames)

#     # Close the pool
#     pool.close()

#     # Merge the results into a single dataframe
#     df = pd.concat(results)

#     # Print the dataframe
#     print(df)


import torch
import random
import numpy as np

import multiprocessing as mp
import queue  # used for the data queue


def fix_random_seed(random_seed: int):
    """
    Set the same random seed for the libraries and modules that Lhotse interacts with.
    Includes the ``random`` module, numpy, torch, and ``uuid4()`` function defined in this file.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    # Ensure deterministic ID creation
    rd = random.Random()
    rd.seed(random_seed)




class LongAudioDataLoader:
    # Dataloader:
    #   - load n long audios + alignments (each audio 200MB)
    #   - downsample each long audio to 16kHz
    #   - get n segments of roughly t seconds according to the alignments

    def __init__(self, 
        dataset, 
        batch_size=1, 
        shuffle=False,
        random_seed=None, 
        num_workers=1, 
        prefech_factor=2,
        num_long_audios=3,
        target_sample_rate=16000,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle        
        self.num_workers = num_workers
        self.prefech_factor = prefech_factor
        self.num_long_audios = num_long_audios
        self.target_sample_rate = target_sample_rate

        fix_random_seed(random_seed)

        self.data_queue = mp.Queue(num_workers)
        self.index_queue = mp.Queue()
        self.workers = []

    def __iter__(self):
        self._reset()
        return self

    def _reset(self):
        # Create index queue
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        for i in range(0, len(indices), self.batch_size):
            self.index_queue.put(indices[i:i+self.batch_size])

        # Start worker processes
        self.workers = [mp.Process(target=self._worker_loop) for _ in range(self.num_workers)]
        for w in self.workers:
            w.start()

    def _worker_loop(self):
        while True:
            try:
                indices = self.index_queue.get(timeout=1)  # get a batch of indices
            except queue.Empty:
                break  # if the queue is empty, exit the loop
            batch = [self.dataset[i] for i in indices]  # load the data
            self.data_queue.put(batch)  # put the data into the data queue

    def __next__(self):
        if not self.workers:
            raise StopIteration
        while True:
            if not self.data_queue.empty():
                return self.data_queue.get()
            else:
                if not any(w.is_alive() for w in self.workers):
                    raise StopIteration

    def __del__(self):
        for w in self.workers:
            w.terminate()