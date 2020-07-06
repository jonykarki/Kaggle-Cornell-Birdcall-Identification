import matplotlib.pyplot as plt 
import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
from librosa.core import resample, to_mono
import tqdm

import concurrent.futures

DATA_DIR = "/content/data"

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))[['ebird_code', 'filename', 'duration']]
train_df.head()

train_df['filepath'] = ""
train_df['filepath'] = train_df.apply(lambda row: os.path.join(DATA_DIR, "train_audio", row["ebird_code"], row["filename"]), axis=1)
train_df.shape

train_sample = train_df[train_df['ebird_code'] == 'aldfly']
train_sample.shape

SR = 32000
D_TIME = 5 # take every 5 seconds
D_SAMPLE = D_TIME * SR
SAVE_FOLDER = "../out"

for bird in train_df['ebird_code'].unique():
    os.makedirs(os.path.join(SAVE_FOLDER, bird), exist_ok=True)

def save_as_npys(row):
    # read the file
    y, sr = librosa.load(row['filepath'], sr=SR)

    # check the delta time
    if y.shape[0] < D_SAMPLE:
        # 0 pad the rest
        sample = np.zeros(shape=(D_SAMPLE,), dtype=np.int16)
        sample[:y.shape[0]] = y

        sample_mel_spec = librosa.feature.melspectrogram(y=y, sr=SR)
        sample_mel_spec = librosa.power_to_db(sample_mel_spec, ref=np.max)

        file_name = f"{SAVE_FOLDER}/{row['ebird_code']}/{row['filename'][:-4]}.npy"
        np.save(file_name, sample_mel_spec)
    else:
        # remove the last part that is not D_SAMPLEs
        trunc = y.shape[0] % D_SAMPLE
        for i, idx in enumerate(np.arange(0, y.shape[0]-trunc, D_SAMPLE)):
            start = int(idx)
            stop = int(idx + D_SAMPLE)
            # get the d_samples
            sample = y[start:stop]

            # get the melspec for that sample
            sample_mel_spec = librosa.feature.melspectrogram(y=sample, sr=SR)
            sample_mel_spec = librosa.power_to_db(sample_mel_spec, ref=np.max)

            file_name = f"{SAVE_FOLDER}/{row['ebird_code']}/{row['filename'][:-4]}_{i}.npy"

            np.save(file_name, sample_mel_spec)


rows_series_as_list = []
for i, j in train_df.iterrows():
    rows_series_as_list.append(j)

from multiprocessing.pool import Pool


with Pool() as p:
    r = list(tqdm.tqdm(p.imap(save_as_npys, rows_series_as_list), total=len(rows_series_as_list)))