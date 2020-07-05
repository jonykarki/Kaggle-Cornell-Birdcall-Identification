# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython


# %%
import sys, os
if os.path.abspath(os.pardir) not in sys.path:
    sys.path.insert(1, os.path.abspath(os.pardir))
import CONFIG
get_ipython().magic(u'reload_ext autoreload')
get_ipython().magic(u'autoreload 2')


# %%
import matplotlib.pyplot as plt 
import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
from librosa.core import resample, to_mono
from tqdm.auto import tqdm
import IPython.display as ipd


# %%
DATA_DIR = CONFIG.CFG.DATA.BASE


# %%
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))[['ebird_code', 'filename', 'duration']]
train_df.head()


# %%
train_df['filepath'] = ""
train_df['filepath'] = train_df.apply(lambda row: os.path.join(DATA_DIR, "train_audio", row["ebird_code"], row["filename"]), axis=1)
train_df.shape


# %%
train_sample = train_df[train_df['ebird_code'] == 'linspa'].sample()
sample_path = train_sample['filepath'].values[0]
train_sample


# %%
transformed_dict = {
    "ebird_code": [],
    "filename": []
}


# %%
SR = 32000
D_TIME = 5 # take every 5 seconds
D_SAMPLE = D_TIME * SR
SAVE_FOLDER = "../out"
y_act, sr = librosa.load(sample_path, sr=SR)
y = y_act[:d_time*SR]
y.shape, y_act.shape, sr


# %%
for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
    # read the file
    y, sr = librosa.load(row['filepath'], sr=SR)

    os.makedirs(os.path.join(SAVE_FOLDER, row['ebird_code']), exist_ok=True)

    # check the delta time
    if y.shape[0] < D_SAMPLE:
        # 0 pad the rest
        sample = np.zeros(shape=(D_SAMPLE,), dtype=np.int16)
        sample[:y.shape[0]] = y

        sample_mel_spec = librosa.feature.melspectrogram(y=y, sr=SR)
        sample_mel_spec = librosa.power_to_db(sample_mel_spec, ref=np.max)

        file_name = f"{SAVE_FOLDER}/{row['ebird_code']}/{row['filename'][:-4]}.npy"
        np.save(file_name, sample_mel_spec)
        transformed_dict['ebird_code'].append(row['ebird_code'])
        transformed_dict['filename'].append(file_name.split("/")[-1])
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

            transformed_dict['ebird_code'].append(row['ebird_code'])
            transformed_dict['filename'].append(file_name.split("/")[-1])


# %%
plt.plot(y)


# %%
mel_spec = librosa.feature.melspectrogram(y=y, sr=SR)
mel_spec = librosa.power_to_db(mel_spec, ref=np.max)


# %%
mel_spec.shape


# %%
def mono_to_color(X: np.ndarray,
                  mean=None,
                  std=None,
                  norm_max=None,
                  norm_min=None,
                  eps=1e-6):
    """
    Code from https://www.kaggle.com/daisukelab/creating-fat2019-preprocessed-data
    """
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V


# %%
image = mono_to_color(mel_spec)
plt.imshow(image)
image.shape


# %%
# cowsth
librosa.display.specshow(mel_spec, y_axis='mel')


# %%
# cowsth
librosa.display.specshow(mel_spec, y_axis='mel')


# %%
# aldfly
librosa.display.specshow(mel_spec, y_axis='mel')


# %%


