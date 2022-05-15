#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import librosa
import numpy as np
from datasets import *
from features import *
from augmentation import *
import time
import shutil
import sys
import random

dataset = load_crema_dataset() + load_emo_dataset() + load_ravdess_dataset()

duration = 1.8
print(f"Features duration: {duration} s\n")

# eta
n=0
n_last=0
last_time = time.time()
for sample in dataset:
    data, sr = librosa.load(sample[0], sr=None)
    cutted_size = int(sr*duration)
    n += data.size // cutted_size

# vytvoreni slozek s kategoriemi
shutil.rmtree('./features/')
for e in CREMA_EMOTIONS.values():
    os.makedirs("./features/" + e, exist_ok=True)

# funkce vrati cestu pro ulozeni daneho souboru
indexes = dict((e,0) for e in CREMA_EMOTIONS.values())
def path(sample):
    indexes[sample[1]] += 1
    return "./features/" + sample[1] + "/" + str(indexes[sample[1]]).zfill(4) + ".npy"

# generovani
for index, sample in enumerate(dataset):
    data, sr = librosa.load(sample[0], sr=None)
    size = int(duration*sr)
    data_splitted = []
    
    # split data
    for i in range(0, data.size, size):
        if data[i:i+size].size == size:
            data_splitted.append(data[i:i+size])
    
    # save splitted
    for wave in data_splitted:
        np.save(open(path(sample), "wb"), extract_features(wave, sr))
        np.save(open(path(sample), "wb"), extract_features(noise(wave), sr))
        np.save(open(path(sample), "wb"), extract_features(shift(wave), sr))
    
    # print ETA
    n_last += len(data_splitted)
    if n_last >= 100:
        n -= n_last
        eta = time.strftime("%M m %S s", time.gmtime( (time.time()-last_time)*(n//n_last)) )
        print(f"ETA: {eta} \t Progress: {index/len(dataset)*100:.1f} %")
        n_last = 0
        last_time = time.time()
        
print("Done...\n\n", indexes)
