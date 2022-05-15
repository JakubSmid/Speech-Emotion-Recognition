from datasets import *
import matplotlib.pyplot as plt
import numpy as np
import librosa

dataset = load_crema_dataset() + load_emo_dataset() + load_ravdess_dataset()

for duration in np.arange(1, 3.2, 0.2):
    indexes = dict((e,0) for e in CREMA_EMOTIONS.values())
    data_splitted = []
    
    for sample in dataset:
        data, sr = librosa.load(sample[0], sr=None)
        size = int(duration*sr)
        usable_data = []
        
        for i in range(0, data.size, size):
            data_splitted.append(data[i:i+size].size/sr)
            if data[i:i+size].size == size:
                usable_data.append(data[i:i+size])
        
        indexes[sample[1]] += len(usable_data)*duration
    
    plt.figure(figsize=(15,8))
    plt.subplot(1, 2, 1)
    plt.bar(indexes.keys(), indexes.values())
    plt.title(f"delka strihu: {round(duration, 2)} s\n pocet vzorku na nahravce: {size}")
    plt.ylabel("delka pouzitelnych nahravek (s)")
    plt.ylim(top=10000)
    plt.locator_params('y', nbins=10)
    plt.grid(axis='y')
    
    plt.subplot(1, 2, 2)
    plt.hist(data_splitted, bins=30)
    plt.title("Histogram vsech delek po splitu")
    plt.ylabel("pocet nahravek")
    plt.xlabel("delka nahravek (s)")
    plt.grid(axis='y')
    plt.ylim(top=20000)