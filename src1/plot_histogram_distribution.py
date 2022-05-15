from datasets import *
import matplotlib.pyplot as plt
import numpy as np
import librosa

emotions = list(CREMA_EMOTIONS.values()) + list(RAVDESS_EMOTIONS.values()) + list(EMO_EMOTIONS.values())
emotions = np.unique(emotions)

emo = []
for emotion in emotions:
    dataset = load_emo_dataset(True)
    files = [e for e in dataset if e[1] == emotion]
    d = 0
    for file in files:
        d += librosa.get_duration(filename=file[0])
    emo.append(d)

ravdess = []
for emotion in emotions:
    dataset = load_ravdess_dataset(True)
    files = [e for e in dataset if e[1] == emotion]
    d = 0
    for file in files:
        d += librosa.get_duration(filename=file[0])
    ravdess.append(d)
    
crema = []
for emotion in emotions:
    dataset = load_crema_dataset()
    files = [e for e in dataset if e[1] == emotion]
    d = 0
    for file in files:
        d += librosa.get_duration(filename=file[0])
    crema.append(d)

# generovani grafu
r = np.arange(emotions.size)
width = 0.3

plt.figure(figsize=(10, 6))
plt.bar(r-width, emo, width=width, label="EmoDB")
plt.bar(r, ravdess, width=width, label="RAVDESS")
plt.bar(r+width, crema, width=width, label="CREMA-D")

plt.xticks(r, emotions)
plt.legend(framealpha=1)
plt.grid(axis='y')
plt.ylabel("délka [s]")

# ulozeni obrazku
title = "Rozdělení časů nahrávek"
plt.suptitle(title, fontsize=16)
plt.tight_layout()
plt.savefig("images/" + title + ".pdf", dpi=300)
plt.show()