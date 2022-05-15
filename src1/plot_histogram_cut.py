from datasets import *
import matplotlib.pyplot as plt
import numpy as np
import librosa

dataset = load_crema_dataset() + load_emo_dataset() + load_ravdess_dataset()

durations = np.arange(0.5, 3.5, 0.1)
usable_time = []
usable_n = []
for duration in durations:
    time = 0
    n = 0
    for sample in dataset:
        data, sr = librosa.load(sample[0], sr=None)
        cutted_size = int(sr*duration)
        time += (data.size // cutted_size) * duration
        n += data.size // cutted_size
    print(time)
    usable_time.append(time)
    usable_n.append(n)

fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1]}, figsize=(10, 6))
a0.bar(durations, usable_n, width=0.1)
a0.grid(axis='both')
a0.set_ylabel("použitelné nahrávky")
a0.set_xlabel("délka střihu [s]")
a0.set_title("Histogram použitelných nahrávek v závislosti na délce střihu")

a1.bar(durations, usable_time, width=0.1)
a1.grid(axis='both')
a1.set_ylabel("použitelná délka [s]")
a1.set_xlabel("délka střihu [s]")
a1.set_title("Histogram použitelného času v závislosti na délce střihu")

title = "Histogramy delek po strihu"
fig.tight_layout()
fig.savefig("images/" + title + ".pdf", dpi=300)
plt.show()