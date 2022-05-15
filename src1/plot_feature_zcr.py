from datasets import *
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

title = "Zero crossing rate"

# nacteni nahravky
sample = load_ravdess_dataset()[14]
data, sr = librosa.load(sample[0], sr=None)

# generovani grafu
fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1]}, figsize=(10, 6))

# prvni osa
librosa.display.waveplot(data, sr=sr, ax=a0)
a0.set_title('Časová oblast')
a0.set_xlabel("Čas [s]")
a0.set_ylabel("Amplituda")

# generovani features
rms = librosa.feature.zero_crossing_rate(data, 2048//3, 512//3)
times = librosa.times_like(rms, sr=sr)
a1.plot(times, rms[0])

# druha osa
a1.set_title(title)
a1.set_xlabel("Čas [s]")
a1.set_xlim(0, data.size/sr)
a1.set_ylabel("Rate")

# ulozeni obrazku
fig.suptitle("Nahrávka naštvání (RAVDESS)", fontsize=16)
fig.tight_layout()
fig.savefig("images/" + title + ".pdf", dpi=300)
fig.show()