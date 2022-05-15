from datasets import *
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

title = "Spektrogram"

# nacteni nahravky
sample = load_ravdess_dataset()[14]
data, sr = librosa.load(sample[0], sr=None)

# generovani grafu
fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]}, figsize=(10, 10))

# prvni osa
librosa.display.waveplot(data, sr=sr, ax=a0)
a0.set_title('Časová oblast')
a0.set_xlabel("Čas [s]")
a0.set_ylabel("Amplituda")

# generovani spektrogramu
stft = librosa.stft(data, n_fft=512)
D = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
img = librosa.display.specshow(D, sr=sr, y_axis='linear', x_axis='time', ax=a1, hop_length=128)

# druha osa
a1.set_title(title)
a1.set_xlabel("Čas [s]")
a1.set_ylabel("Frekvence [Hz]")

# ulozeni obrazku
fig.suptitle("Nahrávka naštvání (RAVDESS)", fontsize=16)
fig.colorbar(img, ax=a1, format="%+2.0f dB", location="bottom", shrink=0.5)
fig.tight_layout()
fig.savefig("images/" + title + ".pdf", dpi=300)
fig.show()