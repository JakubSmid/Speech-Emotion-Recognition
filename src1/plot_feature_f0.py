from datasets import *
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

title = "Základní frekvence"

def fundamental(D_db, sr, n_fft=512, fmin=65, fmax=2093):
    maxima = []
    for i,column in enumerate(D_db.T):
        column_maxima = column[0]
        maxima_index = 0
        for j, value in enumerate(column):
            if value > column[0]+9: # pokud je hodnota vetsi jak nejnizsi frekv. +30dB
                if value < column_maxima and column[j+1] < column_maxima and column[j+2] < column_maxima: # pak hledej maximum
                    maxima_index = j-1
                    break
                column_maxima = value
        if maxima_index != 0 and maxima_index*sr//n_fft<=fmax and maxima_index*sr//n_fft>=fmin:
            maxima.append(maxima_index*sr//n_fft)
    maxima = np.array(maxima)
    return maxima

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

# pyin
f0, voiced_flag, voiced_probs = librosa.pyin(data, fmin=65, fmax=2093, sr=sr)
times = librosa.times_like(f0, sr=sr)
a1.plot(times, f0, label='Probabilistic YIN (mean: 477 Hz, std: 95 Hz)', color='blue', linewidth=2)

# muj algoritmus
maxima = []
m = []
for i,column in enumerate(D.T):
    column_maxima = column[0]
    maxima_index = 0
    for j, value in enumerate(column):
        if value > column[0]+9: # pokud je hodnota vetsi jak nejnizsi frekv. +9 dB
            if value < column_maxima and column[j+1] < column_maxima and column[j+2] < column_maxima: # pak hledej maximum
                maxima_index = j-1
                break
            column_maxima = value
    if maxima_index == 0 or maxima_index*sr//512>=2093 or maxima_index*sr//512<=65:
        maxima_index = -1
    else:
        m.append(maxima_index*sr//512)
    maxima.append(maxima_index*sr//512)
maxima = np.array(maxima)

times = librosa.times_like(maxima, sr=sr, hop_length=128)
a1.scatter(times, maxima, c='lime', label='Vlastní algoritmus (mean: 435 Hz, std: 196 Hz)', s=5)

# druha osa
a1.set_title(title)
a1.legend(loc='upper left', framealpha=1)
a1.set_xlabel("Čas [s]")
a1.set_ylabel("Frekvence [Hz]")

# ulozeni obrazku
fig.suptitle("Nahrávka naštvání (RAVDESS)", fontsize=16)
fig.colorbar(img, ax=a1, format="%+2.0f dB", location="bottom", shrink=0.5)
fig.tight_layout()
fig.savefig("images/" + title + ".pdf", dpi=300)
fig.show()