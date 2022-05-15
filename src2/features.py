import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

def extract_features(data, sr):
    D = np.abs(librosa.stft(data, n_fft=512))
    mel = librosa.feature.melspectrogram(S=D, sr=sr)
    #librosa.display.specshow(mel, sr=sr, x_axis='time', y_axis='mel')
    #plt.show()
    return mel
