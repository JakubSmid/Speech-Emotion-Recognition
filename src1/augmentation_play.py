from augmentation import *
from datasets import *
import random
import librosa
import simpleaudio

dataset = load_ravdess_dataset() + load_crema_dataset() + load_emo_dataset()
sample = random.choice(dataset)

wave, sr = librosa.load(sample[0], sr=None)
print("Playing original")
simpleaudio.WaveObject(wave, num_channels=1, bytes_per_sample=4, sample_rate=sr).play().wait_done()
print("Playing effect")
simpleaudio.WaveObject(test_noise(wave), num_channels=1, bytes_per_sample=4, sample_rate=sr).play().wait_done()