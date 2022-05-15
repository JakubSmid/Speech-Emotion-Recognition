import numpy as np
import librosa

def test_noise(data):
    noise_amp = 0.02*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return np.float32(data)

def noise(data):
    noise_amp = 0.05*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return np.float32(data)

def stretch(data):
    # krome natazeni prida jeste echo, jelikoz librosa to pocita pres stft - nezachova se faze
    random_rate = np.random.uniform(0.6, 0.9)
    return librosa.effects.time_stretch(data, random_rate)

def shift(data):
    factor = 0
    while abs(factor) <= 0.2:
        factor = np.random.uniform(-0.5, 0.5)
    shift_range = int(data.size*factor)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate):
    # krome natazeni prida jeste echo, jelikoz librosa to pocita pres stft - nezachova se faze
    pitch_factor = 0
    while abs(pitch_factor) <= 0.5:
        pitch_factor = np.random.uniform(-1, 1)
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)
