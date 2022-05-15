import librosa
import numpy as np

def extract_features(data, sr):
    result = np.array([])
    
    # ZCR
    zcr = np.mean(librosa.feature.zero_crossing_rate(data, 2048//3, 512//3).T, axis=0)
    result = np.hstack((result, zcr))
    
    
    # Chroma
    stft = librosa.stft(data, n_fft=3072)
    S = np.abs(stft) # umocneni neni dobra cesta
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    chroma_stft = np.mean(chroma.T, axis=0)
    result = np.hstack((result, chroma_stft))
    
    '''
    # MFCC    
    stft = librosa.stft(data, n_fft=512)
    D = np.abs(stft)**2
    mel = librosa.feature.melspectrogram(S=D, sr=sr)
    mfcc = np.mean(librosa.feature.mfcc(S=mel, sr=sr).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally
    '''
    
    # RMS
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))
    
    # Mel spectrogram
    stft = librosa.stft(data, n_fft=2048)
    D = np.abs(stft)**2
    mel = librosa.feature.melspectrogram(S=D, sr=sr)
    mel = np.mean(mel.T, axis=0)
    result = np.hstack((result, mel))
    
    # f0: mean and std_dev
    '''
    f0, voiced_flag, voiced_probs = librosa.pyin(data, fmin=65, fmax=2093, sr=sr)
    if np.isnan(f0).all():
        f0 = 0
    mean = np.nanmean(f0)
    std_dev = np.nanstd(f0)
    result = np.hstack((result, mean))
    result = np.hstack((result, std_dev))'''
    
    '''
    stft = librosa.stft(data, n_fft=512)
    D = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    
    maxima = []
    for i,column in enumerate(D.T):
        column_maxima = column[0]
        maxima_index = 0
        for j, value in enumerate(column):
            if value > column[0]+9: # pokud je hodnota vetsi jak nejnizsi frekv. +9 dB
                if value < column_maxima: # pak hledej maximum
                    maxima_index = j-1
                    break
                column_maxima = value
        if maxima_index == 0 or maxima_index*sr//512>=2093 or maxima_index*sr//512<=65:
            maxima_index = 0
        maxima.append(maxima_index*sr//512)
    maxima = np.array(maxima)
    
    nonzeros = [i for i in maxima if i != 0]
    if len(nonzeros) == 0:
        nonzeros = 0
    mean = np.mean(nonzeros)
    std_dev = np.std(nonzeros)
    result = np.hstack((result, mean))
    result = np.hstack((result, std_dev))
    '''
    
    return result
