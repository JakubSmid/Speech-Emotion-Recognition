# -*- coding: utf-8 -*-
print("Loading libraries...")

import numpy as np
import sounddevice as sd # for recording
from src1.features import extract_features as ef1 # conversion speech to features
from src2.features import extract_features as ef2 # conversion speech to features
from sklearn.preprocessing import OneHotEncoder # for encoding predictions
import signal # to handle termination signal
import sys
from tensorflow import keras

def terminate(sig, frame):
    stream.close()
    print('\nTerminated! Closing input stream...')
    sys.exit(0)

# disable warnings
import warnings
import os
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# select model which will be used
select = ""
while not select.isnumeric() or int(select) not in range(1,5):
    select = input("""
Press CTRL+C for kill this app.

 1 \t src1/model
 2 \t src1/test_model
 3 \t src2/model
 4 \t src2/test_model
Insert corresponding number of model which will be used: """)
    if not select.isnumeric():
        print(f"'{select}' is not a number.")
    if select.isnumeric() and int(select) not in range(1,5):
        print(f"'{select}' is out of range.")

# load nn model
print("Loading NN model...")
if select == "1":
    model = keras.models.load_model('src1/model')
if select == "2":
    model = keras.models.load_model('src1/model_test')
if select == "3":
    model = keras.models.load_model('src2/model')
if select == "4":
    model = keras.models.load_model('src2/model_test')

fs = 11025
duration = 1.8

# one hot encoding
emotions = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad']
emotions_cs = ['hněv', 'znechucení', 'strach', 'štěstí', 'neutrální', 'smutek']
encoder = OneHotEncoder()
encoder.fit(np.array(emotions).reshape(-1,1))

# open stream
stream = sd.InputStream(samplerate=fs, channels=1)
signal.signal(signal.SIGINT, terminate)
stream.start()
print("Recording...\n")

while True:
    recording = stream.read(int(fs*duration))[0]
    if select == "1" or select == "2":
        feature = ef1(recording.flatten(), fs)
    else:
        feature = ef2(recording.flatten(), fs)
    feature = feature[np.newaxis, ..., np.newaxis]
    prediction = model.predict(feature)
    print(emotions_cs[prediction.argmax()])
