# -*- coding: utf-8 -*-

import numpy as np
import random
import os
import sys
import matplotlib.pyplot as plt

from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight


#%% Data preparation

emotions = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad']
emotions_cs = ['hněv', 'znechucení', 'strach', 'štěstí', 'neutrální', 'smutek']
encoder = OneHotEncoder()
encoder.fit(np.array(emotions).reshape(-1,1))

train_files = []
for e in emotions:
    for file in os.listdir("./test/" + e):
        train_files.append(e+"/"+file)

test_files = []
for e in emotions:
    for file in os.listdir("./test_valid/" + e):
        test_files.append(e+"/"+file)
        
noise_files = []
for e in emotions:
    for file in os.listdir("./test_noise/" + e):
        noise_files.append(e+"/"+file)

class DataGenerator(Sequence):
    def __init__(self, files, batch_size, encoder, dataset_type, input_shape=142, shuffle=True):
        self.files = files
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_shape = input_shape
        self.n = len(files)
        self.on_epoch_end()
        self.dataset_type = dataset_type
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.files)
    
    def __getitem__(self, index):
        batch = self.files[index * self.batch_size:(index + 1) * self.batch_size]
        X = np.zeros((self.batch_size, self.input_shape))
        y = []
        for i,file in enumerate(batch):
            if self.dataset_type == 'train':
                X[i] = np.load("./test/" + file)
            if self.dataset_type == 'valid':
                X[i] = np.load("./test_valid/" + file)
            if self.dataset_type == 'noise':
                X[i] = np.load("./test_noise/" + file)
            y.append([file.split("/")[0]])
        X = np.expand_dims(X, 2)
        y = encoder.transform(y).toarray()
        return X,y
    
    def __len__(self):
        return self.n // self.batch_size

batch = 64
train_gen = DataGenerator(train_files, batch, encoder, 'train')
class_weights = dict((x, 0) for x in range(len(emotions)))
for train_batch in train_gen:
    for x in train_batch[1]:
        class_weights[x.argmax()] += 1
class_weights_max = max(class_weights.values())
class_weights = {k: class_weights_max/v for k, v in class_weights.items()}

test_gen = DataGenerator(test_files, batch, encoder, 'valid')
noise_gen = DataGenerator(noise_files, batch, encoder, 'noise')

#%% Model creation
model=Sequential()

model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(142,1)))
model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

model.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
model.add(Dropout(0.2))

model.add(Conv1D(32, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(units=32, activation='relu'))

model.add(Dense(units=6, activation='softmax'))
model.compile(optimizer = 'rmsprop' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

model.summary()


#%% Fit

history = model.fit(train_gen, validation_data=test_gen, epochs=10, class_weight=class_weights)

#%%
accuracy = model.evaluate(test_gen)[1]*100
print(f"Accuracy of our model on test data : {accuracy} %")

epochs = [i for i in range(10)]

train_acc = np.array(history.history['accuracy'])*100
test_acc = np.array(history.history['val_accuracy'])*100

plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(10, 6))

plt.plot(epochs, train_acc, label='Trénovací množina')
plt.plot(epochs, test_acc, label='Validační množina')
plt.title(f'Přesnost na dávce dat v průběhu trénování\n(přesnost modelu: {accuracy:.2f} %)')
plt.legend()
plt.xlabel("Epocha")
plt.ylabel("Přesnost [%]")
plt.grid(axis='y')
plt.ylim(10, 100)

title = f"1"
plt.tight_layout()
plt.savefig("images/test_" + title + ".pdf", dpi=300)
plt.show()

#%% Confusion matrix
test_gen = DataGenerator(test_files, batch, encoder, 'valid', shuffle=False)
predictions = model.predict(test_gen)
predictions = np.argmax(predictions, axis=1)
y_true = []
for sbatch in test_gen:
    for label in sbatch[1].argmax(axis=1):
        y_true.append(label)

matrix = confusion_matrix(y_true, predictions, normalize="true")*100
cmd = ConfusionMatrixDisplay(matrix, display_labels=emotions_cs)

plt.rcParams.update({'font.size': 14})
fig,ax = plt.subplots(figsize=(10, 8))
cmd.plot(ax=ax, values_format='.1f')

plt.title(f'Normalizovaná confusion matrix [%]')
plt.xlabel("Predikce")
plt.ylabel("Anotovaná třída")


plt.tight_layout()
plt.savefig("images/test_cm" + title + ".pdf", dpi=300)
plt.show()

#%% Noise accuracy
accuracy = model.evaluate(noise_gen)[1]*100
print(f"Accuracy of our model on noisy data : {accuracy} %")

#%% Confusion matrix
noise_gen = DataGenerator(noise_files, batch, encoder, 'noise', shuffle=False)
predictions = model.predict(noise_gen)
predictions = np.argmax(predictions, axis=1)
y_true = []
for sbatch in noise_gen:
    for label in sbatch[1].argmax(axis=1):
        y_true.append(label)

matrix = confusion_matrix(y_true, predictions, normalize="true")*100
cmd = ConfusionMatrixDisplay(matrix, display_labels=emotions_cs)

plt.rcParams.update({'font.size': 14})
fig,ax = plt.subplots(figsize=(10, 8))
cmd.plot(ax=ax, values_format='.1f')

plt.title(f'Normalizovaná confusion matrix [%]')
plt.xlabel("Predikce")
plt.ylabel("Anotovaná třída")


plt.tight_layout()
plt.savefig("images/test_noisy_cm" + title + ".pdf", dpi=300)
plt.show()

