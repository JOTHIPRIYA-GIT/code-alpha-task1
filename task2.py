pip install librosa numpy pandas scikit-learn tensorflow matplotlib
import librosa
import os
import numpy as np

def extract_mfcc(file_path, max_pad_len=174):
    try:
        audio, sr = librosa.load(file_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, TimeDistributed, Flatten

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(40, 174, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(8, activation='softmax'))  # 8 emotion classes, adjust as per dataset
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
