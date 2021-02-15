import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv1D, MaxPooling1D
import csv
import pandas as pd
import collections
import pathlib
import re
import string
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras import utils
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

def mlp1(train_x, train_y, test1_x, test1_y):
    model = Sequential([
        Conv1D(16, 1,padding="valid",activation="relu",strides=1,input_shape=(1,MAX_SEQUENCE_LENGTH,)),
        MaxPooling1D(5, data_format='channels_first'),   
        Conv1D(8, 1,padding="valid",activation="relu"),
        MaxPooling1D(5, data_format='channels_first'),
        Dense(32, activation='sigmoid'),

        Dense(6,activation='softmax')
        
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    #model.summary()
    model.fit(train_x, train_y, epochs=5)
    #model.summary()

    print("Evaluating MLP1 on test set 1")
    ynew = model.predict_classes(test1_x)
    for i in range(len(ynew)):
        print("Correct = %s Predicted = %s, Song = %s" %( np.where(test1_y[i][0]==1)[0],ynew[i],validateSongs[i]))
    return model.evaluate(test1_x, test1_y)

def mlp2(train_x, train_y, test1_x, test1_y):

    model = Sequential([
        Conv1D(32, 1,padding="valid",activation="relu",strides=1,input_shape=(1,MAX_SEQUENCE_LENGTH,)),
        MaxPooling1D(5, data_format='channels_first'),   
        Conv1D(5, 1,padding="valid",activation="relu"),
        MaxPooling1D(5, data_format='channels_first'),   
        Dense(12, activation='sigmoid'),
        Dense(6,activation='softmax')
        
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_x, train_y, epochs=5)

    print("Evaluating MLP2 the new one on test set 1")
    ynew = model.predict_classes(test1_x)
    for i in range(len(ynew)):
        print("Correct = %s Predicted = %s, Song = %s" %( np.where(test1_y[i][0]==1)[0],ynew[i],validateSongs[i]))
    return model.evaluate(test1_x, test1_y)



with open('newExcel.csv', 'r') as f:
    reader = pd.read_csv(f)
    genreList = reader["Genre"]
    lyricsList = reader["Lyrics"]
    lyricsTrain = []
    genreTrain = []

    for lyric in range(len(lyricsList)):
        lyricsTrain.append(str(lyricsList[lyric]))

    for item in genreList:
        thelist = [0,0,0,0,0,0]
        thelist[item] = 1
        genreTrain.append([thelist])

testGenre = []
with open('validateset.csv', 'r') as afile:
    reader = pd.read_csv(afile)

    print(reader.dtypes)
    genreListValidate = reader["Genre"]
    lyricsList = reader["Lyrics"]
    validateSong = reader["song"]
    validateLyrics = []
    validateGenre = []
    validateSongs = []

    for lyric in range(len(lyricsList)):
        validateLyrics.append(str(lyricsList[lyric]))

    for item in genreListValidate:
        thelist = [0,0,0,0,0,0]
        thelist[item] = 1
        validateGenre.append([thelist])

    for item in validateSong:
        validateSongs.append(item)


VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 250

int_vectorize_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH)

int_vectorize_layer.adapt(lyricsTrain)

def int_vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return int_vectorize_layer(text), label

new_lyricsTrain = []
for i in range(len(lyricsTrain)):
    new_lyricsTrain.append(int_vectorize_text(lyricsTrain[i],genreList[i])[0])

new_lyricsValidate = []
for i in range(len(validateLyrics)):
    new_lyricsValidate.append(int_vectorize_text(validateLyrics[i],validateGenre[i])[0])


genreTrain=np.asarray(genreTrain)
new_lyricsTrain=np.asarray(new_lyricsTrain)

# duplicate entries
genreTrain= np.repeat(genreTrain,25,0)
new_lyricsTrain = np.repeat(new_lyricsTrain,25,0)

new_lyricsValidate = np.asarray(new_lyricsValidate)
validateGenre = np.asarray(validateGenre)


mlp1(new_lyricsTrain, genreTrain ,new_lyricsValidate, validateGenre)
mlp2(new_lyricsTrain, genreTrain ,new_lyricsValidate, validateGenre)