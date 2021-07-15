
from sklearn.metrics import pairwise_distances
from sklearn import metrics
import python_speech_features as mfcc
import sys
# import python_speech_features as mfcc
import os.path
from sklearn import preprocessing
import glob
import shutil
# import joblib
# import json
import time
import requests
from sklearn.mixture import GaussianMixture as GMM
from featureextractionGMM import extract_features
from scipy.io.wavfile import read
import pickle
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
# import seaborn as sns
# import numpy as numpy
import random
import pandas as pd
import contextlib
import wave
import math
import pydub
import numpy as np
# import librosa.display
# import cv2
import os
# import matplotlib.pyplot as plt
import librosa
# from PIL import Image
# import PIL
# from skimage import io
# from skimage import color
from os import path
import speech_recognition as sr
from pydub.silence import split_on_silence
from pydub import AudioSegment
import warnings
warnings.filterwarnings("ignore")


features_num = 9

# for loading and visualizing audio files

# for loading and visualizing audio files


#from speakerfeatures import extract_features

# print("hello world")


def detect_leading_silence(sound, silence_threshold=-30, chunk_size=5):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0  # ms

    assert chunk_size > 0  # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        # print(sound[trim_ms:trim_ms+chunk_size].dBFS)
        trim_ms += chunk_size
    # print(trim_ms)
    return trim_ms


def train(name, text_file_name, GMM_folder, username):
    # path joining version for other paths
    DIR = new_audiofile_folder
    numberofsamples = len([name for name in os.listdir(DIR)
                           if os.path.isfile(os.path.join(DIR, name))])
    m = new_audiofile_folder+name
    count = 1
#     m=new_audiofile_folder+name
    # train("mithil",25)
    l1 = []
    l2 = list(range(1, numberofsamples+1))
    # print('l2', l2)
    # print('num od samples', numberofsamples)
    s = [str(i) for i in l2]
#     open('file.txt', 'w').close()                      #empty the text file
    for i in s:
        l1.append(m+"_"+i+".wav")
    text_file = open(text_file_name, "wt")
    l1
    for i in l1:
        text_file.write(i+"\n")
    text_file.close()
    file_paths = open(text_file_name, 'r')
    dest = GMM_folder
    features = np.asarray(())
    for path in file_paths:
        path = path.strip()
        # print(path)

    # read the audio
        sr, audio = read(path)

        # extract 40 dimensional MFCC & delta MFCC features
        vector = extract_features(audio, sr)
        # print("YAY")
        if features.size == 0:
            features = vector
            # print("uyuyu")
        else:
            features = np.vstack((features, vector))
        # when features of 5 files of speaker are concatenated, then do model training
        # -> if count == 5: --> edited below
        if count == numberofsamples:
            # for num in range(2, 25):
                # gmm = GMM(n_components=num, max_iter=200,
                #           covariance_type='spherical', n_init=3)
                # gmm.fit(features)
                # score = gmm.predict(features)
                # print(metrics.silhouette_score(
                #     features, score, metric='euclidean'))
            gmm = GMM(n_components=4, max_iter=200,
                      covariance_type='spherical', n_init=3)
            gmm.fit(features)
            score = gmm.predict(features)
            print('spherical ', metrics.silhouette_score(
                features, score, metric='euclidean'))
            # dumping the trained gaussian model

            picklefile = username+".gmm"
#             print(pick)
            pickle.dump(gmm, open(dest + picklefile, "wb"))
            # print('+ modeling completed for speaker:', picklefile,
            #       " with data point = ", features.shape)
            features = np.asarray(())
            count = 0
        # else:
            # print("task left")
            # print(count)
        count = count + 1
    file_paths.close()


def calculate_delta(array):
    """Calculate and returns the delta of given feature vector matrix"""

    rows, cols = array.shape
    deltas = np.zeros((rows, features_num))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
                first = 0
            else:
                first = i-j
            if i+j > rows-1:
                second = rows-1
            else:
                second = i+j
            index.append((second, first))
            j += 1
        deltas[i] = (array[index[0][0]]-array[index[0][1]] +
                     (2 * (array[index[1][0]]-array[index[1][1]]))) / 10
    return deltas


def extract_features(audio, rate):
    """extract 20 dim mfcc features from an audio, performs CMS and combines 
    delta to make it 40 dim feature vector"""

    mfcc_feature = mfcc.mfcc(audio, rate, 0.025, 0.01,
                             features_num, nfft=1200, appendEnergy=True)
    mfcc_feature = preprocessing.scale(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature, delta))
    return combined


def trim_audio_1s(audioname):
    global raw_audiofile_folder, new_audiofile_folder
    count = 0
    # for path in [x[0] for x in os.walk(raw_audiofile_folder)]:
    #     # count1=0
    #     for filename in glob.glob(os.path.join(path, '*.wav')):
    #         # count1+=1
    #         print(filename)
    #         fname = filename
    #         with contextlib.closing(wave.open(fname, 'r')) as f:
    #             frames = f.getnframes()
    #             rate = f.getframerate()
    #             duration = frames / float(rate)
    #             print(duration)
    #             duration = duration
    #         newAudio = AudioSegment.from_wav(filename)
    #         # count=0
    #         for i in range(1, int(duration+1)):
    #             # count+=1
    #             if i == int(duration):
    #                 newAudio1 = newAudio
    #             else:
    #                 t1 = 1*1000
    #                 newAudio1 = newAudio[:t1]
    #                 newAudio = newAudio[t1:]
    #             start_trim = detect_leading_silence(newAudio1)
    #             end_trim = detect_leading_silence(newAudio1.reverse())
    #             trimmed_sound = newAudio1[start_trim:duration-end_trim]
    #             if len(trimmed_sound) >= 500:
    #                 count += 1
    #                 # print(count, len(trimmed_sound))
    #                 newAudio1.export(new_audiofile_folder +
    #                                  str(audioname) + str(count)+'.wav', format='wav')

    # count1+=1
    filename = raw_audiofile_folder
    # print(filename)
    fname = filename
    with contextlib.closing(wave.open(fname, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        # print('duration===> ', duration)
        # duration = duration
    newAudio = AudioSegment.from_wav(filename)
    # count=0
    for i in range(1, int(duration+1)):
        # count+=1
        if i == int(duration):
            newAudio1 = newAudio
        else:
            t1 = 1*1000
            newAudio1 = newAudio[:t1]
            newAudio = newAudio[t1:]
        start_trim = detect_leading_silence(newAudio1)
        end_trim = detect_leading_silence(newAudio1.reverse())
        trimmed_sound = newAudio1[start_trim:(1000-end_trim)]
        # print(len(trimmed_sound), duration)
        if len(trimmed_sound) >= 500:
            count += 1
            # print(new_audiofile_folder)
            newAudio1.export(new_audiofile_folder +
                             str(audioname) + str('_') + str(count)+'.wav', format='wav')
    # print('samples formed- ', count)


raw_audiofile_folder = sys.argv[1]
new_audiofile_folder = sys.argv[2]
textfile = sys.argv[3]
GMM_folder = sys.argv[4]
username = sys.argv[5]


for filename in os.listdir(new_audiofile_folder):
    file_path = os.path.join(new_audiofile_folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))
# audioname= 'mithil_'


def train_new(username):
    # with open(csv_file, 'r') as f:
    #     for row in (list(csv.reader(f))):
    #         new_name = row[-1]
    #             print((row[-1]))

    # Ahiya chunking and train call karideje
    trim_audio_1s("user")
    train("user", textfile, GMM_folder, username)


train_new(username)
print('done')
