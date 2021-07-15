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
# from featureextractionGMM import extract_features
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
import json
import random

features_num = 9
modelpath = sys.argv[1]             # GMM  folder

gmm_files = [os.path.join(modelpath, fname)
             for fname in os.listdir(modelpath) if fname.endswith('.gmm')]

# Load the Gaussian gender Models
models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
speakers = [fname.split("/")[-1].split(".gmm")[0] for fname in gmm_files]

error = 0
total_sample = 0.0


def speechRecognition(AUDIO_FILE):
    ##  AUDIO_FILE = 'C:/Users/aksat/Downloads/AUDIO_DATA/2020-03-03T10_38_56.553Z.wav'
    # use the audio file as the audio source
    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source)  # read the entire audio file
        try:
            return(r.recognize_google(audio))
        except:
            return("can't convert")


def transcription():
    global transcript
    transcript = []
    for i in response['results']:
        flag = 0
        f1 = 0
        for j in i['alternatives']:
            if flag == 0:
                # print(j['transcript'])
                flag = 1
                transcript.append(j['transcript'])


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
    # print('mfcc....', mfcc_feature)
    mfcc_feature = preprocessing.scale(mfcc_feature)
    # print('mfcc.....0', mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature, delta))
    return combined


def confidence_count():
    for i in response['results']:
        flag = 0
        f1 = 0
        for j in i['alternatives']:
            try:
                # print(j['confidence'])
                confidence.append(j['confidence'])
                # print(confidence)
            except:
                continue


def detect(source, sp):
    sec1_temp_folder = "D:/ATOM docs/fyp_desktop/src/temp1sec_trans_files/"
    # print(sp)
    winner_list = np.zeros(len(speakers))
    with contextlib.closing(wave.open(source, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    sourceAudio = AudioSegment.from_wav(source)
    # sourceAudio = AudioSegment.from_mp3(source)
    # sr, sourceAudio = read(newAudio)
    for i in range(101):
        n = random.random() * duration
        newAudio = sourceAudio[(n)*1000: (n+1)*1000]
        newAudio.export(sec1_temp_folder + str('temp') + '.wav', format='wav')
        # newAudio = newAudio[1000:]
        sr, audio = read(sec1_temp_folder+"temp.wav")
        vector = extract_features(audio, sr)
        log_likelihood = np.zeros(len(models))
        for j in range(len(models)):
            gmm = models[j]
            scores = np.array(gmm.score(vector))
            log_likelihood[j] = scores.sum()
        # print(log_likelihood)
        winner = np.argmax(log_likelihood)
        winner_list[winner] += 1
    # for i in range(int(duration)-1):
    #     newAudio = sourceAudio[(i)*1000: (i+1)*1000]
    #     newAudio.export(sec1_temp_folder + str('temp') + '.wav', format='wav')
    #     # newAudio = newAudio[1000:]
    #     sr, audio = read(sec1_temp_folder+"temp.wav")
    #     vector = extract_features(audio, sr)
    #     log_likelihood = np.zeros(len(models))
    #     for j in range(len(models)):
    #         gmm = models[j]
    #         scores = np.array(gmm.score(vector))
    #         log_likelihood[j] = scores.sum()
    #     print(log_likelihood)
    #     winner = np.argmax(log_likelihood)
    #     winner_list[winner] += 1

    # sr, audio = read(source)
    # vector = extract_features(audio, sr)
    # log_likelihood = np.zeros(len(models))
    # for i in range(len(models)):
    #     gmm = models[i]
    #     scores = np.array(gmm.score(vector))
    #     log_likelihood[i] = scores.sum()
    # print(log_likelihood)
    # winner = np.argmax(log_likelihood)
    winner = np.argmax(winner_list)
    detected_winner = speakers[winner]
    print(winner_list)
    print('winner..', detected_winner, '....\n\n')
    # f = open(transcript_file, "a")
    # f.write(detected_winner+" " + sp + '\n\n')
    # print(detected_winner, "|", sp)
    # sys.stdout.flush()
    if confidence[count] < .75 and sp != "can't convert":
        f = open(transcript_file, "a")
        f.write(detected_winner+" " + sp + '\n\n')
        print(detected_winner, "|", sp)
        sys.stdout.flush()
        # print(sp)
        # sys.stdout.flush()
    else:
        f = open(transcript_file, "a")
        f.write(detected_winner+" " + transcript[count] + '\n\n')
        print(detected_winner, "|", transcript[count])
        sys.stdout.flush()
        # print(transcript[count])
        # sys.stdout.flush()


def trim_audio(l1):
    # print('yyyyyyyyyyyyyy')
    global raw_audiofile_folder, new_audiofile_folder, source, sp, confidence
    audioname = "trimmedaudio.wav"
#     count=0
#     for path in [x[0] for x in os.walk(raw_audiofile_folder)]:
#           # count1=0
#         for filename in glob.glob(os.path.join(path, '*.wav')):
#               # count1+=1
#             print(filename)
    fname = Audio_fil
    with contextlib.closing(wave.open(fname, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        # print(duration)
        duration = duration
    newAudio = AudioSegment.from_wav(fname)

    trimmed_sound = newAudio[((l1[0]*1000)):((l1[1]*1000))]
    # trimmed_sound = newAudio[((l1[0]*1)):((l1[1]*1))]

    trimmed_sound.export(raw_audiofile_folder + str(audioname), format='wav')
    source = raw_audiofile_folder+audioname
    temp = AudioSegment.from_wav(source)
    # print(source)


def speech2text(Audio_fil):
    global count, response, confidence
    count = 0
    url = 'https://api.eu-gb.speech-to-text.watson.cloud.ibm.com/instances/5e76a984-9748-4132-8154-62c476c787fe/v1/recognize?timestamps=true&max_alternatives=3'
    headers = {"Content-Type": "audio/wav"}
    fin = open(Audio_fil, 'rb')
    files = {'data-binary': fin}
    r = requests.post(url=url, data=fin, auth=(
        'apikey', 'mzlzC5ChvUmycNBs36vGsaVfUxHo93Mrcf_FWULFelxB'), headers=headers)
    response = json.loads(r.content)
    transcription()
    confidence_count()
    print(confidence[count])
    # audioData = AudioSegment.from_wav(Audio_fil)
    # currTime = 0
    # chunk_size = 10
    # silenceTime = 0
    # silence_threshold = -50
    # startTime = 0
    # while(currTime < len(audioData)):
    #     startTime = currTime
    #     while audioData[currTime:currTime+chunk_size].dBFS > silence_threshold and currTime < len(audioData):
    #         # print(sound[trim_ms:trim_ms+chunk_size].dBFS)
    #         currTime += chunk_size
    #     silenceTime = currTime
    #     while audioData[silenceTime:silenceTime+chunk_size].dBFS < silence_threshold and silenceTime < len(audioData):
    #         # print(sound[trim_ms:trim_ms+chunk_size].dBFS)
    #         silenceTime += chunk_size
    #     if (silenceTime > 500 and silenceTime-startTime > 1000):
    #         print(startTime, silenceTime)
    #         trim_audio([startTime, silenceTime])
    #         sp = speechRecognition(source)
    #         # print(sp)
    #         detect(source, sp)
    #         count += 1
    #     currTime = silenceTime
    for i in response['results']:
        flag = 0
        f1 = 0
        for j in i['alternatives']:
            try:
                #         print(j['transcript'])
                for k in j['timestamps']:
                    if f1 == 0:
                        start = k[1]
                        f1 = 1
    #                 print(k[0])
                    end = k[2]
            except:
                flag = 1
                break
        # print(start, end)
        trim_audio([start, end])
        sp = speechRecognition(source)
        # print(sp)
        detect(source, sp)
        count += 1


transcript_file = sys.argv[4]       # text file where transcript will be saved
a_file = sys.argv[2]              # audio file which is to be transcripted
audio_chunk_folder = sys.argv[3]    # temp audio chunk folder path

# mp3audio = AudioSegment.from_mp3(a_file)
# mp3audio.export(
#     str('D:/ATOM docs/fyp_desktop/src/waveAudio') + '.wav', format='wav')
# a_file = r"D:/ATOM docs/fyp_desktop/src/waveAudio.wav"
f = open(transcript_file, "w+")
Audio_fil = a_file
# Audio_fil= r"D:/VIT/FYP/fyp_data/train_audio/arpan/arpan_nr.wav"
raw_audiofile_folder = audio_chunk_folder
confidence = []
speech2text(Audio_fil)
