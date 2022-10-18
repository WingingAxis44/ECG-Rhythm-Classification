import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import cv2
from sklearn.preprocessing import (MaxAbsScaler)
from imblearn.over_sampling import ( RandomOverSampler)


#This oversamples the disease class by default
#If it proves useful, it can be adapted to allow for oversampling either class

def oversample(X,Y):
  
    #oversamp = SMOTE()
    oversamp = RandomOverSampler()
    dim2 = X.shape[1]
    dim3 = X.shape[2]
    X = X.reshape(len(X),-1)

    Y = Y if isinstance(Y,list) else Y.tolist()
    X,Y = oversamp.fit_resample(X,Y)
  
    X = X.reshape(int(X.shape[0]), dim2, dim3)
    

    return X, np.asarray(Y, dtype=np.int8)


def normalizeSegment(X):

    row = 0
    scaler = MaxAbsScaler(copy=False)
    for sample in X:
        X[row] = scaler.fit_transform(sample)
        row += 1



#Applies bandstop filter for powerline interference removal
#Also applies bandpass filter to remove baseline wander and some high frequency noise
def denoise(X, fs = 500):

   # order = int(0.3 * fs)
    order = 4
    sos_stop = signal.butter(order, [59.0,61.0], btype='bandstop', output='sos', fs = fs)
    
    
    sos_pass = signal.butter(order, [0.5,100.0], btype='bandpass', output='sos', fs = fs)
    
    row = 0

    for sample in X:

        X[row] = signal.sosfiltfilt(sos_stop,sample)
        X[row] = signal.sosfiltfilt(sos_pass,sample)
       
        row = row + 1
        