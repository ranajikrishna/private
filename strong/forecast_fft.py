
import sys
import pdb
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from scipy import fftpack

def forecast():

    return 

def reconstruct_signal(data, frac_harm, train_pc=0.8, plot=False):

    col = 'wave_height_51201h'
    data_train = data.iloc[:int(data.shape[0]*train_pc)]
    data_test = data.iloc[int(data.shape[0]*train_pc):]
    n = data_train.shape[0]
    t = np.arange(0,n)
    p = np.polyfit(t, data_train[col], 1)      # find linear trend in x
    data_detrend = data_train[col] - p[0] * t  # detrended x
#    tmp = fftpack.fft(np.array(data_detrend))
    data_fft = fftpack.fft(np.array(data_detrend))[0:int(n/2)]  # detrended x in frequency domain
#    tmp_f = fft.fftfreq(n)              # frequencies
    f = fftpack.fftfreq(n)[0:int(n/2)]      # frequencies
    indexes = list(range(int(n/2)))
    # sort indexes by amplitude in descending order.
    indexes.sort(key = lambda i: np.absolute(data_fft[i]))
    indexes.reverse()
 
    n_harm = len(data_fft)
    harm = int(np.ceil(frac_harm * n_harm))
    t = np.arange(0, n + data_test.shape[0])
    restored_sig = np.ones(t.size) * np.absolute(data_fft[0])/n
    for i in indexes[1:1 + harm]:
        ampli = 2 * np.absolute(data_fft[i]) / n   # amplitude
        phase = np.angle(data_fft[i])              # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    
    restored_sig += p[0] * t
    
    if plot:
        plt.close()
        plt.scatter(data.index, restored_sig, c='r', marker='o', label = 'extrapolation')
        plt.plot(data_train.index, data_train[col], 'b', label = 'train', linewidth = 3)
        plt.plot(data_test.index, data_test[col], 'g', label = 'test', linewidth = 3)
        plt.legend()

    data['restored'] = restored_sig
    data['error'] = data[col] - data['restored']
    pdb.set_trace()
    return data


def compute_fft(data):

    col = 'wave_height_51201h'
    n = data.shape[0]
    t = np.arange(0, n)
    p = np.polyfit(t, data[col], 1)         # find linear trend in x
    data_detrend = data[col] - p[0] * t - p[1]

    from scipy import fftpack
    data_fft = fftpack.fft(np.array(data_detrend))
#    tmp_coeff = 2/n * abs(data_fft)
    coeff = 2/n * abs(data_fft[0:int(n/2)])
#    fr = n/2 * np.linspace(0,1,int(n/2))
    f = fftpack.fftfreq(n)[0:int(n/2)]        # frequencies
#    tmp_f1 = fftpack.fftfreq(n)              # frequencies

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
    ax[0].plot(data[col])    # plot time series
    ax[0].set_ylabel('Wave hgt.')
    ax[0].set_xlabel('Date')
    ax[0].set_title('Station: 51201h; Wave height')

    ax[1].stem(f,coeff)     # plot freq domain
    ax[1].set_ylabel('Coefficient')
    ax[1].set_xlabel('Frequency')
    ax[1].set_title('FFT Decomposition')

    return 


