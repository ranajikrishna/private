
import sys
import pdb
import numpy as np
import pandas as pd

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt 

def compute_correlation(data):

    col = ['wave_height_51201h']
    L = 500

    # ----- ACF Plots ----
    # Original Series
    fig, axes = plt.subplots(3, 2) 
    axes[0, 0].plot(data[col])
    axes[0,0].grid()
    axes[0, 0].set_title('Original Series')
    plot_acf(data[col], ax=axes[0, 1], lags=L)
                                                                                
    # 1st Differencing
    axes[1, 0].plot(data[col].diff().dropna())
    axes[1, 0].set_title('1st Order Differencing')
    plot_acf(data[col].diff().dropna(), ax=axes[1, 1], lags=L)
    axes[1,0].grid()
                                                                                
    # 2nd Differencing
    axes[2, 0].plot(data[col].diff().diff().dropna())
    axes[2, 0].set_title('2nd Order Differencing')
    plot_acf(data[col].diff().diff().dropna(),ax=axes[2, 1], lags=L)
    axes[2,0].grid()
                                                                                
    # ----- PACF Plots ----
    fig, axes = plt.subplots(3, 2)
    # Original Series
    axes[0, 0].plot(data[col])
    axes[0, 0].set_title('Original Series')
    plot_pacf(data[col], ax=axes[0, 1], lags=L)
    axes[0,0].grid()
                                                                                
    # 1st Differencing
    axes[1, 0].plot(data[col].diff().dropna())
    axes[1, 0].set_title('1st Order Differencing')
    plot_pacf(data[col].diff().dropna(), ax=axes[1, 1], lags=L)
    axes[1,0].grid()
                                                                                
    # 2nd Differencing
    axes[2, 0].plot(data[col].diff().diff().dropna())
    axes[2, 0].set_title('2nd Order Differencing')
    plot_pacf(data[col].diff().diff().dropna(),ax=axes[2, 1], lags=L)
    axes[2,0].grid()

    pdb.set_trace()

    return 
