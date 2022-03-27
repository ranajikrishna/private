
from imblearn.under_sampling import RandomUnderSampler

import sys
import pdb


def under_sample(X,y,alpha):
    # define undersample strategy
    undersample = RandomUnderSampler(sampling_strategy=alpha)
    X_under, y_under = undersample.fit_resample(X, y)
    return X_under, y_under

