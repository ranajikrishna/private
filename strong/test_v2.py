
import pdb
import sys
import numpy as np

from scipy import fftpack

def main():

    s = [3,1,4,1,5,9,2,6,9]   # Some random signal(first few digits of pi in this case)
    t = np.linspace(0,8/9,9)  # Sampling at each of these equally spaced time intervals
    f = fftpack.fft(s)
    a = abs(f)          # Magnitude of each DFT component
    an = np.angle(f)    # Phase of each DFT component

    recs =  (1*a[0]/9)*np.cos(0*np.pi*t + an[0]) + \
        (2*a[1]/9)*np.cos(2*np.pi*t + an[1]) + \
        (2*a[2]/9)*np.cos(4*np.pi*t + an[2]) + \
        (2*a[3]/9)*np.cos(6*np.pi*t + an[3]) + \
        (2*a[4]/9)*np.cos(8*np.pi*t + an[4])
        #(1*a[5]/9)*np.cos(10*np.pi*t + an[5])

    pdb.set_trace()
    return 

if __name__ == '__main__':
    status = main()
    sys.exit()

