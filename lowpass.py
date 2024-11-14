import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt


# Set filter parameters
def butter_lowpass(cutoff, fs, order=1):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=1):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Filter parameters
order = 2
fs = 100  # Sampling frequency (replace with your actual sampling frequency)
cutoff_frequency = 10  # Cutoff frequency, can be adjusted as needed
