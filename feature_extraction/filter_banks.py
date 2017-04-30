from __future__ import division
from math import log
import numpy as np

class MelFilterBank(object):
    def __init__(self, samplerate, num_linear_bins, num_bands, min_freq=0, max_freq=None, normalise=True):
        nyquist_freq = samplerate / 2
        if not max_freq or max_freq > nyquist_freq:
            max_freq = nyquist_freq
        if min_freq > nyquist_freq:
            min_freq = nyquist_freq
        
        max_mel_freq = 1000 * log(1 + max_freq / 700) / log(1 + 1000 / 700)
        min_mel_freq = 1000 * log(1 + min_freq / 700) / log(1 + 1000 / 700)
        mel_centre_freqs = np.linspace(min_mel_freq, max_mel_freq, num_bands+2)
        mel_centre_indices = np.floor(0.5 + 700 * num_linear_bins * (np.exp(mel_centre_freqs * log(1 + 1000 / 700) / 1000) - 1) / nyquist_freq).astype(int)
        
        self.filterbank = np.zeros((num_bands, num_linear_bins))
        for band_idx, filter in enumerate(self.filterbank):
            start, centre, end = mel_centre_indices[band_idx:band_idx+3]
            filter[start:centre] = np.linspace(0, 1, centre-start, False)
            filter[centre:end] = np.linspace(1, 0, end-centre, False)
        if normalise:
            norm = self.filterbank.sum(axis=1, keepdims=True)
            norm[norm == 0] = 1
            self.filterbank /= norm

    def apply(self, linear_spectrum):
        return np.dot(self.filterbank, linear_spectrum)
