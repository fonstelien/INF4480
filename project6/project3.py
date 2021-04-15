import numpy as np
from scipy.io import loadmat
from scipy.signal import correlate, resample
from scipy.fft import fft
import matplotlib.pyplot as plt

def sft(x, zero_pad=0):
    '''Implements the "Slow Fourier Transform". Discrete Fourier Transform for x; pads with a factor zero_pad.'''
    N = x.shape[0]

    # Apply zero padding
    if zero_pad:
        x = np.hstack((x, np.zeros(zero_pad*N)))
        N = x.shape[0]

    # Calculate DFT
#    n = np.arange(N)
#    k = n.reshape(-1,1)
#    e = np.exp(-1j*2*np.pi/N*k*n)
#    x_dft = (x*e).sum(axis=1)

    x_dft = fft(x)

    # Rearrange around center frequency
    x_dft_left, x_dft_right = x_dft[N//2:], x_dft[:N//2]
    x_dft = np.hstack((x_dft_left, x_dft_right))

    return x_dft

def window(N, n0=0, wN=None, wname='rectangular'):
    '''Returns a window sequence starting at n0, with window length N, and zero padding up to wN; Choose wname={'rectangular', 'hamming', 'hanning'}.'''
    assert wname in ['rectangular', 'hamming', 'hanning']
    if not wN:
        wN = N + n0

    w = np.zeros(wN)
    if wname == 'rectangular':
        w[n0:n0+N] = 1
    if wname == 'hamming':
        w[n0:n0+N] = .54 - .46*np.cos(2*np.pi*np.arange(N)/(N-1))
    if wname == 'hanning':
        w[n0:n0+N] = .5 - .5*np.cos(2*np.pi*np.arange(N)/(N-1))

    return w

def periodogram(x, normalize=False, wname=None, **kwargs):
    '''Estimates the Power Spectrum Density with the Periodogram method. Optional arguments are window wname={'rectangular', 'hamming', 'hanning'}.'''
    N = x.shape[0]

    # Apply window
    if wname is not None:
        w = window(N, wname=wname)
        x = x*w

    # Calculate Periodogram
    x = sft(x, **kwargs)
    x = np.abs(x)**2
    Pxx = x/N

    # Normalize around the mean of Pxx
    if normalize:
        Pxx = Pxx/Pxx.mean()

    return Pxx

def welch(x, L=None, D=None, normalize=False, **kwargs):
    '''Estimates the Power Spectrum Density with Welch's method. Periodogram length L, offset by D.'''
    N = x.shape[0]
    Pxx = list()

    # Set up the ranges for each periodogram
    if L is None:
        L = N//8
    if D is None:
        D = L//2
    K = (N-L)//D + 1

    # Calculate periodograms
    for i in range(K):
        pxx = periodogram(x[i*D:i*D+L], **kwargs)
        Pxx.append(pxx)
    Pxx = np.concatenate(Pxx).reshape(-1, pxx.shape[0])

    # Estimate PSD
    Pxx = Pxx.mean(axis=0)

    # Normalize around the mean of Pxx
    if normalize:
        Pxx = Pxx/Pxx.mean()

    return Pxx

def multitaper(x, K=8, normalize=False, **kwargs):
    '''Estimates the Power Spectrum Density with the Multi-Taper method, using "poor-man's" tapers.'''
    N = x.shape[0]
    Pxx = list()

    # Calculate the poor-man tapers
    n = np.arange(N)
    k = np.arange(K).reshape(-1,1)
    W = np.sqrt(2/(N+1))*np.sin(np.pi*(k+1)*(n+1)/(N+1))

    # Calculate periodograms
    for i in range(K):
        w = W[i,:]
        pxx = periodogram(x*w, **kwargs)
        Pxx.append(pxx)
    Pxx = np.concatenate(Pxx).reshape(-1, pxx.shape[0])

    # Estimate PSD
    Pxx = Pxx.mean(axis=0)

    # Normalize around the mean of Pxx
    if normalize:
        Pxx = Pxx/Pxx.mean()

    return Pxx

def spectrogram(x, L, D=None, normalize=False, fperiodogram=periodogram, fargs={}):
    '''Calculates the Spectrogram of x with estimates of its Power Spectrum Density in sequence lengths L, offset by D. The PSD is estimated using fperiodogram={periodogram, welch, multitaper} applied with arguments fargs.'''
    N = x.shape[0]
    Pxx = list()

    # Set shape of the Spectrogram
    if D is None:
        D = L
    K = (N-L)//D + 1

    # Calculate periodograms
    for i in range(K):
        pxx = fperiodogram(x[i*D:i*D+L], **fargs)
        Pxx.append(pxx)
    Pxx = np.concatenate(Pxx).reshape(-1, pxx.shape[0])

    # Normalize around the mean of Pxx
    if normalize:
        Pxx = Pxx/Pxx.mean(axis=1, keepdims=True)

    return Pxx
