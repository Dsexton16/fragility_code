import numpy as np
from scipy import signal


def compute_source_sink_index(A_hat, normalize=True):
    N = A_hat.shape[1]

    # Take absolute value and set diagonal to zero
    A_abs = np.abs(A_hat)
    for ii in range(N):
        A_abs[ii, ii] = 0

    # Compute row rank
    row_sums = np.sum(A_abs, axis=1)
    row_order = np.argsort(row_sums)
    row_rank = (np.argsort(row_order) + 1) / N

    # Compute column rank
    col_sums = np.sum(A_abs, axis=0)
    col_order = np.argsort(col_sums)
    col_rank = (np.argsort(col_order) + 1) / N
    
    # Compute sink, source indices
    
    #sink_idx = np.sqrt(2) - np.sqrt((row_rank - 1) ** 2 + (col_rank - 1/N)**2)
    sink_idx = np.sqrt(2) - np.sqrt((row_rank - 1) ** 2 + (col_rank - 1/N)**2)
    source_idx = np.sqrt(2) - np.sqrt((row_rank - 1/N) ** 2 + (col_rank - 1)**2)
    
    # Compute source influence
    source_influence = np.matmul(np.abs(A_hat), source_idx)
    source_influence /= np.max(source_influence)
    
    # Compute sink connectivity
    sink_connectivity = np.matmul(np.abs(A_hat), sink_idx)
    sink_connectivity /= np.max(sink_connectivity)
    
    # Normalize
    if normalize:
        source_idx /= np.sqrt(2)
        sink_idx /= np.sqrt(2)
    
    # Compute SSI
    ssi = sink_idx * source_influence * sink_connectivity
    
    return sink_idx, source_idx, source_influence, sink_connectivity, ssi


def computeA(data, alpha=0):
    """Compute the A transition matrix from a vector timeseries
    """
    
    nchns, T = data.shape
    
    Z = data[:, 0:T-1]
    Y = data[:, 1:T]
    D = np.linalg.inv(np.matmul(Z, Z.transpose()) + alpha*np.eye(nchns))
    Ahat = np.matmul(Y, np.matmul(Z.transpose(), D))
    return Ahat


def computeR1(data, alpha=0):
    """Compute the A transition matrix from a vector timeseries
    """
    
    nchns, T = data.shape
    
    Z = data[:, 0:T-1]
    Y = data[:, 1:T]
    R1 = np.matmul(Y, Z.transpose()) / (T - 1)
    return R1


def get_spectral_entropy(feature):
    ss = []
    for cc in range(18):
        f, t, Sxx = signal.spectrogram(feature[:, cc], nperseg=200, noverlap=100)
        Pxx = np.power(np.abs(Sxx[1:, :]), 2)
        se = -np.sum(np.log2(Pxx) * Pxx, axis=0)
        ss.append(se)
    return np.asarray(ss).transpose()


def spectral_entropy(data):
    D = np.fft.fft(data.transpose())
    D = np.abs(D[:, 1:])
    D = D / np.sum(D, axis=1)[:, np.newaxis]
    return np.sum(-D * np.log(D), axis=1)
