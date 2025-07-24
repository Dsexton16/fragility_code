import numpy as np
import shutil
import os

from mvar_model import SystemIDModel


def compute_samplepoints(winsamps, stepsamps, numtimepoints):
    # Creates a [n,2] array that holds the sample range of each window that
    # is used to index the raw data for a sliding window analysis
    samplestarts = np.arange(0, numtimepoints - winsamps + 1.0, stepsamps).astype(int)
    sampleends = np.arange(winsamps, numtimepoints + 1, stepsamps).astype(int)

    samplepoints = np.append(
        samplestarts[:, np.newaxis], sampleends[:, np.newaxis], axis=1
    )
    return samplepoints


def compute_statelds_func(eegwin, **model_params):
    mvar_model = SystemIDModel(**model_params)
    # 2: compute state transition matrix using mvar model
    mvar_model.fit(eegwin)
    A_mat = mvar_model.state_array
    return A_mat


def state_lds_array(
    arr: np.ndarray,
    winsize: int = 250,
    stepsize: int = 125,
    l2penalty: float = None,
    method_to_use: str = "pinv",
    n_jobs: int = -1,
    memmap: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """Compute A state matrix from numpy array.

    If you have a ``mne.io.Raw`` object, and you want to
    run state-lds estimation, then use the ``state_lds_derivative``
    function instead.

    Parameters
    ----------
    arr : np.ndarray
        ndarray containing EEG data; nxT where n is the number of channels and T is the number of timepoints
    sfreq : float
        Sampling frequency of the EEG
    winsize : int
        Window size to sample A matrices in samples
    stepsize : int
        Increment size, in samples, for sliding window for A matrices
    l2penalty : float
        penalty for the A matrix estimation - Adds noise when channels are too similar
    method_to_use : str
        Either 'pinv' or 'hankel'
    %(n_jobs)s
    %(verbose)s

    Returns
    -------
    A_mats : np.ndarray
        nxnxW ndarray where n is the number of channels and W is the number of windows.
        
    """
    # data array should be C x T
    n_chs, n_signals = arr.shape

    # set parameters
    model_params = {
        "l2penalty": l2penalty,
        "method_to_use": method_to_use,
    }

    # compute time and sample windows array
    sample_points = compute_samplepoints(winsize, stepsize, n_signals)
    n_wins = sample_points.shape[0]

    # initialize storage container
    A_mats = np.zeros((n_chs, n_chs, n_wins))

    for idx in range(n_wins):
        data = arr[:, sample_points[idx, 0] : sample_points[idx, 1]].T
        A_mats[..., idx] = compute_statelds_func(data, **model_params)

    return A_mats
