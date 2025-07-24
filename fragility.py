from typing import Tuple, Union
import numpy as np
import os
from pathlib import Path

from perturbation_model import StructuredPerturbationModel
from mvar_model import SystemIDModel


def _get_tempfilename(x):
    """Hardcoded temporary file name for storing temporary results."""
    return "temp_{}.npz".format(x)


def _compute_fragility_func(
    shared_mvarmodel: SystemIDModel,
    shared_pertmodel: StructuredPerturbationModel,
    raw_data: np.ndarray,
    samplepoints: np.ndarray,
    tempdir: Union[str, Path],
    win: int,
    # type_indices: Dict[str, List],
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # pragma: no cover
    """Parallel computation of network.

    Computes for a single window, the A matrix and
    corresponding perturbation vector with minimum 2-norm.

    Note: The `raw_data` and `win` parameters can be replaced with `raw`
        of the MNE data structure in the future if mem-mapping is desired.

    Parameters
    ----------
    shared_mvarmodel : eztrack.network.MvarModel
        The model used to compute the A state matrix for a window of time.
    shared_pertmodel : eztrack.network.MinNormPerturbModel
        The model used to compute the vector of perturbation norms for each
        column/row perturbed in the A matrix.
    raw_data : np.ndarray
        The raw EEG data C x T (channels by time)
    samplepoints : np.ndarray
    tempdir : str | pathlib.Path
    win : int
        The specific window that `raw_data` was obtained from within
        the larger EEG snapshot.

    Returns
    -------
    pert_mat : np.ndarray
    A_mat : np.ndarray
    delta_vecs : np.ndarray
    """
    # Avoid circular import
    from eeg_prep.network.mvar_model import inner_transpose_multicompanion

    # 1: fill matrix of all channels' next EEG data over window
    win_begin = samplepoints[win, 0]
    win_end = samplepoints[win, 1] + 1
    eegwin = raw_data[:, win_begin:win_end].T  # samples x channels

    # 2: compute state transition matrix using mvar model
    shared_mvarmodel.fit(eegwin)
    A_mat = shared_mvarmodel.state_array

    # in higher-order models, one needs to apply perturbations
    # to a multi-companion matrix
    if shared_mvarmodel.order > 1 and shared_pertmodel.perturb_type == "C":
        # flip matrix
        to_perturb_A = inner_transpose_multicompanion(
            A_mat.copy().T, order=shared_mvarmodel.order
        )
    else:
        to_perturb_A = A_mat.copy()

    # 3: compute perturbation model
    pert_mat = shared_pertmodel.fit(to_perturb_A, **kwargs)
    delta_vecs = shared_pertmodel.minimum_delta_vectors

    if tempdir is not None:
        # save adjacency matrix
        tempfilename = os.path.join(tempdir, _get_tempfilename(win))
        try:
            np.savez(tempfilename, A=A_mat, pertmat=pert_mat, delta_vecs=delta_vecs)
        except BaseException as e:
            return (None, None, None)
    return pert_mat, A_mat, delta_vecs


def _compute_perturbation_func(
    A_mat: np.ndarray, radius: float, perturb_type: str, order: int, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:  # pragma: no cover
    """Parallel computation of network.

    Computes for a single window, the A matrix and
    corresponding perturbation vector with minimum 2-norm.

    Note: The `raw_data` and `win` parameters can be replaced with `raw`
        of the MNE data structure in the future if mem-mapping is desired.

    Parameters
    ----------
    radius : float
        The radius to make perturbed eigenvalue go to.
    perturb_type : str
        The type of structured perturbation to take. Either ``'R'``,
        or ``'C'``.
    A_mat : np.ndarray
        The state matrix (C x C) for x(t+1) = Ax(t).

    Returns
    -------
    pert_mat : np.ndarray
    delta_vecs : np.ndarray
    """
    # Avoid circular import
    from eeg_prep.network.mvar_model import inner_transpose_multicompanion

    pertmodel_kwargs = {
        "radius": radius,
        "perturb_type": perturb_type,
    }
    pertmodel_kwargs.update(kwargs)
    pert_model = StructuredPerturbationModel(**pertmodel_kwargs)

    # which column/row to start perturbations at
    n_perturbations = A_mat.shape[0] / order
    if order > 1:
        start_idx = int((order - 1) * n_perturbations)
    else:
        start_idx = 0

    # in higher-order models, one needs to apply perturbations
    # to a multi-companion matrix
    if order > 1 and perturb_type == "C":
        # flip matrix
        to_perturb_A = inner_transpose_multicompanion(A_mat.copy().T, order=order)
    else:
        to_perturb_A = A_mat.copy()

    # 3: compute perturbation model
    pert_mat = pert_model.fit(
        to_perturb_A, start=start_idx, n_perturbations=n_perturbations
    )
    delta_vecs = pert_model.minimum_delta_vectors

    return pert_mat, delta_vecs


def state_perturbation_array(
    arr: np.ndarray,
    order: int = 1,
    radius: float = 1.5,
    perturb_type: str = "C",
):
    """Compute network on state-lds dataset.

    This function operates on a numpy array that is assumed
    to be structured channels X channels X time.

    If you have a ``StateLDSDerivative`` object, then
    use ``state_perturbation_derivative`` function
    instead.

    Parameters
    ----------
    arr : np.ndarray
        The state matrices in the form (C x C x T)
    radius : float
        The radius to make perturbed eigenvalue go to.
    perturb_type : str
        The type of structured perturbation to take. Either ``'R'``,
        or ``'C'``.
    %(verbose)s

    Returns
    -------
    pert_mats : np.ndarray
    delta_vecs_arr : np.ndarray
    """
    if arr.shape[0] != arr.shape[1]:
        raise RuntimeError(
            f"State matrix data must be in a " f"channel X channel X time shape."
        )
    n_chs, n_chs_, n_wins = arr.shape

    model_params = {
        "order": order,
        "radius": radius,
        "perturb_type": perturb_type,
    }

    if order > 1:
        n_perturbations = int(arr.shape[0] / order)
    else:
        n_perturbations = n_chs

    # initialize numpy arrays to return results
    pert_mats = np.zeros((n_perturbations, n_wins))
    delta_vecs_arr = np.zeros((n_perturbations, n_chs, n_wins), dtype=complex)

    for idx in range(n_wins):
        pert_mat, delta_vecs = _compute_perturbation_func(arr[..., idx], **model_params)
        pert_mats[:, idx] = pert_mat
        delta_vecs_arr[..., idx] = delta_vecs

    return pert_mats, delta_vecs_arr
