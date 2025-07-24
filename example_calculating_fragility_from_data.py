from fragility import state_perturbation_array
from mvar import state_lds_array


from mne.io import read_raw_edf

from pathlib import Path


def run_fragility(edf_fpath, model_params):
    raw = read_raw_edf(edf_fpath)

    sfreq = raw.info["sfreq"]

    winsize_sec = model_params.get("winsize_sec")
    stepsize_sec = model_params.get("stepsize_sec")
    l2penalty = model_params.get("l2penalty")

    winsize_samps = int(round(winsize_sec * sfreq))
    stepsize_samps = int(round(stepsize_sec * sfreq))

    # Do preprocessing
    data = raw.get_data()

    A_mats = state_lds_array(data, winsize=winsize_samps, stepsize=stepsize_samps, l2penalty=l2penalty)
    perturbation_mats, delta_vecs_arr = state_perturbation_array(A_mats)


if __name__ == "__main__":
    # Path to your file
    edf_fpath = Path('/Users/dsexton/Research/CoganLab/BIDS-1.0_Neighborhood_Sternberg/BIDS/derivatives/clean/sub-D0094/ieeg/sub-D0094_task-NeighborhoodSternberg_acq-01_run-01_desc-clean_ieeg.edf')

    model_params = {
        "winsize_sec": 0.5,
        "stepsize_sec": 0.5,
        "l2penalty": None
    }

    run_fragility(edf_fpath, model_params)
