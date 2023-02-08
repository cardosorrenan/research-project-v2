#!/usr/bin/env python
# coding: utf-8

# In[1]:


from operator import attrgetter
from typing import List

import neurokit2 as nk
import numpy as np
import wfdb

# In[2]:


def get_records_id(path: str) -> List[str]:
    """method to get all available record ids from AFDB

    Args:
        path (str): the path of MIT-BIH AFDB files

    Returns:
        List[str]: a list of string containing all records id's
    """
    with open(f"{path}/RECORDS") as f:
        lines = f.readlines()
    return list(map(lambda line: line.strip(), lines))


def detect_r_peaks(signal_lead: np.ndarray, fs: int) -> np.ndarray:
    """method to get the R-peak indices of a signal

    Args:
        signal_lead (np.ndarray): a signal recording coming from an ecg lead
        fs (int): sampling rate

    Returns:
         np.ndarray: a numpy array containing found indices of R peaks
    """
    _, rpeaks = nk.ecg_peaks(signal_lead, sampling_rate=fs)
    return rpeaks["ECG_R_Peaks"]


def get_boundaries_array(ecg_r_peaks: np.ndarray, signal_lead_size: int) -> np.ndarray:
    """method to get boundaries array from ecg R-peaks and signal lead size

    Args:
        ecg_r_peaks (np.ndarray): numpy array containing found R-peaks indices
        signal_lead_size (int): size of the signal lead

    Returns:
        np.ndarray: numpy array containing found boundaries indices
    """
    boundaries_array = np.zeros(ecg_r_peaks.size + 2, dtype=int)
    boundaries_array[1:-1] = ecg_r_peaks
    boundaries_array[-1] = signal_lead_size
    return boundaries_array


def get_irr_intervals(signal_lead, boundaries_array):
    """method to get IRR intervals from signal lead and boundaries array

    Args:
        signal_lead (np.ndarray): a signal recording coming from an ECG lead
        boundaries_array (np.ndarray): a numpy array containing found boundaries indices

    Returns:
        List[List[float]]: a list of list of floats, each sublist is an IRR interval
    """
    n_boundaries = boundaries_array.size
    irr_intervals = []
    for i in range(n_boundaries - 1):
        irr_interval = signal_lead[boundaries_array[i] : boundaries_array[i + 1]]
        irr_intervals.append(irr_interval.tolist())
    return irr_intervals


# In[3]:


PATH_AFDB = "mit-bih-atrial-fibrillation-database-1.0.0/files"
records_id_list = get_records_id(PATH_AFDB)

records_id_list.remove("00735")
records_id_list.remove("03665")

for record_id in records_id_list:
    path = f"{PATH_AFDB}/{record_id}"
    record = wfdb.rdrecord(path)
    sample, aux_note, fs = attrgetter("sample", "aux_note", "fs")(
        wfdb.rdann(path, "atr")
    )

    leads = [0, 1]
    for lead in leads:
        signal_lead = record.p_signal[:, lead]
        signal_lead_size = signal_lead.size
        ecg_r_peaks = detect_r_peaks(signal_lead, fs)
        boundaries_array = get_boundaries_array(ecg_r_peaks, signal_lead_size)
        irr_intervals = get_irr_intervals(signal_lead, boundaries_array)

        print(len(irr_intervals))


# In[5]:


sample

