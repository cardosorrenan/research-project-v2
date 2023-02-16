#!/usr/bin/env python
# coding: utf-8

# In[51]:


import os
from operator import attrgetter
from typing import List

import neurokit2 as nk
import numpy as np
import wfdb


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


def get_intervals_afib(
    sample: List[int], aux_note: List[str], signal_len: int
) -> List[List[int]]:
    """
    Get the intervals of atrial fibrillation (AFIB) from a list of sample values and corresponding annotations.

    Args:
    - sample (List[int]): A list of ECG sample values.
    - aux_note (List[str]): A list of annotation labels for each sample.

    Returns:
    - afib_intervals (List[List[int]]): A list of start and end indices for each interval of AFIB.
    """
    afib_intervals = []
    for i, label in enumerate(aux_note):
        if label == "(AFIB":
            afib_start = sample[i]
            last_notation = len(sample) == (i + 1)
            afib_end = signal_len if last_notation else sample[i + 1] - 1
            afib_intervals.append([afib_start, afib_end])
    return afib_intervals


def get_values_within_intervals(
    intervals: List[List[int]], signal: List[float]
) -> List[float]:
    """
    Extract values from a signal within the specified intervals.

    Args:
    - intervals (List[List[int]]): A list of start and end indices for each interval of interest.
    - signal (List[float]): A list of values representing a signal.

    Returns:
    - values_within_intervals (List[float]): A list of values within the specified intervals.
    """
    values_within_intervals = []
    for interval in intervals:
        start_index, end_index = interval
        if start_index < 0 or end_index > len(signal) or start_index > end_index:
            # Skip invalid intervals
            continue
        values_within_intervals.extend(signal[start_index : end_index + 1])
    return values_within_intervals


def extract_rri_signal(ecg_r_peaks: np.ndarray, signal_lead_size: int) -> List[float]:
    """method to get IRR intervals from signal lead and founded R-peaks indices

    Args:
        ecg_r_peaks (np.ndarray): numpy array containing found R-peaks indices
        signal_lead_size (int): a size of signal recording coming from an ECG lead

    Returns:
        List[float]: a list of floats, is the RRIs extracted from signal
    """
    delimiters = np.zeros(ecg_r_peaks.size + 2, dtype=int)
    delimiters[1:-1] = ecg_r_peaks
    delimiters[-1] = signal_lead_size
    rri_signal = []
    for i in range(delimiters.size - 1):
        rri_beat = int(delimiters[i + 1] - delimiters[i])
        rri_signal.append(rri_beat)
    return rri_signal


def resample_ms(rri_signal, freq) -> List[float]:
    MILISECONDS = 1000
    return [(MILISECONDS / freq) * rri for rri in rri_signal]


def extract_afib():
    AFDB_PATH = "mit-bih-atrial-fibrillation-database-1.0.0/files"

    # Get list of record IDs
    record_ids = get_records_id(AFDB_PATH)

    # Remove problematic records
    record_ids.remove("00735")
    record_ids.remove("03665")

    RRI = []
    # Process each record
    for record_id in record_ids:
        record_path = f"{AFDB_PATH}/{record_id}"

        # Load ECG signal and annotation data
        ecg_signal, ecg_metadata = wfdb.rdsamp(record_path)
        signal_len = ecg_metadata["sig_len"]

        # Extract AFIB intervals from annotation data
        sample, aux_note = attrgetter("sample", "aux_note")(
            wfdb.rdann(record_path, "atr")
        )
        afib_intervals = get_intervals_afib(sample, aux_note, signal_len)

        # Process each lead in the ECG signal
        for lead_idx, lead_signal in enumerate(ecg_signal.T):
            # Segment signal into AFIB intervals and detect R-peaks
            afib_signal = get_values_within_intervals(afib_intervals, lead_signal)
            _, rpeaks = nk.ecg_peaks(afib_signal, sampling_rate=ecg_metadata["fs"])
            ecg_rpeaks = rpeaks["ECG_R_Peaks"]

            # Compute RRI signal and resample to 1 ms resolution
            rri_signal = extract_rri_signal(ecg_rpeaks, signal_len)
            rri_signal_ms = resample_ms(rri_signal, ecg_metadata["fs"])
            RRI.append(rri_signal_ms)

    directory = "./RRI"
    if not os.path.exists(directory):
        os.makedirs(directory)

    np.save(f"{directory}/afib_output.npy", RRI)
