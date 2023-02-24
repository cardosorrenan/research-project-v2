from typing import List

import numpy as np


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


def extract_rri_signal(ecg_r_peaks: np.ndarray, signal_lead_size: int) -> List[float]:
    """method to get IRR intervals from signal lead and founded R-peaks indices

    Args:
        ecg_r_peaks (np.ndarray): numpy array containing found R-peaks indices
        signal_lead_size (int): a size of signal recording coming from an ECG lead

    Returns:
        List[float]: a list of floats, is the RRIs extracted from signal
    """
    rri_signal = []
    for i in range(ecg_r_peaks.size - 1):
        rri_beat = int(ecg_r_peaks[i + 1] - ecg_r_peaks[i])
        rri_signal.append(rri_beat)
    return rri_signal


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


def resample_ms(rri_signal, freq) -> List[float]:
    MILISECONDS = 1000
    return [(MILISECONDS / freq) * rri for rri in rri_signal]
