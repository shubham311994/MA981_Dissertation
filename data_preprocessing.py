import mne
import numpy as np
import pandas as pd
from mne_icalabel import label_components
from mne import make_fixed_length_epochs
from typing import Tuple


def read_file(path: str) -> mne.io.eeglab.eeglab.RawEEGLAB:
    """
    Read the EEG data from the given path of fdt and set.
    :param: path: path of the data file in string format.
    :return: raw object of type mne.io.eeglab.eeglab.RawEEGLAB
    """
    raw = mne.io.read_raw_eeglab(path, preload=True)

    return raw


def extract_data_categories(raw: mne.io.eeglab.eeglab.RawEEGLAB) -> Tuple[
    mne.io.eeglab.eeglab.RawEEGLAB, mne.io.eeglab.eeglab.RawEEGLAB,
    mne.io.eeglab.eeglab.RawEEGLAB, mne.io.eeglab.eeglab.RawEEGLAB]:
    """
    Extract the data of different categories from the raw file and return the segregated raw data files.
    :param: raw:
    :return: eyes_open, eyes_close, warm_feel, hot_feel, noise_data
    """
    for each in range(0, 101):

        if raw._annotations[each]['description'] == '5':
            eyes_open = raw.copy().crop(tmin=raw._annotations[each]['onset'],
                                        tmax=raw._annotations[each]['onset'] + 150)
        if raw._annotations[each]['description'] == '4':
            eyes_close = raw.copy().crop(tmin=raw._annotations[each]['onset'],
                                         tmax=raw._annotations[each]['onset'] + 150)
        if raw._annotations[each]['description'] == '41':
            warm_feel = raw.copy().crop(tmin=raw._annotations[each]['onset'],
                                        tmax=raw._annotations[each]['onset'] + 300)
        if raw._annotations[each]['description'] == '11':
            hot_feel = raw.copy().crop(tmin=raw._annotations[each]['onset'],
                                       tmax=raw._annotations[each]['onset'] + 300)
        if raw._annotations[each]['description'] == '71':
            noise_data = raw.copy().crop(tmin=raw._annotations[each]['onset'],
                                         tmax=raw._annotations[each]['onset'] + 300)

        return eyes_open, eyes_close, warm_feel, hot_feel, noise_data


def filter_data(raw: mne.io.eeglab.eeglab.RawEEGLAB) -> mne.io.eeglab.eeglab.RawEEGLAB:
    """
    Use the band pass filter to filter out the noise and unwanted frequency data from the actual dataset.
    :param: raw: raw object of the data which needs to be filtered.
    :return: raw object with filtered data.
    """
    filtered_data = mne.io.Raw.filter(raw, l_freq=0.5, h_freq=100)

    return filtered_data


def standardize_raw_data(raw: mne.io.eeglab.eeglab.RawEEGLAB) -> mne.io.eeglab.eeglab.RawEEGLAB:
    """
    Use the standardization method to standardise the raw signals with z-score method.
    :param: raw: raw object of the data to be standardized.
    :return: raw object with standardized values.
    """
    raw = raw.apply_function(lambda x: (x - np.mean(x) / np.std(x)))

    return raw


def get_ica_labels(raw: mne.io.eeglab.eeglab.RawEEGLAB, ica: mne.preprocessing.ica.ICA) -> dict:
    """
    USe the raw data to get the labels from the ica obj
    :param: raw: get the labels of ICA fit raw object
    :param: ica: ica object of the fitted raw object.
    :return: dictionary containing the labels for ica components
    """
    ic_labels = label_components(raw, ica, method="iclabel")

    return ic_labels


def fit_ica(raw: mne.io.eeglab.eeglab.RawEEGLAB) -> mne.preprocessing.ica.ICA:
    """
    Fit the ica on the raw object to obtain the ica components.
    :param: raw: raw object containing the data.
    :return: ica object after it is fitted on the raw data.
    """
    raw = filter_data(raw=raw)
    ica = mne.preprocessing.ICA(n_components=0.99, random_state=42)
    ica.fit(raw)

    return ica


def get_label_index_to_exclude(ica_labels: dict) -> list:
    """
    Find the ICA components indexes which are to be excluded during artifact repair.
    :param: ica_labels:
    :return: exclude_index list
    """
    exclude_index = []
    for i, each in enumerate(ica_labels['labels']):
        if (each == 'eye blink') or (each.endswith('noise')):
            exclude_index.append(i)

    return exclude_index


def repair_ica_artifact(raw: mne.io.eeglab.eeglab.RawEEGLAB) -> mne.io.eeglab.eeglab.RawEEGLAB:
    """
    Repair the raw object artifact using the ica
    :param: raw: raw object to be repaired
    :return: repaired raw object with removed noises.
    """
    ica = fit_ica(raw=raw)
    ic_labels = get_ica_labels(raw=raw, ica=ica)
    exclude_index = get_label_index_to_exclude(ica_labels=ic_labels)
    raw = ica.apply(raw, exclude=exclude_index)

    return raw


def convert_epoch_to_frequency_vs_time(epoch_object: mne.epochs.Epochs) -> np.ndarray:
    """
    Convert the epochs into times vs frequency using psd_welch method.
    :param epoch_object:
    :return: numpy array of 3 dimensions with (n_epochs, n_channels, n_freq)
    """

    epoch_array, _frequencies = mne.time_frequency.psd_welch(epoch_object, fmin=0.5, fmax=50, average='median')

    return epoch_array


def create_dataframe_from_epoch(epoch_data: np.ndarray, label_value: int) -> pd.DataFrame:
    """
    Convert the numpy data into dataframe and add target label to it.
    :param: epoch_data: numpy array obtained from the epoch conversion of raw data.
    :param: label_value: label value to be used for the particular type of data
    :return: pandas dataframe containing the feature and target value.
    """

    data_set = pd.DataFrame(epoch_data.reshape(epoch_data.shape[0], -1))
    data_set['target'] = np.full(data_set.shape[0], label_value, dtype='int')

    return data_set


def create_epoch_data(raw: mne.io.eeglab.eeglab.RawEEGLAB, duration: int) -> np.ndarray:
    """
    Get the epoch of the raw object using the fixed length epoch method.
    :param: raw: raw object containing the data.
    :param: duration: duration of the epoch to be used.
    :return: 3 dimensional array containing the data of the raw object.
    """
    epoch_data = make_fixed_length_epochs(raw, duration=duration, preload=True, overlap=float(duration / 2))
    epoch_frequency_data = convert_epoch_to_frequency_vs_time(epoch_data)

    return epoch_frequency_data


def combine_dataframes(dataframe1, dataframe2):
    """
    Merge the two dataframes into one
    :param: dataframe1: dataframe of one type with label.
    :param: dataframe2: dataframe of another type with label.
    """
    merged_dataframe = pd.concat([dataframe1, dataframe2])

    return merged_dataframe


def calculate_duration(raw: mne.io.eeglab.eeglab.RawEEGLAB) -> float:
    """
    Calculate the duration of the raw object reading.
    :param: raw object
    """
    scan_duration = raw._data.shape[1] / raw.info['sfreq']

    return scan_duration
