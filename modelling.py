from pathlib import Path
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from data_preprocessing import (extract_data_categories, repair_ica_artifact, create_epoch_data,
                                create_dataframe_from_epoch, read_file, combine_dataframes, calculate_duration)
from helper_methods import check_file_exists, get_file_list_from_path, apply_stratified_cv
from exceptions import SizeNotMatched

from lightgbm import LGBMClassifier

metrics_list_eo_ec = []
metrics_list_wf_hf = []
metrics_list_wf_sf = []
metrics_list_hf_sf = []

lg_clf = LGBMClassifier(device="gpu", colsample_bytree=0.6798358579930577,
                        learning_rate=0.00919045768056308, n_estimators=600,
                        num_leaves=10, objective='binary')
lda_clf = LinearDiscriminantAnalysis(shrinkage=0.21797193476378346, solver='lsqr')
knn_clf = KNeighborsClassifier(metric='manhattan', n_neighbors=8)

PATH = '/home/sc21703/dissertation_project/'


def warm_vs_hot(warm_dataframe, hot_dataframe, participant_number: int):
    """
    Compare warm vs hot condition by fitting machine learning algorithm and computing accuracy and other measures for
    each subject.
    :return: results of each subject in dictionary
    """
    result_dict = {'participant_id': participant_number, 'lightgbm_accuracy': [], 'lda_accuracy': [],
                   'knn_accuracy': [], 'lightgbm_f1_score': [], 'lda_f1_score': [], 'knn_f1_score': [],
                   'lightgbm_sensitivity': [], 'lda_sensitivity': [], 'knn_sensitivity': [], 'lightgbm_specificity': [],
                   'lda_specificity': [], 'knn_specificity': []}
    data = combine_dataframes(warm_dataframe, hot_dataframe)
    x = data.drop(columns=['target']).values
    y = data['target'].values
    for each in [lg_clf, lda_clf, knn_clf]:
        accuracy_list, f1_score_list, sensitivity_list, specificity_list = apply_stratified_cv(each, x=x, y=y)
        if type(each).__name__ == 'LGBMClassifier':
            result_dict['lightgbm_accuracy'] = accuracy_list
            result_dict['lightgbm_f1_score'] = f1_score_list
            result_dict['lightgbm_sensitivity'] = sensitivity_list
            result_dict['lightgbm_specificity'] = specificity_list
        if type(each).__name__ == 'LinearDiscriminantAnalysis':
            result_dict['lda_accuracy'] = accuracy_list
            result_dict['lda_f1_score'] = f1_score_list
            result_dict['lda_sensitivity'] = sensitivity_list
            result_dict['lda_specificity'] = specificity_list
        if type(each).__name__ == 'KNeighborsClassifier':
            result_dict['knn_accuracy'] = accuracy_list
            result_dict['knn_f1_score'] = f1_score_list
            result_dict['knn_sensitivity'] = sensitivity_list
            result_dict['knn_specificity'] = specificity_list
    return result_dict


def warm_vs_sound(warm_dataframe, sound_dataframe, participant_number: int):
    """
    Compare warm vs sound pain by fitting machine learning algorithm and computing accuracy and other measures for each
    subject.
    :return: results of each subject in dictionary
    """
    result_dict = {'participant_id': participant_number, 'lightgbm_accuracy': [], 'lda_accuracy': [],
                   'knn_accuracy': [], 'lightgbm_f1_score': [], 'lda_f1_score': [], 'knn_f1_score': [],
                   'lightgbm_sensitivity': [], 'lda_sensitivity': [], 'knn_sensitivity': [], 'lightgbm_specificity': [],
                   'lda_specificity': [], 'knn_specificity': []}
    data = combine_dataframes(warm_dataframe, sound_dataframe)
    x = data.drop(columns=['target']).values
    y = data['target'].values
    for each in [lg_clf, lda_clf, knn_clf]:
        accuracy_list, f1_score_list, sensitivity_list, specificity_list = apply_stratified_cv(each, x=x, y=y)
        if type(each).__name__ == 'LGBMClassifier':
            result_dict['lightgbm_accuracy'] = accuracy_list
            result_dict['lightgbm_f1_score'] = f1_score_list
            result_dict['lightgbm_sensitivity'] = sensitivity_list
            result_dict['lightgbm_specificity'] = specificity_list
        if type(each).__name__ == 'LinearDiscriminantAnalysis':
            result_dict['lda_accuracy'] = accuracy_list
            result_dict['lda_f1_score'] = f1_score_list
            result_dict['lda_sensitivity'] = sensitivity_list
            result_dict['lda_specificity'] = specificity_list
        if type(each).__name__ == 'KNeighborsClassifier':
            result_dict['knn_accuracy'] = accuracy_list
            result_dict['knn_f1_score'] = f1_score_list
            result_dict['knn_sensitivity'] = sensitivity_list
            result_dict['knn_specificity'] = specificity_list

    return result_dict


def hot_vs_sound(hot_dataframe, sound_dataframe, participant_number: int):
    """
    Compare hot vs sound condition by fitting machine learning algorithm and computing accuracy and other measures for
    each subject.
    :return: results of each subject in dictionary
    """
    result_dict = {'participant_id': participant_number, 'lightgbm_accuracy': [], 'lda_accuracy': [],
                   'knn_accuracy': [], 'lightgbm_f1_score': [], 'lda_f1_score': [], 'knn_f1_score': [],
                   'lightgbm_sensitivity': [], 'lda_sensitivity': [], 'knn_sensitivity': [], 'lightgbm_specificity': [],
                   'lda_specificity': [], 'knn_specificity': []}
    data = combine_dataframes(hot_dataframe, sound_dataframe)
    x = data.drop(columns=['target']).values
    y = data['target'].values
    for each in [lg_clf, lda_clf, knn_clf]:
        accuracy_list, f1_score_list, sensitivity_list, specificity_list = apply_stratified_cv(each, x=x, y=y)
        if type(each).__name__ == 'LGBMClassifier':
            result_dict['lightgbm_accuracy'] = accuracy_list
            result_dict['lightgbm_f1_score'] = f1_score_list
            result_dict['lightgbm_sensitivity'] = sensitivity_list
            result_dict['lightgbm_specificity'] = specificity_list
        if type(each).__name__ == 'LinearDiscriminantAnalysis':
            result_dict['lda_accuracy'] = accuracy_list
            result_dict['lda_f1_score'] = f1_score_list
            result_dict['lda_sensitivity'] = sensitivity_list
            result_dict['lda_specificity'] = specificity_list
        if type(each).__name__ == 'KNeighborsClassifier':
            result_dict['knn_accuracy'] = accuracy_list
            result_dict['knn_f1_score'] = f1_score_list
            result_dict['knn_sensitivity'] = sensitivity_list
            result_dict['knn_specificity'] = specificity_list

    return result_dict


def eyes_open_vs_eyes_close(eyes_open_dataframe, eyes_close_dataframe,
                            participant_number: int):
    """
    Compare eyes open vs eyes_close condition by fitting machine learning algorithm and computing accuracy and other
    measures for each subject.
    :return: results of each subject in dictionary
    """
    result_dict = {'participant_id': participant_number, 'lightgbm_accuracy': [], 'lda_accuracy': [],
                   'knn_accuracy': [], 'lightgbm_f1_score': [], 'lda_f1_score': [], 'knn_f1_score': [],
                   'lightgbm_sensitivity': [], 'lda_sensitivity': [], 'knn_sensitivity': [], 'lightgbm_specificity': [],
                   'lda_specificity': [], 'knn_specificity': []}
    data = combine_dataframes(eyes_open_dataframe, eyes_close_dataframe)
    x = data.drop(columns=['target']).values
    y = data['target'].values
    for each in [lg_clf, lda_clf, knn_clf]:
        accuracy_list, f1_score_list, sensitivity_list, specificity_list = apply_stratified_cv(each, x=x, y=y)
        if type(each).__name__ == 'LGBMClassifier':
            result_dict['lightgbm_accuracy'] = accuracy_list
            result_dict['lightgbm_f1_score'] = f1_score_list
            result_dict['lightgbm_sensitivity'] = sensitivity_list
            result_dict['lightgbm_specificity'] = specificity_list
        if type(each).__name__ == 'LinearDiscriminantAnalysis':
            result_dict['lda_accuracy'] = accuracy_list
            result_dict['lda_f1_score'] = f1_score_list
            result_dict['lda_sensitivity'] = sensitivity_list
            result_dict['lda_specificity'] = specificity_list
        if type(each).__name__ == 'KNeighborsClassifier':
            result_dict['knn_accuracy'] = accuracy_list
            result_dict['knn_f1_score'] = f1_score_list
            result_dict['knn_sensitivity'] = sensitivity_list
            result_dict['knn_specificity'] = specificity_list

    return result_dict


def main(file_path: Path):
    """
    Run all the preprocessing and epoch conversion. Convert the epochs to dataframe and then use it for Machine Learning
    :param: file_path: Path of the files where the data is stored.
    """
    list_of_fdt_file_path, list_of_set_file_path = get_file_list_from_path(file_path=file_path)
    participant = 0
    if list_of_fdt_file_path and list_of_set_file_path:
        try:
            if len(list_of_fdt_file_path) == len(list_of_set_file_path):
                for fdt_file, set_file in zip(list_of_fdt_file_path, list_of_set_file_path):
                    if (check_file_exists(fdt_file)) & (check_file_exists(set_file)):
                        raw = read_file(set_file)
                        if calculate_duration(raw) < 2200:
                            continue
                        else:
                            repaired_raw = repair_ica_artifact(raw=raw)
                            eyes_open_raw, eyes_close_raw, warm_feel_raw, hot_feel_raw, sound_feel_raw = \
                                extract_data_categories(repaired_raw)
                            eyes_open_epoch = create_epoch_data(raw=eyes_open_raw, duration=1)
                            eyes_close_epoch = create_epoch_data(raw=eyes_close_raw, duration=1)
                            warm_feel_epoch = create_epoch_data(raw=warm_feel_raw, duration=1)
                            hot_feel_epoch = create_epoch_data(raw=hot_feel_raw, duration=1)
                            sound_feel_epoch = create_epoch_data(raw=sound_feel_raw, duration=1)
                            metrics_list_wf_hf.append(
                                warm_vs_hot(create_dataframe_from_epoch(warm_feel_epoch, label_value=0),
                                            create_dataframe_from_epoch(hot_feel_epoch, label_value=1),
                                            participant_number=participant))
                            metrics_list_wf_sf.append(
                                warm_vs_sound(create_dataframe_from_epoch(warm_feel_epoch, label_value=0),
                                              create_dataframe_from_epoch(sound_feel_epoch, label_value=1),
                                              participant_number=participant))
                            metrics_list_hf_sf.append(
                                hot_vs_sound(create_dataframe_from_epoch(hot_feel_epoch, label_value=0),
                                             create_dataframe_from_epoch(sound_feel_epoch, label_value=1),
                                             participant_number=participant))
                            metrics_list_eo_ec.append(
                                eyes_open_vs_eyes_close(create_dataframe_from_epoch(eyes_open_epoch, label_value=0),
                                                        create_dataframe_from_epoch(eyes_close_epoch, label_value=1),
                                                        participant_number=participant))
                            print(f'Participant #: {participant + 1}')
                            participant += 1
                            accuracy_eo_ec = pd.DataFrame(metrics_list_eo_ec)
                            accuracy_eo_ec.to_csv(Path(f'{PATH}/eo_ec/{participant}_eo_ec.csv'))
                            accuracy_hf_sf = pd.DataFrame(metrics_list_hf_sf)
                            accuracy_hf_sf.to_csv(Path(f'{PATH}/hf_sf/{participant}_accuracy_hf_sf.csv'))
                            accuracy_wf_hf = pd.DataFrame(metrics_list_wf_hf)
                            accuracy_wf_hf.to_csv(Path(f'{PATH}/hf_wf/{participant}_accuracy_wf_hf.csv'))
                            accuracy_wf_sf = pd.DataFrame(metrics_list_wf_sf)
                            accuracy_wf_sf.to_csv(Path(f'{PATH}/wf_sf/{participant}accuracy_wf_sf.csv'))
                            del raw, repaired_raw, eyes_open_raw, eyes_close_raw, warm_feel_raw, hot_feel_raw, \
                                sound_feel_raw
            else:
                raise SizeNotMatched({
                    f"The number of FDT and SET files is not same fdt is {len(list_of_fdt_file_path)} \
                     != set is {len(list_of_set_file_path)}"})
        except FileNotFoundError as fe:
            fe.strerror = f"Either fdt_file or set_file doesn't exist"
            raise fe


if __name__ == '__main__':
    main(Path('/home/sc21703/dissertation_project/faulty/'))
