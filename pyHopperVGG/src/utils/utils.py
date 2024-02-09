import warnings
from pyHopper.utils.feature.utils import feat_space

import numpy as np
from pyHopper.utils.model import HopperModel

from pyHopper.utils.data.utils import read_hd5
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, r2_score, mean_absolute_percentage_error,
    mean_squared_error, accuracy_score, precision_score, recall_score
)


def predict_from_train_windows(train_windows, test_windows, feature_name, model_config):
    x_train, y_train = feature_extract_from_windows(train_windows, feature_name)
    x_test, y_test = feature_extract_from_windows(test_windows, feature_name)

    n_obs, n_features = x_train.shape
    class_labels = list(np.unique(y_train))
    model = HopperModel.create(
        "NN",
        model_config,
        n_features=n_features,
        class_labels=class_labels,
    )

    y_train = np.array(y_train)
    model.train(x_train, y_train)

    y_pred = model.predict(x_test)

    if len(set(y_pred)) == 1:
        warnings.warn("All predictions are the same.")

    return y_test, y_pred

#done
def feature_extract_from_windows(windows, feature_name):
    x = np.vstack([feat_space(window[0], feature_name, use_python=False) for window in windows])
    y = [x[1] for x in windows]
    return x, y


def augment_train_windows_pairwise_mixing_within_class(train_windows, factor=1.5):
    new_train_windows = []
    class_counts = {}  # Dictionary to keep track of the number of windows generated for each class
    original_train_windows = len(train_windows)  # Original number of training windows

    # Group windows by class
    class_windows = {}
    for window, class_label in train_windows:
        if class_label not in class_windows:
            class_windows[class_label] = []
        class_windows[class_label].append(window)

    # Create a tqdm progress bar for the outer loop
    for class_label, windows in tqdm(class_windows.items(), desc='Augmenting Classes', leave=False):
        # Calculate the number of windows to generate for this class
        target_windows = int(factor * len(windows))

        # Create a tqdm progress bar for the inner loop
        for _ in tqdm(range(target_windows), desc=f'Augmenting Class {class_label}', leave=False):
            i = np.random.randint(0, len(windows))
            j = np.random.randint(0, len(windows))

            window1 = windows[i]
            window2 = windows[j]

            # Randomly take the weighted sum of both window1 and window2 to create a new window
            window1_weight = np.random.uniform(0, 1)
            window2_weight = 1 - window1_weight
            new_window = window1_weight * window1 + window2_weight * window2

            new_train_windows.append((new_window, class_label))
            class_counts[class_label] = class_counts.get(class_label, 0) + 1

    return new_train_windows + train_windows

#done
def get_windows_from_paths(raw_data, index_csv_path):
    raw_data, index_csv = read_hd5(
        data_path=raw_data,
        index_csv_path=index_csv_path,
    )
    windows = []
    for _, row in index_csv.iterrows():
        start_index = row['start']
        end_index = row['end']
        window_data = raw_data[start_index:end_index + 1, :]
        windows.append((window_data, row['class']))
    target = [x[1] for x in windows]
    return windows, target


def kfold_metric_post_augmentation(windows, target, n_splits=3, random_state=42, is_reg=False):
    results = {'confusion_matrix': None, 'accuracy': None, 'precision': None, 'recall': None,
               'r2': None, 'rmse': None, 'mape': None, 'best_truth_values': None, 'best_prediction_values': None}

    metric_functions = {
        'classification': {
            'metric_list': [
                confusion_matrix,
                lambda y_true, y_pred: accuracy_score(y_true, y_pred),
                lambda y_true, y_pred: precision_score(y_true, y_pred, average='micro'),
                # specifying the average as weighted to deal with multiclass
                lambda y_true, y_pred: recall_score(y_true, y_pred, average='micro')
            ],
            'metric_names': ['confusion_matrix', 'accuracy', 'precision', 'recall'],
            'score_func': lambda y_true, y_pred: -accuracy_score(y_true, y_pred)
        },
        'regression': {
            'metric_list': [r2_score, mean_squared_error, mean_absolute_percentage_error],
            'metric_names': ['r2', 'rmse', 'mape'],
            'score_func': lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False)
        }
    }

    metric_data = metric_functions['classification' if not is_reg else 'regression']

    # Initialize StratifiedKFold
    stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_score_yet = -np.inf if is_reg else -np.inf

    # Initialize lists to store results
    original_counts_per_fold = []
    augmented_counts_per_fold = []
    accuracy_per_fold = []
    y_test_per_fold = []
    y_pred_per_fold = []

    # Perform k-fold cross-validation
    for train_index, test_index in stratified_kfold.split(windows, target):
        # train_windows, test_windows = windows[train_index], windows[test_index]
        train_windows = [windows[i] for i in train_index]
        test_windows = [windows[i] for i in test_index]
        # target_train, target_test = target[train_index], target[test_index]
        target_train = [target[i] for i in train_index]
        target_test = [target[i] for i in test_index]

        # Augment the training set
        augmented_train_windows = augment_train_windows_pairwise_mixing_within_class(train_windows, factor=10)

        # Store original and augmented counts for each fold
        original_counts = {label: sum(1 for w, l in zip(train_windows, target_train) if l == label) for label in
                           set(target)}
        augmented_counts = {label: sum(1 for w, l in zip(augmented_train_windows, target_train) if l == label) for label
                            in set(target)}

        original_counts_per_fold.append(original_counts)
        augmented_counts_per_fold.append(augmented_counts)

        # Model configuration
        model_config = {
            "strategy": None,
        }

        # Predict using the current fold
        y_test, y_pred = predict_from_train_windows(augmented_train_windows, test_windows, "C5", model_config)

        # Calculate accuracy for the current fold
        accuracy = accuracy_score(y_test, y_pred)

        # calculate the metric once, change this later
        if is_reg:
            if r2_score(y_test, y_pred) > best_score_yet:
                best_score_yet = r2_score(y_test, y_pred)
                best_test = y_test
                best_pred = y_pred
        else:
            if accuracy_score(y_test, y_pred) > best_score_yet:
                print(accuracy_score(y_test, y_pred))
                best_score_yet = accuracy_score(y_test, y_pred)
                best_test = y_test
                best_pred = y_pred

    for metric, metric_name in zip(metric_data['metric_list'], metric_data['metric_names']):
        if not is_reg:
            results[metric_name] = metric(best_test, best_pred)
        else:
            results[metric_name] = metric(best_test, best_pred)

    return results


def resub_metric_post_augmentation(windows, target, is_reg=False):
    results = {'confusion_matrix': None, 'accuracy': None, 'precision': None, 'recall': None,
               'r2': None, 'rmse': None, 'mape': None, 'best_truth_values': None, 'best_prediction_values': None}

    metric_functions = {
        'classification': {
            'metric_list': [
                confusion_matrix,
                lambda y_true, y_pred: accuracy_score(y_true, y_pred),
                lambda y_true, y_pred: precision_score(y_true, y_pred, average='micro'),
                # specifying the average as weighted to deal with multiclass
                lambda y_true, y_pred: recall_score(y_true, y_pred, average='micro')
            ],
            'metric_names': ['confusion_matrix', 'accuracy', 'precision', 'recall'],
            'score_func': lambda y_true, y_pred: -accuracy_score(y_true, y_pred)
        },
        'regression': {
            'metric_list': [r2_score, mean_squared_error, mean_absolute_percentage_error],
            'metric_names': ['r2', 'rmse', 'mape'],
            'score_func': lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False)
        }
    }

    metric_data = metric_functions['classification' if not is_reg else 'regression']

    # Initialize lists to store results
    original_counts = {label: sum(1 for w, l in zip(windows, target) if l == label) for label in set(target)}

    # Model configuration
    model_config = {
        "strategy": None,
    }

    augmented_train_windows = augment_train_windows_pairwise_mixing_within_class(windows, factor=10)
    # Predict using the current fold
    y_test, y_pred = predict_from_train_windows(augmented_train_windows, windows, "C5", model_config)

    # Calculate accuracy for the current fold
    accuracy = accuracy_score(y_test, y_pred)

    if is_reg:
        # Calculate regression metrics
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        resub_results = {'predictions': y_pred, 'r2': r2, 'mape': mape, 'rmse': rmse}
    else:
        # Calculate classification metrics
        confusion_mat = confusion_matrix(y_test, y_pred)
        # check that the confusion matrix is the same dimension as
        # Check if any label has no predicted values
        if not all(np.sum(confusion_mat, axis=1) > 0):
            warnings.warn("Some labels have no predicted values.")

        accuracy = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, average='weighted'))
        recall = float(recall_score(y_test, y_pred, average='weighted'))
        resub_results = {'predictions': y_pred, 'accuracy': accuracy, 'precision': precision, 'recall': recall,
                         'confusion_matrix': confusion_mat}

    return resub_results


def load_data_from_hd5(data_path: str, index_csv_path: str):
    """ Reads the HD5, determines the window length and reshapes
        the vector into a NxWxC matrix where N is the number of
        windows, W is the window length and C is the number of
        channels
        """
    _raw_data, _index_csv = read_hd5(data_path, index_csv_path)
    num_chans = _raw_data.shape[1]
    win_len = np.max((_index_csv.end-_index_csv.start+1).values)

    # drop len(window) not equal win_len, and align with labels
    dropped = []
    n = 0
    HData = np.empty((len(_index_csv), win_len, num_chans))
    for k in range(0, len(_index_csv)):
        start = _index_csv.start[k]  # 1-based?
        end = _index_csv.end[k]
        if (end - start + 1) == win_len:
            HData[n] = _raw_data[start-1:end]
            n += 1
        else:
            dropped += [k]

    HData   = HData[0:n]
    HLabels = np.delete(np.array(_index_csv['class']), dropped)
    HGroups = np.delete(np.array(_index_csv['group']), dropped)
    return HData, HLabels, HGroups