import hashlib
import pickle
import os
import shutil
from xml.etree.ElementInclude import include

import pandas as pd
from google.cloud.speech_v2.types.cloud_speech import AutoDetectDecodingConfig
import json
import urllib
import numpy as np
from torch import Tensor
from sklearn import metrics as sk_metrics

def hash_from_dict(config: dict, hash_len=None):
    """
    Create hexadecimal hash of length len from a dictionary d
    Can be used to create directory for storing temporary results etc.
    """
    hash = hashlib.sha1(bytes(pickle.dumps(config))).hexdigest()
    if hash_len is not None:
        assert 0 < hash_len < len(hash)
        hash = hash[:hash_len]
    return hash

def hash_list(data: list, hash_len=None):
    dict_from_list = {'data': data}
    return hash_from_dict(dict_from_list, hash_len=hash_len)

def create_directory(path, empty_dir = False):
    """ Creates directory if it doesnt exist yet, optionally deleting all files in there """
    if not os.path.exists(path):
        os.makedirs(path)

    if empty_dir:
        shutil.rmtree(path)
        os.makedirs(path)

    try:
        os.chmod(path, 0o777)
    except PermissionError:
        pass


def python_to_json(d):
    # dump python data to json with some special handling
    # this is used to write to easily handleable text files
    def process(data):
        if isinstance(data, dict):
            return {key: process(data[key]) for key in data}  # recursive
        if isinstance(data, list):
            return [process(list_item) for list_item in data]  # recursive
        if isinstance(data, tuple):
            return tuple([process(list_item) for list_item in data])  # recursive
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy().tolist()
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (pd.DataFrame, pd.Series)):
            return process(data.to_dict())
        elif isinstance(data, AutoDetectDecodingConfig):
            return str(data)
        elif isinstance(data, (np.float32, np.float64, np.float_)):
            return float(data)
        elif isinstance(data, (np.int32, np.int64, np.int_)):
            return int(data)
        else:
            return data

    return json.dumps(process(d))

def dataset_name_to_url_part(name: str):
    """
    Make database name good for part of a url (e.g. directory name)
    """
    return urllib.parse.quote(name.replace(" ", "_").replace("(", "").replace(")", ""))



def store_obj_to_disk(obj_name, obj, base_path):
    # store obj to file, depending on type
    if isinstance(obj, np.ndarray):
        file_path = f"{obj_name}.npy"
        obj_type = "numpy"
        with open(os.path.join(base_path, file_path), 'wb') as f:
            np.save(f, obj)
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        file_path = f"{obj_name}.pkl"
        obj_type = "pandas"
        obj.to_pickle(os.path.join(base_path, file_path))
    elif isinstance(obj, str):
        file_path = f"{obj_name}.txt"
        obj_type = "text"
        with open(os.path.join(base_path, file_path), "w") as f:
            f.write(obj)
    else:
        file_path = f"{obj_name}.pkl"
        obj_type = "pickle"
        with open(os.path.join(base_path, file_path), "wb") as f:
            pickle.dump(obj, f)

    return file_path, obj_type

def get_obj_from_disk(file_path, obj_type, base_path):
    # get obj from file, which has previously been written to there using above function store_obj_to_disk
    if obj_type == "numpy":
        with open(os.path.join(base_path, file_path), 'rb') as f:
            obj = np.load(f, allow_pickle=True)
    elif obj_type == "pandas":
        obj = pd.read_pickle(os.path.join(base_path, file_path))
    elif obj_type == "text":
        with open(os.path.join(base_path, file_path), "r") as f:
            obj = f.read()
    elif obj_type == "pickle":
        with open(os.path.join(base_path, file_path), "rb") as f:
            obj = pickle.load(f)
    else:
        raise ValueError(f"Invalid obj_type {obj_type}")

    return obj


def safe_divide(a, b):
    return a / b if b != 0 else 0

def objects_equal(a, b):
    """
    Check if two objects are equal. The == operator doesn't work for numpy arrays, where an exception is raised
    if the arrays are not of the same size, and a element-wise comparison is done if they are
    """
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    else:
        return a == b


def python_to_json(d):
    # dump python data to json with some special handling
    # this is used to write to easily handleable text files
    def process(data):
        if isinstance(data, dict):
            return {key: process(data[key]) for key in data}  # recursive
        if isinstance(data, list):
            return [process(list_item) for list_item in data]  # recursive
        if isinstance(data, tuple):
            return tuple([process(list_item) for list_item in data])  # recursive
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (pd.DataFrame, pd.Series)):
            return process(data.to_dict())
        elif isinstance(data, AutoDetectDecodingConfig):
            return str(data)
        elif isinstance(data, (np.float32, np.float64, np.float_)):
            return float(data)
        elif isinstance(data, (np.int32, np.int64, np.int_)):
            return int(data)
        if isinstance(data, Tensor):
            return data.cpu().numpy().tolist()
        else:
            return data

    return json.dumps(process(d))


def prepare_demographics(demographics, include_socioeconomic=False):
    # prepare the most important demographic for use in modelling (quantitative)
    mapping = {
        'gender_unified': {'f': 0, 'm': 1},
        'education_binary': {'low-education': 0, 'high-education': 1},
        'country': {'uk': 0, 'usa': 1},
    }

    if demographics is None:
        print("Attention: No demographics provided")
        return None, mapping

    important_demographic_cols = ['age', 'gender_unified', 'education_binary', 'country']
    if include_socioeconomic:
        important_demographic_cols.append('socioeconomic')

    df = demographics.copy()[important_demographic_cols]
    assert demographics.shape[1] >= len(important_demographic_cols), f"There are only {demographics.shape[1]} demographic variables, should be more?"

    assert set(demographics['gender_unified']) == {'f', 'm'}, f"Gender should be f/m before preprocessing, but is {set(demographics['gender_unified'])}. Have the demographics already been preprocessed?"
    assert {'high-education', 'low-education'}.issubset(set(demographics['education_binary'])), f"Education should be high-education / low-education before preprocessing, but is {set(demographics['education_binary'])}. Have the demographics already been preprocessed?"

    for col in ['gender_unified', 'education_binary', 'country']:
        df[col] = df[col].apply(lambda x: mapping[col].get(x, None))

    if np.any(df.isna().sum(axis=0) == df.shape[0]):
        # All NaN columns --> indicates that the demographics have already been preprocessed before?
        raise ValueError(f"All NaN demographic column. Has it been preprocessed already? {df.isna().sum(axis=0)}")

    return df, mapping

def prepare_stratification(data: pd.DataFrame):
    source_stratification_cols = ['gender_unified', 'education_binary', 'country', 'age', 'mean_composite_cognitive_score']
    final_stratification_cols = ['gender_unified', 'education_binary', 'country', 'age_binned', 'mean_composite_cognitive_score_binned']
    stratification_df = data[source_stratification_cols].copy()
    stratification_df['mean_composite_cognitive_score_binned'] = pd.qcut(stratification_df['mean_composite_cognitive_score'], q=4)
    stratification_df['age_binned'] = pd.qcut(stratification_df['age'], q=5)
    stratification_df = stratification_df.drop(columns=['age', 'mean_composite_cognitive_score'])
    assert set(stratification_df.columns) == set(final_stratification_cols), f"Stratification columns should be {final_stratification_cols}, but are {stratification_df.columns}"
    assert stratification_df.shape[0] == data.shape[0], "Stratification dataframe should have the same number of rows as the original data"
    assert not stratification_df.isna().any().any(), "Stratification dataframe should not contain NaN values"
    stratification_array = stratification_df.apply(lambda row: "".join(row.astype(str)), axis=1)
    return stratification_array

def calculate_cohens_d(d1: pd.Series, d2: pd.Series) -> float:
    # Cohen's d effect size between d1 and d2
    # source: https://stackoverflow.com/a/71875070
    n1, n2 = len(d1), len(d2)
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    u1, u2 = np.mean(d1), np.mean(d2)
    return (u1 - u2) / pooled_std

def mean_absolute_percentile_error(y_true, y_pred):
    # The mean absolute error, after converting to percentiles.
    # This is a more interpretable metric of regressionin a medical context
    q = np.arange(1, 101)
    target_percentiles = np.percentile(y_true, q, method="linear")
    target_transformed = np.array([np.argmin(val > target_percentiles) for val in y_true])
    target_transformed = q[target_transformed]
    prediction_transformed = np.array([np.argmin(val > target_percentiles) if val < np.max(target_percentiles) else 99 for val in y_pred])
    prediction_transformed = q[prediction_transformed]
    return sk_metrics.mean_absolute_error(target_transformed, prediction_transformed)


composite_target_to_string_mapping = {'composite_speed': 'Speed', 'composite_language': 'Language', 'composite_executive_function': 'Executive Function', 'composite_memory': 'Memory',
                                      'speed': 'Speed', 'language': 'Language', 'executive_function': 'Executive Function', 'memory': 'Memory'}
