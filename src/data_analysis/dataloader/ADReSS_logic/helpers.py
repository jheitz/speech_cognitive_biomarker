import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
import re

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
        elif isinstance(data, (np.float32, np.float64, np.float_)):
            return float(data)
        elif isinstance(data, (np.int32, np.int64, np.int_)):
            return int(data)
        else:
            return data

    return json.dumps(process(d))


def objects_equal(a, b):
    """
    Check if two objects are equal. The == operator doesn't work for numpy arrays, where an exception is raised
    if the arrays are not of the same size, and a element-wise comparison is done if they are
    """
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    else:
        return a == b

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


def get_adress_sample_name_from_path(path):
    """
    Takes the URL path of a input file (audio file, transcription) and returns the sample_name of the corresponding sample,
    depending on the dataset.
    E.g. '/home/ubuntu/methlab/Students/Jonathan/data/dementiabank_extracted/0extra/ADReSS-IS2020-data/test/transcription/S160.cha' --> S160
    """

    basename = os.path.basename(path)
    regex = r"^(.*)\.(cha|mp3|wav)$"
    parts = re.search(regex, basename)
    try:
        sample_name = parts.group(1)
    except:
        sample_name = ""
        print(f"Error getting sample_name from path {path}. Setting to empty string")

    return sample_name


def get_adress_sample_names_from_paths(paths: np.ndarray):
    assert isinstance(paths, np.ndarray)
    assert paths.shape[0] > 1 and (len(paths.shape) == 1 or paths.shape[1] == 1), \
        f"paths should be a 2d column vector or 1d vector, but has dimensions {paths.shape}"

    vectorized_extract = np.vectorize(get_adress_sample_name_from_path)
    return vectorized_extract(paths)
