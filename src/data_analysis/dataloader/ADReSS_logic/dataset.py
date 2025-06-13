import os, shutil
import sys, os, json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics
import torch
import re
import hashlib
import pickle
import urllib.parse
from config.constants import Constants
from abc import abstractmethod
from sklearn import model_selection
import inspect

from data_analysis.dataloader.ADReSS_logic.helpers import *


class Dataset:
    def __init__(self, data, labels, sample_names, name="", config={}, **extra_data):
        """
        :param data: Features, can be text or numbers or whatever
        :param labels: Labels
        :param sample_names: Name for each sample, to identify it in the source data if required
        :param name: Name of the dataset (for printing etc.)
        :param config: Any config information that defines the dataset.
                       If given the hash of it is used for saving to and loading from disk
        """
        self.config = config
        self.name = name
        self.sample_names = sample_names
        self.cv_test_splits = None  # For each sample: Which cross validation split should it be in the test split

        self.data = data
        self.labels = labels

        #self.cv_shuffler: CrossValidationShuffler = None

        # add arbitrary additional attributes as instance variables
        self.__dict__.update(extra_data)

        # instance variables that represent data, i.e. should be one line per sample
        self.potential_data_variable_names = ['data', 'sample_names', 'labels', 'paths', 'tokens', 'demographics']

        self._check_data()

    @property
    def data_variables(self):
        # instance variables that represent data, i.e. should be one line per sample
        # this is used to e.g. merge or concatenate data
        return {var: self.__dict__[var] for var in self.potential_data_variable_names if var in self.__dict__}

    def _check_data(self):
        """
        Check that the dataset is in the expected format
        """
        assert isinstance(self.sample_names, np.ndarray)

        assert isinstance(self.data, (np.ndarray, pd.DataFrame)), \
            f"Invalid type for self.data: {type(self.data)}"

        if self.labels is not None:
            assert isinstance(self.labels, np.ndarray), f"Invalid type for self.labels: {type(self.labels)}"

        if 'paths' in vars(self) and self.paths is not None:
            assert isinstance(self.paths, np.ndarray), f"Invalid type for self.paths: {type(self.paths)}"

        # check sizes of data variables
        for data_var in self.data_variables:
            if self.data_variables[data_var] is None:
                continue

            try:
                data_shape = self.data_variables[data_var].shape[0]
            except Exception as e:
                raise Exception(f"Exception getting data shape for {data_var}:", str(e))

            try:
                reference_var = list(self.data_variables.keys())[0]
                reference_shape = self.data_variables[reference_var].shape[0]
            except Exception as e:
                raise Exception(f"Exception getting reference shape for {data_var}:", str(e))

            assert data_shape == reference_shape, \
                f"Invalid shape for {data_var}: {data_shape}, should be compatible to {reference_var}, which has shape {reference_shape}"

        assert self.config is None or isinstance(self.config, dict)
        return True

    def __len__(self):
        """ Length of dataset, must be set as a child class of torch.utils.data.Dataset """
        self._check_data()
        return self.data.shape[0]

    def __getitem__(self, index):
        """
        Get dictionary of n-th item
        This is to be compatible with torch.utils.data.Dataset
        """
        row = {}
        for var in self.data_variables:
            if self.data_variables[var] is None:
                row[var] = None
            elif isinstance(self.data_variables[var], pd.DataFrame):
                row[var] = self.data_variables[var].iloc[index, :]
            elif isinstance(self.data_variables[var], np.ndarray):
                row[var] = self.data_variables[var][index]
            elif isinstance(self.data_variables[var], list):
                row[var] = self.data_variables[var][index]
            else:
                raise NotImplementedError(f"__getitem__ not implemented for {var} of type {type(self.data_variables[var])}")

        return row

    def __str__(self):
        return f"Dataset {self.name} with variables {[(var, self.data_variables[var].shape if self.data_variables[var] is not None else 'None') for var in self.data_variables]}, config ({self.config})"

    def concatenate(self, other: 'Dataset'):
        # concatenate with other Dataset and return new dataset
        self._check_data()
        other._check_data()

        new_name = f"Concatenate({self.name}, {other.name})"
        new_config = {}
        for key in list(self.config.keys()) + list(other.config.keys()):
            if key in self.config and key in other.config and objects_equal(other.config[key], self.config[key]):
                new_config[key] = self.config[key]
            elif key in self.config and key in other.config:
                new_config[key] = [self.config.get(key, None), other.config.get(key, None)]

        # check whether the same data variables are present
        data_variables_self = set(self.data_variables.keys())
        data_variables_other = set(other.data_variables.keys())
        assert data_variables_self == data_variables_other, \
            f"Cannot concatenate datasets, as one has data_variables {data_variables_self} while other has {data_variables_other}"

        def concat_data(one, two):
            assert type(one) == type(two), f"Can only concatenate if same type, but found {type(one)}, {type(two)}"
            if one is None:
                return None
            if isinstance(one, np.ndarray):
                new = np.concatenate((one, two), axis=0)
                assert new.shape[0] == one.shape[0] + two.shape[0]
            elif isinstance(one, pd.Series):
                new = np.concatenate((one, two), axis=0)
                assert new.shape[0] == one.shape[0] + two.shape[0]
            elif isinstance(one, pd.DataFrame):
                new = pd.concat((one, two), axis=0).reset_index(drop=True)
                assert new.shape[0] == one.shape[0] + two.shape[0]
            else:
                raise TypeError(f"Invalid types for concatenation. Found {type(one)}")

            return new

        # combine data
        new_data = {}
        for var in data_variables_self:
            new_data[var] = concat_data(self.data_variables[var], other.data_variables[var])

        DatasetClass = self.__class__
        return DatasetClass(**new_data, name=new_name, config=new_config)

    def shuffle(self, random_state=None):
        # shuffle dataset and return as new dataset
        random_indices = np.random.default_rng(seed=random_state).permutation(range(len(self)))
        new_name = f"Shuffled({self.name})"
        new_config = {**self.config, 'shuffled': True}

        # recreate data
        new_data = {}
        for var in self.data_variables.keys():
            if isinstance(self.data_variables[var], np.ndarray):
                new_data[var] = self.data_variables[var][random_indices]
            elif isinstance(self.data_variables[var], pd.DataFrame):
                new_data[var] = self.data_variables[var].iloc[random_indices, :].reset_index(drop=True)

        DatasetClass = self.__class__
        return DatasetClass(**new_data, name=new_name, config=new_config)

    def subset_from_indices(self, indices):
        # create new subset Dataset from indices, return it
        self._check_data()

        assert np.max(indices) < len(self), \
            f"Index should be integer index, but is too large {np.max(indices)}, with a dataset of size {len(self)}"
        assert np.min(indices) >= 0

        new_name = f"Subset({self.name}, {len(indices) / len(self):.1%})"
        new_config = {**self.config, 'indices': indices}

        # get data
        new_data = {}
        for var in self.data_variables.keys():
            if isinstance(self.data_variables[var], np.ndarray):
                new_data[var] = self.data_variables[var][indices]
            elif isinstance(self.data_variables[var], pd.DataFrame):
                new_data[var] = self.data_variables[var].iloc[indices, :].reset_index(drop=True)

        DatasetClass = self.__class__
        return DatasetClass(**new_data, name=new_name, config=new_config)

    def load_cv_split_assignment(self, n_splits):
        """
        Load which CV test split each sample should be assigned to, based on sample_name
        """
        assert 'cv_shuffler' in self.config, \
            f"CV Shuffler must be set in self.config when loading the data, so it's clear where to take the assignments from, for dataset {self}"
        #assert isinstance(self.config['cv_shuffler'], CrossValidationShuffler), \
        #    f"CV Shuffler must of type CrossValidationShuffler, but is {type(self.config['cv_shuffler'])}"
        assert self.cv_test_splits is None, \
            "CV split assignments have been calculated already. This should not be called twice, access self.cv_test_split directly"

        self.config['cv_shuffler'].create_mapping(n_splits=n_splits)
        mapping = self.config['cv_shuffler'].get_mapping()

        # make sure the sample_names are in fact available in the mapping
        missing_sample_names = [sample_name for sample_name in self.sample_names if sample_name not in mapping.sample_name.values]
        assert len(missing_sample_names) == 0, f"Missing samples name in mapping: {missing_sample_names}"

        # extract test_split using a Pandas merge
        sample_names_df = pd.DataFrame(self.sample_names, columns=['sample_name'])
        merged = sample_names_df.merge(mapping, on="sample_name", how="left")
        assert np.sum(merged['test_split'].isna()) == 0, "Missing test_split for at least one sample_name?"
        assert np.all(np.array(merged.sample_name) == self.sample_names), "Order has changed for test_split calculation. This will result in a wrong assignment of test_split"
        self.cv_test_splits = np.array(merged.test_split).astype(int)

    def store_to_disk(self, base_path):
        os.makedirs(base_path, exist_ok=True)
        data = {'class': type(self).__name__}
        for obj_name in vars(self):
            obj = getattr(self, obj_name)
            if obj is not None:
                file_path, obj_type = store_obj_to_disk(obj_name, obj, base_path)
                data[obj_name] = {'obj_type': obj_type, 'file_path': file_path}

        # store info on all elements, to recover
        with open(os.path.join(base_path, "dataset.txt"), "w") as file:
            file.write(python_to_json(data))

    @classmethod
    def from_disk(cls, base_path):
        assert os.path.isdir(base_path), "Invalid path for directory of stored dataset"
        with open(os.path.join(base_path, 'dataset.txt')) as json_file:
            info = json.load(json_file)

        try:
            new_class = getattr(sys.modules[__name__], info['class'])
        except:
            raise ValueError(f"Cannot get class {info.get('class')}")

        data = {}
        for obj_name in info:
            if obj_name != 'class':
                obj_type = info[obj_name]['obj_type']
                file_path = info[obj_name]['file_path']
                data[obj_name] = get_obj_from_disk(file_path, obj_type, base_path)

        # get required attributes for constructor
        argspec = inspect.getfullargspec(cls.__init__)
        init_arguments = argspec.args
        n_default_values = len(argspec.defaults)
        required_arguments = init_arguments[:len(init_arguments) - n_default_values]
        required_arguments = [a for a in required_arguments if a != 'self']
        for arg in required_arguments:
            assert arg in data, f"Required argument {arg} not available on disk?"

        # create object
        dataset = new_class(**data)
        print(f"Created object of class {new_class}: {str(dataset)}")
        return dataset





class AudioDataset(Dataset):
    """
    Dataset of audio paths
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

