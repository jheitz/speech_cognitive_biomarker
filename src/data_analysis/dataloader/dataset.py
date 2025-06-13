import numpy as np
import pandas as pd
import sys, os, json
import inspect
import copy
from collections.abc import Iterable
from enum import Enum

from datasets import Dataset as HfDataset
from util.helpers import objects_equal, python_to_json, create_directory, get_obj_from_disk, store_obj_to_disk


class DatasetType(Enum):
    LUHA = 1
    ADReSS = 2

class Dataset:
    def __init__(self, name, type: DatasetType, config, sample_names, **data):
        """
        :param name: Name of the dataset (for printing etc.)
        :param type: Type of the dataset (for case distinctions)
        :param config: Any config information that defines the dataset.
                       If given the hash of it is used for saving to and loading from disk
        :param data: Any data fields (audio, text, transcripts, features, cognitive scores)
        :param sample_names: Name for each sample, to identify it in the source data if required
        """
        self.config = config
        self.name = name
        self.type = type
        self.sample_names = sample_names

        # add arbitrary additional attributes as instance variables
        for key in data.keys():
            self.__setattr__(key, data[key])

        self.initialization_finished = True
        self._check_data()

    @property
    def data_variables(self):
        # instance variables that represent data, i.e. should be one line per sample
        # this is used to e.g. merge or concatenate data
        return {var: self.__dict__[var] for var in self.__dict__.keys() if isinstance(self.__dict__[var], (list, np.ndarray, pd.DataFrame))}

    def __setattr__(self, name, value):
        # print(f"Setting variable {name}")
        name, value = self._preprocess_data(name, value)
        object.__setattr__(self, name, value)
        self._check_data()

    def _preprocess_data(self, key, value):
        # preprocess data variables (audio, text, transcripts, features, cognitive scores)
        # we take any sample_name column, check it against self.sample_name for consistency, then remove it
        def check_sample_names_against_reference(df, sample_name_cols):
            if len(sample_name_cols) == 0:
                return True
            sample_name_col_values = np.array(df[sample_name_cols[0]])
            assert np.all(sample_name_col_values == np.array(self.sample_names)), \
                "List of sample names / study_submission-ids do not match"
            return True

        if isinstance(value, pd.DataFrame):
            sample_name_cols = [c for c in value.columns if c in ['sample_name', 'sample_names', 'study_submission_id', 'study_submission_ids']]
            assert check_sample_names_against_reference(value, sample_name_cols)
            value_preprocessed = value.copy().drop(columns=sample_name_cols)
        else:
            value_preprocessed = value

        return key, value_preprocessed

    def _check_data(self):
        """
        Check that the dataset is in the expected format
        """

        if not hasattr(self, 'initialization_finished') or not self.initialization_finished:
            return True

        def data_length(obj):
            if isinstance(obj, np.ndarray) or isinstance(obj, pd.DataFrame):
                return obj.shape[0]
            elif isinstance(obj, Iterable):
                return len(obj)

        # check lengths (number of samples) is the same in all data_variables
        data_lengths = pd.Series({var: data_length(self.data_variables[var]) for var in self.data_variables.keys()})
        if not all([data_lengths.loc[idx] == data_lengths.iloc[0] for idx in data_lengths.index]):
            print(f"Incompatible data lengths: {data_lengths}")
            raise ValueError("Incompatible data lengths")

        # check index is the same and increasing sequence in all data_variables if pd.DataFrame
        index_list = pd.Series({var: self.data_variables[var].index.to_list() for var in self.data_variables.keys() if
                                isinstance(self.data_variables[var], (pd.DataFrame, pd.Series))})
        if len(index_list.index) > 1:
            first_index_name = index_list.index[0]
            for idx in index_list.index:
                if index_list.loc[idx] != index_list.loc[first_index_name]:
                    raise ValueError(f"Incompatible indices between {first_index_name} and {idx}\n\n{first_index_name}:\n{index_list.loc[first_index_name]}\n\n\n\n{idx}:\n{index_list.loc[idx]}")

        # check that sample_name is not a column in any dataframe, it should only be present in self.sample_names
        sample_name_columns = {'sample_name', 'sample_names', 'study_submission_id', 'study_submission_ids'}
        assert all([len(sample_name_columns.intersection(set(self.data_variables[var].columns))) == 0
                    for var in self.data_variables.keys()
                    if isinstance(self.data_variables[var], pd.DataFrame)])

        assert self.type in [DatasetType.LUHA, DatasetType.ADReSS], f"Invalid dataset type: {self.type}"

        return True

    def __len__(self):
        """ Length of dataset, must be set as a child class of torch.utils.data.Dataset """
        return len(self.sample_names)

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
        def shape(obj):
            if isinstance(obj, np.ndarray) or isinstance(obj, pd.DataFrame):
                return obj.shape
            elif isinstance(obj, Iterable):
                return len(obj)

        shorten_string = lambda str: str[:100]+"..." if len(str) > 100 else str
        config_str = str({key: shorten_string(str(val)) for key, val in self.config.items()})
        return f"Dataset {self.name} with variables {[(var, shape(self.data_variables[var]) if self.data_variables[var] is not None else 'None') for var in self.data_variables]}, config ({config_str})"

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
        return DatasetClass(**new_data, name=new_name, config=new_config, type=self.type)

    def subset_from_sample_names(self, sample_names):
        # create new subset Dataset from sample_names, return it
        sample_names_filter = np.array(sample_names).astype("str")
        non_existing = [s for s in sample_names_filter if s not in self.sample_names]
        assert len(non_existing) == 0, f"Sample names {non_existing} do not exist in the dataset"

        # get indices for sample_names
        indices = np.isin(self.sample_names, sample_names_filter).nonzero()[0]

        new_dataset = self.subset_from_indices(indices)
        assert set(sample_names_filter) == set(new_dataset.sample_names)
        return new_dataset

    def store_to_disk(self, base_path):
        create_directory(base_path)
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
        n_default_values = len(argspec.defaults) if argspec.defaults is not None else 0
        required_arguments = init_arguments[:len(init_arguments) - n_default_values]
        required_arguments = [a for a in required_arguments if a != 'self']
        for arg in required_arguments:
            assert arg in data, f"Required argument {arg} not available on disk?"

        # create object
        dataset = new_class(**data)
        print(f"Created object of class {new_class}: {str(dataset)}")
        return dataset

    def copy(self):
        """
        Deep copy the dataset
        """
        config = copy.deepcopy(self.config)
        name = self.name
        type = self.type
        sample_names = copy.deepcopy(self.sample_names)

        data = {}
        for var_name in self.data_variables.keys():
            if var_name == 'sample_names':
                continue
            if isinstance(self.data_variables[var_name], (pd.DataFrame, np.ndarray)):
                data[var_name] = self.data_variables[var_name].copy()
            else:
                data[var_name] = copy.deepcopy(self.data_variables[var_name])

        return Dataset(name=name, type=type, sample_names=sample_names, config=config, **data)


