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
from data_analysis.dataloader.ADReSS_logic.dataset import Dataset, AudioDataset
from data_analysis.dataloader.ADReSS_logic.helpers import get_adress_sample_names_from_paths



class DataLoader:
    def __init__(self, name, debug=False, local=None, constants=None, config=None):
        self.debug = debug
        self.name = name
        self.preprocessors = []
        if constants is None:
            self.CONSTANTS = Constants(local=local)
        else:
            self.CONSTANTS = constants
        self.config = config
        print(f"Initializing dataloader {self.name}")

        # Allow option to use only train data, no test data, useful for e.g. hyperparameter tuning, where test
        # set should not be touched
        try:
            self.only_train = self.config.config_data.only_train
        except (AttributeError, KeyError):
            self.only_train = False

    @abstractmethod
    def _load_train(self):
        pass

    @abstractmethod
    def _load_test(self):
        pass

    def load_data(self):
        print(f"Loading data using dataloader {self.name}")
        train = self._load_train()
        test = self._load_test()

        if self.only_train:
            print("Using only train data (dropping test), splitting into new train / test split")
            indices = np.arange(len(train))
            new_train_indices, new_test_indices = model_selection.train_test_split(indices, test_size=0.3,
                                                                                   shuffle=True, random_state=123,
                                                                                   stratify=train.labels)
            test = train.subset_from_indices(new_test_indices)
            train = train.subset_from_indices(new_train_indices)
            train_label_distribution = {label: np.sum(np.where(train.labels == label, 1, 0)) for label in set(train.labels)}
            test_label_distribution = {label: np.sum(np.where(test.labels == label, 1, 0)) for label in set(test.labels)}
            print(f"New train: {len(train)} (labels {train_label_distribution})")
            print(f"New test: {len(test)} (labels {test_label_distribution})")
        return train, test


class ADReSSDataLoader(DataLoader):

    def __init__(self, name="ADReSS audio", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.preprocessors = []
        self.dir_train_ad = self.CONSTANTS.DATA_ADReSS_TRAIN_AD
        self.dir_train_cc = self.CONSTANTS.DATA_ADReSS_TRAIN_CONTROL
        self.dir_test = self.CONSTANTS.DATA_ADReSS_TEST

    def _load_train(self):
        paths = []
        labels = []
        for dir_path in [self.dir_train_ad, self.dir_train_cc]:
            label = 1 if os.path.basename(dir_path) == 'cd' else 0  # cc = control or cd = dementia
            for i, file_name in enumerate(os.listdir(dir_path)):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(dir_path, file_name)
                    paths.append(file_path)
                    labels.append(label)
                if self.debug and i >= 1:
                    break

        metadata_collected = []
        for meta_file in [self.CONSTANTS.DATA_ADReSS_TRAIN_AD_METADATA, self.CONSTANTS.DATA_ADReSS_TRAIN_CONTROL_METADATA]:
            assert os.path.exists(meta_file), f"Train label / metadata file {meta_file} not available"
            metadata = pd.read_csv(meta_file, delimiter=';')
            metadata.columns = [c.strip() for c in metadata.columns]
            metadata['ID'] = metadata['ID'].str.strip()
            metadata['gender'] = metadata['gender'].str.strip().replace("male", 1).replace("female", 0).astype(int)  # f - 0, m - 1, according to LUHA logic, not ADReSS logic
            metadata_collected.append(metadata)
        metadata_collected = pd.concat(metadata_collected, ignore_index=True).set_index('ID')
        demographics = metadata_collected[['age', 'gender', 'mmse']].loc[get_adress_sample_names_from_paths(np.array(paths))]

        dataset = AudioDataset(data=np.array(paths), labels=np.array(labels),
                               sample_names=get_adress_sample_names_from_paths(np.array(paths)),
                               name=f"{self.name} (train)", demographics=demographics,
                               config={'preprocessors': self.preprocessors, 'debug': self.debug})
        return dataset

    def _load_test(self):
        paths = []
        for i, file_name in enumerate(os.listdir(self.dir_test)):
            if file_name.endswith('.wav'):
                file_path = os.path.join(self.dir_test, file_name)
                file_id = re.sub(r'\.wav', '', os.path.basename(file_path))
                paths.append((file_id, file_path))
            if self.debug and i > 1:
                break

        paths = pd.DataFrame(paths, columns=['ID', 'path'])

        assert os.path.exists(self.CONSTANTS.DATA_ADReSS_TEST_METADATA), "Test label / metadata file not available"

        metadata = pd.read_csv(self.CONSTANTS.DATA_ADReSS_TEST_METADATA, delimiter=';')
        metadata.columns = [c.strip() for c in metadata.columns]
        metadata['ID'] = metadata['ID'].str.strip()
        metadata['gender'] = metadata['gender'].replace({0: 1, 1: 0})  # reverse labels to be compatible with LUHA logic

        data = metadata.merge(paths, on="ID", how="inner")
        # labels = np.squeeze(metadata['Label'])
        demographics = data[['age', 'gender', 'mmse']]

        dataset = AudioDataset(data=np.array(data['path']), labels=np.array(data['Label']),
                               sample_names=get_adress_sample_names_from_paths(np.array(data['path'])),
                               name=f"{self.name} (test)", demographics=demographics,
                               config={'preprocessors': self.preprocessors, 'debug': self.debug})
        return dataset


class ADReSSWithPITTDataLoader(ADReSSDataLoader):
    """
    ADReSS dataset, but using the non-preprocessed corresponding files from the PITT corpus
    Also called ADReSS_RAW
    """

    def __init__(self, *args, **kwargs):
        name = "ADReSS PITT audio"
        super().__init__(name, *args, **kwargs)
        self.dir_train_ad = os.path.join(self.CONSTANTS.ADReSS_ORIGINAL_PITT_FILES, "train/cd")
        self.dir_train_cc = os.path.join(self.CONSTANTS.ADReSS_ORIGINAL_PITT_FILES, "train/cc")
        self.dir_test = os.path.join(self.CONSTANTS.ADReSS_ORIGINAL_PITT_FILES, "test")