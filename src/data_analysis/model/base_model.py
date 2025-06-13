"""Abstract base model"""

from abc import ABC, abstractmethod
from config.config import Config
from config.constants import Constants
from config.run_parameters import RunParameters

from data_analysis.dataloader.dataset import Dataset


class BaseModel(ABC):
    """Abstract Model class that is inherited to all models"""

    def __init__(self, name, run_parameters: RunParameters, config: Config, constants: Constants):
        self.name = name
        print(f"Initializing model {self.name}")
        self.config = None
        if config is None or not isinstance(config, Config):
            raise Exception("Model should have a config")
        if run_parameters is None or not isinstance(run_parameters, RunParameters):
            raise Exception("Model should have run parameters")
        if constants is None or not isinstance(constants, Constants):
            raise Exception("Model should have CONSTANTS")

        self.config = config
        self.run_parameters = run_parameters
        self.CONSTANTS = constants

        self.data: Dataset = None

    @abstractmethod
    def prepare_data(self):
        pass

    def set_data(self, dataset: Dataset):
        assert isinstance(dataset, Dataset)
        self.data = dataset

    @abstractmethod
    def run(self):
        pass
