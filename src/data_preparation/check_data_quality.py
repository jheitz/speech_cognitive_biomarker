import sys

sys.path.insert(0, '..') # to make the import from parent dir work

from config.config import Config
from config.constants import Constants
from config.run_parameters import RunParameters
from data_preparation.preparation_logic.data_chality_checker import DataQualityChecker

if __name__ == '__main__':
    # run parameters from command line arguments
    run_parameters = RunParameters.from_command_line_args()

    # configuration based on config file
    config = Config.from_yaml(run_parameters.config)

    # constants for e.g. directory paths
    CONSTANTS = Constants(local=run_parameters.local)

    data_quality_checker = DataQualityChecker(run_parameters, config, CONSTANTS)
    data_quality_checker.check()
