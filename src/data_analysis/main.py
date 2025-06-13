import sys
import argparse
from typing import Type
import os
import socket

sys.path.append(sys.path[0] + '/..')  # to make the import from parent dir util work

from config.config import Config
from config.run_parameters import RunParameters
from config.constants import Constants
from data_analysis.model.regression import Regression
from data_analysis.model.classification import Classification
from util.helpers import create_directory
from data_analysis.dataloader.dataloader import DataLoader, ADReSSDataLoader
from data_analysis.data_transformation.linguistic_features import LinguisticFeatures
from data_analysis.data_transformation.cognitive_residuals import CognitiveResiduals
from data_analysis.data_transformation.feature_residuals import FeatureResiduals
from data_analysis.data_transformation.audio_features import AudioFeatures
from data_analysis.data_transformation.demographic_features import DemographicFeatures
from data_analysis.data_transformation.forced_alignment_word_timestamps import ForcedAlignmentWordTimestamps



def run(run_parameters: RunParameters, config: Config, CONSTANTS: Constants):
    print(f"Running pipeline from user {os.getlogin()} on host {socket.gethostname()}...")
    print("Run Parameters:")
    print(run_parameters, end="\n\n")
    print("Config:")
    print(config, end="\n\n")

    try:
        debug = config.debug
    except AttributeError:
        debug = False

    try:
        dataset = config.config_data.dataset
    except AttributeError:
        dataset = "LUHA"

    if dataset == 'LUHA':
        dataloader = DataLoader(debug=debug, config=config)
    elif dataset == 'ADReSS':
        dataloader = ADReSSDataLoader(debug=debug, config=config)
    else:
        raise ValueError(f"Invalid dataset {dataset}")

    data_transformers = []
    if config.data_transformers is not None:
        for p in config.data_transformers:
            if p == "Linguistic Features":
                data_transformers.append(LinguisticFeatures(config=config, constants=CONSTANTS, run_parameters=run_parameters))
            elif p == "Cognitive Residuals":
                data_transformers.append(CognitiveResiduals(config=config, constants=CONSTANTS, run_parameters=run_parameters))
            elif p == "Feature Residuals":
                data_transformers.append(FeatureResiduals(config=config, constants=CONSTANTS, run_parameters=run_parameters))
            elif p == "Audio Features":
                data_transformers.append(AudioFeatures(config=config, constants=CONSTANTS, run_parameters=run_parameters))
            elif p == "Demographic Features":
                data_transformers.append(DemographicFeatures(config=config, constants=CONSTANTS, run_parameters=run_parameters))
            elif p == "Force Alignment Word Timestamps":
                data_transformers.append(ForcedAlignmentWordTimestamps(config=config, constants=CONSTANTS, run_parameters=run_parameters))
            elif p == "Outlier Removal and Imputation" or p == "Feature Standardizer":
                raise ValueError(f"{p} is no longer a data transformer. Instead it is implemented as a preprocessing step. This is to avoid data leakage (should not be run on the entire dataset). Update the config accordingly.")
            else:
                raise ValueError("Invalid data transformer:", p)
            pass

    data = dataloader.load_data()
    print("Data before transformations: ", data)
    for p in data_transformers:
        print(f"Running data transformer {p}...")
        data = p.preprocess_dataset(data)
    print("Data after transformations:", data)

    if config.model is None:
        raise ValueError("Model not specified")
    elif config.model == 'Regression':
        ModelClass = Regression
    elif config.model == 'Classification':
        ModelClass = Classification
    else:
        raise ValueError(f"Invalid model specified: {config.model}")

    model = ModelClass(config=config, run_parameters=run_parameters, constants=CONSTANTS)
    model.set_data(data)
    model.prepare_data()
    model.run()

    print("done")




if __name__ == '__main__':
    # run parameters from command line arguments
    run_parameters = RunParameters.from_command_line_args()

    # configuration based on config file
    config = Config.from_yaml(run_parameters.config)

    # constants for e.g. directory paths
    CONSTANTS = Constants(local=run_parameters.local)

    # create results file
    create_directory(run_parameters.results_dir)

    run(run_parameters, config, CONSTANTS)
