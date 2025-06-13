import os
import pandas as pd
import shutil
import sys
from datetime import datetime
import logging
import numpy as np


sys.path.insert(0, '..') # to make the import from parent dir work

from config.constants import Constants
from preparation_logic.ACS_outlier_removal_imputation import ACSOutlierRemovalImputation

class ProcessedDataCombiner:
    def __init__(self, constants):
        self.constants = constants
        self.PREPROCESSED_COMBINED_PATH = constants.DATA_PROCESSED_COMBINED
        self.PREPROCESSED_COMBINED_DATA_PATH = os.path.join(self.PREPROCESSED_COMBINED_PATH, 'data')

    def _combine_csv(self, data_paths, csv_path):
        data = []
        for data_path in data_paths:
            csv_path_full = os.path.join(data_path, csv_path)
            if not os.path.exists(csv_path_full):
                logging.error(f"{csv_path_full} does not exist")
                continue
            data.append(pd.read_csv(csv_path_full))
        concatenated = pd.concat(data, ignore_index=True)
        return concatenated

    def _combine_automatic_test_scoring_csv(self, round_paths, dir_to_store):
        print("\n")
        print("Combining overall CSV data files for automatic_scoring")
        os.makedirs(dir_to_store, exist_ok=True)
        phonemicFluencyScores = self._combine_csv(round_paths, "../automatic_test_scoring/phonemicFluencyScores.csv")
        #phonemicFluencyScores.sort_values(by="study_submission_id").to_csv(os.path.join(dir_to_store, "phonemicFluencyScores.csv"), index=False)

        semanticFluencyScores = self._combine_csv(round_paths, "../automatic_test_scoring/semanticFluencyScores.csv")
        #semanticFluencyScores.sort_values(by="study_submission_id").to_csv(os.path.join(dir_to_store, "semanticFluencyScores.csv"), index=False)

        pictureNamingScores = self._combine_csv(round_paths, "../automatic_test_scoring/pictureNamingScores.csv")
        #pictureNamingScores.sort_values(by="study_submission_id").to_csv(os.path.join(dir_to_store, "pictureNamingScores.csv"), index=False)

        combined = (
            phonemicFluencyScores.rename(columns={'score': 'phonemic_fluency_score'})
            .merge(semanticFluencyScores.rename(columns={'score': 'semantic_fluency_score'}), on="study_submission_id", how="outer")
            .merge(pictureNamingScores.rename(columns={'score': 'picture_naming_score'}), on="study_submission_id", how="outer")
        )
        combined.sort_values(by="study_submission_id").to_csv(os.path.join(dir_to_store, "language_task_scores.csv"), index=False)

    def _store_study_csv_files(self, dir_to_store, combined_csv_data: dict[str, pd.DataFrame], submission_id_subset=None):
        os.makedirs(dir_to_store, exist_ok=True)

        for csv_name in combined_csv_data.keys():
            path = os.path.join(dir_to_store, f"{csv_name}.csv")
            data = combined_csv_data[csv_name]
            data['study_submission_id'] = data['study_submission_id'].round(0).astype(int)

            if submission_id_subset is not None:
                data = data[data.study_submission_id.isin(submission_id_subset)]
                if data.shape[0] != len(submission_id_subset):
                    logging.error(
                        f"{len(submission_id_subset)} submission ids given, but only found {data.shape[0]} in {csv_name}, missing ids {[id for id in submission_id_subset if id not in data.study_submission_id.to_list()]}")

            data.sort_values(by="study_submission_id").to_csv(path, index=False)


    def _combine_study_csv_files(self, round_paths, dir_to_store):
        prolific_data = self._combine_csv(round_paths, "prolific_data.csv")
        prolific_data.sort_values(by="study_submission_id").to_csv(os.path.join(dir_to_store, "_prolific_data.csv"), index=False)
        study_submissions = self._combine_csv(round_paths, "study_submissions.csv")
        study_submissions.sort_values(by="study_submission_id").to_csv(os.path.join(dir_to_store, "_study_submissions.csv"), index=False)

        demographics = self._combine_csv(round_paths, "demographics.csv")
        demographics.sort_values(by="study_submission_id").to_csv(os.path.join(dir_to_store, "demographics.csv"), index=False)

        # ACS scores in three versions: orginal scores / with outlier removal based on norms / with imputation
        # version two is based on the default demographic parameters instead of mouse_type, see kw02/acs_norms for details
        outlier_imputation = ACSOutlierRemovalImputation(constants=self.constants, version_logic=1)
        ## original scores
        acs_outcomes_raw = self._combine_csv(round_paths, "acs_outcomes.csv")
        acs_outcomes_raw.sort_values(by="study_submission_id").to_csv(os.path.join(dir_to_store, "acs_outcomes_raw.csv"), index=False)
        ## remove outliers
        acs_outcomes_outliers_removed = outlier_imputation._remove_outliers(acs_outcomes_raw.copy(), demographics)
        acs_outcomes_outliers_removed.sort_values(by="study_submission_id").to_csv(os.path.join(dir_to_store, "acs_outcomes_outliers_removed.csv"), index=False)
        ## with imputation of missing values
        acs_outcomes_imputed = outlier_imputation._impute_missing_values(acs_outcomes_outliers_removed, verbose=False)
        acs_outcomes_imputed.sort_values(by="study_submission_id").to_csv(os.path.join(dir_to_store, "acs_outcomes_imputed.csv"), index=False)

        # Note: prolific data and study_submissions should not be needed anymore
        # all necessary information should be in demographics
        # we keep them with a underscore prefix as "internal" data right now, could be removed in the future.
        csv_data = {
            "_prolific_data": prolific_data,
            "_study_submissions": study_submissions,
            "demographics": demographics,
            "acs_outcomes_raw": acs_outcomes_raw,
            "acs_outcomes_outliers_removed": acs_outcomes_outliers_removed,
            "acs_outcomes_imputed": acs_outcomes_imputed
        }

        return csv_data

    def combine_data(self):
        print("Combining preprocessed data:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        if os.path.exists(self.PREPROCESSED_COMBINED_PATH):
            print(f"Deleting old process_combined data at {self.PREPROCESSED_COMBINED_PATH}")
            shutil.rmtree(constants.DATA_PROCESSED_COMBINED)
        #os.makedirs(self.PREPROCESSED_COMBINED_DATA_PATH, exist_ok=True)

        round_paths = []
        for round in sorted(os.listdir(constants.DATA_PROCESSED)):
            if os.path.isdir(os.path.join(constants.DATA_PROCESSED, round)):
                source_path = os.path.join(constants.DATA_PROCESSED, round, "data")
                print(f"\nCopying data from round {round} ({source_path}): \n... ", end="")
                assert os.path.exists(source_path), "{} does not exist".format(os.path.join(constants.DATA_PROCESSED, round, "data"))
                round_paths.append(source_path)
                for submission_id in sorted(os.listdir(source_path)):
                    if not submission_id.isdigit():  # other subfolder, such as automatic_test_scoring
                        continue
                    source_dir = os.path.join(source_path, submission_id)
                    if os.path.isdir(source_dir):
                        print(submission_id, end=" ")
                        destination_dir = os.path.join(self.PREPROCESSED_COMBINED_DATA_PATH, submission_id)
                        shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)

        #language_task_score_dir = os.path.join(self.PREPROCESSED_COMBINED_PATH, "automatic_test_scoring")
        language_task_score_dir = self.PREPROCESSED_COMBINED_DATA_PATH
        self._combine_automatic_test_scoring_csv(round_paths, dir_to_store=language_task_score_dir)

        print("\nCombining overall CSV data files")
        combined_csv_data = self._combine_study_csv_files(round_paths, dir_to_store=self.PREPROCESSED_COMBINED_DATA_PATH)
        self._store_study_csv_files(self.PREPROCESSED_COMBINED_DATA_PATH, combined_csv_data)

        print("\nSplit dataset in two")
        assignment = pd.read_csv(os.path.join(constants.RESOURCES_DIR, "data_split_2024-07-10_16-11.csv"))
        ids_split1 = assignment.query("split == 1").study_submission_id
        self._store_study_csv_files(os.path.join(self.PREPROCESSED_COMBINED_PATH, "split1"), combined_csv_data, submission_id_subset=ids_split1)
        ids_split2 = assignment.query("split == 2").study_submission_id
        self._store_study_csv_files(os.path.join(self.PREPROCESSED_COMBINED_PATH, "split2"), combined_csv_data, submission_id_subset=ids_split2)


if __name__ == '__main__':
    constants = Constants()
    data_combiner = ProcessedDataCombiner(constants)
    data_combiner.combine_data()
    print("Done!")
