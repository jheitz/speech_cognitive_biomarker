import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import ffmpeg
import re
import shutil
import logging
import warnings
import librosa
import sys
from datetime import datetime

sys.path.insert(0, '..') # to make the import from parent dir work

from config.config import Config
from config.constants import Constants
from config.run_parameters import RunParameters
from util.google_speech_transcription import GoogleSpeechTranscriber
from util.whisper_transcription import Whisper_Transcriber
from util.acs_norm_data import OriginalACSNormDataCalculator


class DataPreparator:
    def __init__(self, run_parameters: RunParameters, config: Config, CONSTANTS: Constants, extra_logic_processors):
        self.run_parameters = run_parameters
        self.config = config
        self.CONSTANTS = CONSTANTS

        # raw dir
        self.RAW_DIR = os.path.join(CONSTANTS.DATA_RAW, config.name)
        assert os.path.exists(self.RAW_DIR)

        # results / preprocessed dir
        try:
            self.RESULTS_DIR = os.path.join(run_parameters.results_dir, "data")
        except:
            self.RESULTS_DIR = os.path.join(CONSTANTS.DATA_PROCESSED, config.name, "data")

        # delete old directory if exists, recreate empty
        if os.path.exists(self.RESULTS_DIR):
            shutil.rmtree(self.RESULTS_DIR)
        os.makedirs(self.RESULTS_DIR, exist_ok=False)

        # also delete old status file
        status_file_path = os.path.join(self.RESULTS_DIR, "..", "_status.txt")
        if os.path.exists(status_file_path):
            os.remove(status_file_path)

        # potential extra logic
        self.extra_logic_processors = extra_logic_processors

    def _run_extra_logic_processor(self, hook):
        for extra_logic_processor in self.extra_logic_processors:
            if hook == 'after_load_raw_data':
                extra_logic_processor.after_load_raw_data(self)
            elif hook == 'after_prepare_data':
                extra_logic_processor.after_prepare_data(self)
            else:
                raise ValueError(f"Invalid extra logic hook {hook}")

    def log_exception(self, message, participant_id=None):
        logging.exception(message)
        if participant_id is not None:
            participant_log_dir = os.path.join(self.RESULTS_DIR, str(participant_id), "log")
            os.makedirs(participant_log_dir, exist_ok=True)
            with open(os.path.join(participant_log_dir, "errorlog.txt"), "a") as errorlog:
                errorlog.write(f"Exception: {message}\n")

    def log_error(self, message, participant_id=None):
        logging.error(message)
        if participant_id is not None:
            participant_log_dir = os.path.join(self.RESULTS_DIR, str(participant_id), "log")
            os.makedirs(participant_log_dir, exist_ok=True)
            with open(os.path.join(participant_log_dir, "errorlog.txt"), "a") as errorlog:
                errorlog.write(f"Error: {message}\n")

    def log(self, message, participant_id=None, only_in_subdir=False):
        if not only_in_subdir:
            print(message)
        if participant_id is not None:
            participant_log_dir = os.path.join(self.RESULTS_DIR, str(participant_id), "log")
            os.makedirs(participant_log_dir, exist_ok=True)
            with open(os.path.join(participant_log_dir, "log.txt"), "a") as errorlog:
                errorlog.write(f"{message}\n")

    def load_raw_data(self):
        print("Loading raw data...")
        study_id = self.config.study_id

        # Load prolific exported data
        self.prolific = pd.read_csv(os.path.join(self.RAW_DIR, "prolific_export.csv"))
        self.prolific = self.prolific[self.prolific.Status.isin(['AWAITING REVIEW', 'APPROVED'])]
        assert self.prolific.shape[0] > 0, "No data in prolific export with proper status?"
        print(f"... {self.prolific.shape[0]} rows in prolific data for study {study_id}")

        # Study submission table
        assert study_id is not None, "Study id not provided in config file"
        self.study_submissions = pd.read_csv(os.path.join(self.RAW_DIR, "database/study_submissions.csv"))
        self.study_submissions = self.study_submissions.query(f"study_id == '{study_id}'")
        self.study_submissions = self.study_submissions.rename(columns={'id': 'study_submission_id'})
        # keep only those in prolific export
        self.study_submissions = self.study_submissions[self.study_submissions.prolific_id.isin(self.prolific['Participant id'])]
        print(f"... {self.study_submissions.shape[0]} rows in study submissions")
        # to debug below error: self.study_submissions.merge(self.prolific, left_on="prolific_id", right_on="Participant id", how="outer")
        assert self.study_submissions.shape[0] == self.prolific.shape[0], \
            f"Number of study_submissions and prolific rows should be the same, but is {self.study_submissions.shape[0]} vs. {self.prolific.shape[0]}"

        # audio recording table
        self.audio_recordings = pd.read_csv(os.path.join(self.RAW_DIR, "database/audio_recordings.csv"))
        # keep only those for study_submissions
        self.audio_recordings = self.audio_recordings[self.audio_recordings.study_submission_id.isin(self.study_submissions['study_submission_id'])]
        print(f"... {self.audio_recordings.shape[0]} rows in audio recordings (for {self.audio_recordings.study_submission_id.drop_duplicates().shape[0]} submissions)")
        assert self.audio_recordings.study_submission_id.drop_duplicates().shape[0] == self.study_submissions.shape[0], "There are submissions without audio files?"

        # rey figure table
        self.rey_figure_drawings = pd.read_csv(os.path.join(self.RAW_DIR, "database/rey_figure_drawings.csv"))
        # keep only those for study_submissions
        self.rey_figure_drawings = self.rey_figure_drawings[self.rey_figure_drawings.study_submission_id.isin(self.study_submissions['study_submission_id'])]
        print(f"... {self.rey_figure_drawings.shape[0]} rows in rey figure drawings (for {self.rey_figure_drawings.study_submission_id.drop_duplicates().shape[0]} submissions)")

        # ACS tokens
        self.acs_tokens = pd.read_csv(os.path.join(self.RAW_DIR, "database/acs_tokens.csv"))
        assert self.acs_tokens.token.shape[0] == self.acs_tokens.token.drop_duplicates().shape[0], "Duplicate ACS tokens?"
        # keep only those for study_submissions
        self.acs_tokens = self.acs_tokens[self.acs_tokens.id.isin(self.study_submissions['acs_token_id'])]
        print(f"... {self.acs_tokens.shape[0]} rows in ACS tokens")
        if self.acs_tokens.shape[0] != self.study_submissions.shape[0]:
            self.log_error(f"Number of ACS tokens and study_submissions should be the same, but are {self.acs_tokens.shape[0]} != {self.study_submissions.shape[0]}")

        # ACS cognitive scores
        self.cognitive_scores = pd.read_excel(os.path.join(self.RAW_DIR, "ACS_RESULTS_PROCESSED.xlsx"), sheet_name="Export_template")

        self._run_extra_logic_processor('after_load_raw_data')

    def prepare_id_mappings(self):
        # mapping between ACS token, study_submission_id, and prolific_id
        self.id_mapping = self.study_submissions[['study_submission_id', 'prolific_id', 'acs_token_id']]
        self.id_mapping = self.id_mapping.merge(self.acs_tokens[['id', 'token']].rename(columns={
            'id': 'acs_token_id',
            'token': 'acs_token',
        }), on='acs_token_id', how="left")
        self.id_mapping = self.id_mapping.sort_values(by='study_submission_id')
        self.id_mapping = self.id_mapping.drop(columns=['acs_token_id'])

        file_path = os.path.join(self.RESULTS_DIR, "id_mapping.csv")
        print(f"Writing id mapping to {file_path}")
        self.id_mapping.to_csv(file_path, index=False)
        file_path = os.path.join(self.RESULTS_DIR, "id_mapping.txt")
        self.id_mapping.apply(lambda row: " / ".join(row.astype(str)), axis=1).to_csv(file_path, index=False)

    def prepare_prolific_data(self):
        # prepare prolific data: remove unnecessary rows and use the study submission id instead of the prolific_id,
        # as the latter is somewhat private

        print("Preparing Prolific data")

        # check status
        assert np.all(self.prolific['Status'].isin(['AWAITING REVIEW', 'APPROVED']))

        self.prolific = self.prolific.drop(
            columns=["Submission id", "Reviewed at", "Archived at", "Completion code", "Total approvals", "Student status",
                     "Employment status", "Status"]).rename(columns={
            'Participant id': 'prolific_id',
            'Started at': "start_time",
            'Completed at': "end_time",
            'Time taken': "total_time",
            "Age": "age",
            "Sex": "sex",
            "Ethnicity simplified": "ethnicity",
            "Country of birth": "country_of_birth",
            "Country of residence": "country_of_residence",
            "Nationality": "nationality",
            "Language": "language",
        })

        # total time convert to minutes
        self.prolific['total_time'] = self.prolific['total_time'] / 60

        # join study submission id
        n_rows_before = self.prolific.shape[0]
        self.prolific = self.prolific.merge(self.study_submissions[['study_submission_id', 'prolific_id']], on="prolific_id", how="left")
        assert n_rows_before == self.prolific.shape[0], "Duplicate study submission id?"
        assert self.prolific.study_submission_id.isna().sum() == 0, "NA study submission id?"
        self.prolific = self.prolific.drop(columns=['prolific_id'])
        self.prolific = self.prolific[['study_submission_id'] + [c for c in self.prolific.columns if c != "study_submission_id"]]

        prolific_data_path = os.path.join(self.RESULTS_DIR, "prolific_data.csv")
        print(f"Writing preprocessed Prolific data to {prolific_data_path}")
        self.prolific.sort_values(by='study_submission_id').to_csv(prolific_data_path, index=False)

        for _, row in self.prolific.iterrows():
            self.log(f"\nProlific data: {row}\n", row.study_submission_id, only_in_subdir=True)

        return self.prolific

    def prepare_study_submission_data(self):
        print("Preparing study submissions")

        # add acs_token
        self.study_submissions = self.study_submissions.merge(self.acs_tokens[['id', 'token']].rename(columns={
            'id': 'acs_token_id',
            'token': 'acs_token',
        }), on="acs_token_id", how="left")

        print(f"... Calculating laterality index from handedness questionnaire")
        # calculate laterality index from handedness questionnaire based on
        # - https://www.brainmapping.org/shared/Edinburgh.php# /
        # - Oldfield, R.C. "The assessment and analysis of handedness: the Edinburgh inventory." Neuropsychologia. 9(1):97-113. 1971.
        def calc_points(question, id):
            # calculate the points counting towards right: 2 if answer=right, otherHand=False, 1 if answer=right, otherHand=True, 1 if answer=no_preference
            # analogous for left: 2 if answer=left, otherHand=False, 1 if answer=left, otherHand=True, 1 if answer=no_preference
            answer = question['answer']
            other_hand = question['otherHand']
            if answer == "no_preference":
                right, left = 1, 1
            elif answer == "right":
                right = 1 if other_hand else 2
                left = 0
            elif answer == "left":
                left = 1 if other_hand else 2
                right = 0
            else:
                self.log_error(f"Invalid answer for handedness questionaire (submission {id}): {question}", id)
                right, left = None, None
            return {'right': right, 'left': left}

        def handedness(row):
            handedness_json_text = row.handedness
            d = json.loads(handedness_json_text)
            points = pd.DataFrame([calc_points(question, row.study_submission_id) for question in d])
            # display(points)
            if np.sum(points.right) + np.sum(points.left) == 0:
                return None
            index = 100 * (np.sum(points.right) - np.sum(points.left)) / (np.sum(points.right) + np.sum(points.left))
            # print(index)
            return index

        self.study_submissions['laterality_index'] = self.study_submissions.apply(handedness, axis=1)

        print("... Removing prolific_id")
        # remove prolific_id (as this is somewhat private and should not be shared - we use our own database id for this dataset)
        self.study_submissions = self.study_submissions.drop(columns=['prolific_id'])
        study_submissions_path = os.path.join(self.RESULTS_DIR, "study_submissions.csv")
        print(f"Saving preprocessed study submissions to {study_submissions_path}")
        self.study_submissions.to_csv(study_submissions_path, index=False)

        for _, row in self.study_submissions.iterrows():
            self.log(f"\nStudy submission data: {row}\n", row.study_submission_id, only_in_subdir=True)
            self.log(f"\nFeedback: {row.feedback}", row.study_submission_id, only_in_subdir=True)


        return self.study_submissions

    def check_demographics_consistency(self):
        # check whether the demographics (age & sex/gender) in prolific data match those collected in our study
        study_submission_data = self.study_submissions[['study_submission_id', 'age', 'gender']].rename(columns={
            'age': 'age_study',
            'gender': 'gender_study'
        })
        prolific_data = self.prolific[['study_submission_id', 'age', 'sex']].rename(columns={
            'age': 'age_prolific',
            'sex': 'sex_prolific'
        }).replace({
            'Female': 'f',
            'Male': 'm'
        })
        merged = study_submission_data.merge(prolific_data, on='study_submission_id', how='inner')
        merged['age_diff'] = merged['age_study'].astype(float) - merged['age_prolific'].astype(float)
        for idx, row in merged.query("age_diff != 0").iterrows():
            self.log_exception(f"Participants {row.study_submission_id} has conflicting age information: {row.age_prolific} (Prolific) vs. {row.age_study} (Study Submission)", row.study_submission_id)

        for idx, row in merged.query("sex_prolific != gender_study").iterrows():
            self.log_exception(f"Participants {row.study_submission_id} has conflicting gender information: {row.sex_prolific} (Prolific) vs. {row.gender_study} (Study Submission)", row.study_submission_id)

        print("ok")

    def prepare_demographics(self):
        # Prepare a unified dataframe of demographic information, from Prolific and study_submissions
        prolific_cols = ['study_submission_id', 'sex', 'ethnicity', 'country_of_birth', 'country_of_residence',
                         'nationality', 'age']
        self.demographics = self.prolific[prolific_cols].rename(columns={'age': 'age_prolific', 'sex': 'sex_prolific'})

        study_submission_cols = ['study_submission_id', 'age', 'gender', 'education', 'language', 'country',
                                 'socioeconomic', 'laterality_index', 'consent_data_further_use']
        self.demographics = self.demographics.merge(
            self.study_submissions[study_submission_cols], on='study_submission_id', how='outer'
        )

        # We have Sex information from Prolific, and gender from study_submissions. The latter is sometimes
        # prefer-not-to-say or other, both of which are problematic for norm data (which is calculated per gender)
        # Here, we define the "unified-gender" as the gender information from the study submission (if m / f),
        # or Prolific, if not available.
        def create_unified_gender(row):
            if row['gender'] in ['f', 'm']:
                return 'f' if row['gender'] == 'f' else 'm' if row['gender'] == 'm' else None
            elif row['sex_prolific'] in ['Male', 'Female']:
                return 'f' if row['sex_prolific'] == 'Female' else 'm' if row['sex_prolific'] == 'Male' else None
            else:
                logging.error(f"Invalid gender information: study_submission: {row['gender']}, Prolific data: {row['sex_prolific']}")
                return row['gender']
        self.demographics['gender_unified'] = self.demographics.apply(create_unified_gender, axis=1)

        # low vs. high education, based on logic as defined in ACS:
        #    -> "Education" (coded as 0 for Verhage 1-5/ISCED 0-4, 1 for Verhage 6-7/ISCED 5-8)
        # see https://ec.europa.eu/eurostat/statistics-explained/index.php?title=International_Standard_Classification_of_Education_(ISCED)#ISCED_1997_.28fields.29_and_ISCED-F_2013
        self.demographics['education_binary'] = self.demographics.loc[:, 'education'].apply(
            lambda x: 'low-education' if x in ['less_than_highschool', 'high_school', 'vocational'] else 'high-education' if x in ['bachelor', 'master', 'phd'] else x)

        self.demographics.loc[:, 'age'] = self.demographics.loc[:, 'age'].round(0).astype(int)

        demographics_data_path = os.path.join(self.RESULTS_DIR, "demographics.csv")
        print(f"Writing preprocessed demographics data to {demographics_data_path}")
        self.demographics.sort_values(by='study_submission_id').to_csv(demographics_data_path, index=False)

        return self.demographics

    def descriptive_statistics(self):
        # write descriptive statistics
        print("Calculating descriptive statistics")

        # study submissions
        categorical = ['education', 'gender', 'country', 'language', 'consent_data_further_use']
        continuous = ['age', 'socioeconomic', 'laterality_index']
        variables = categorical + continuous

        nrows = int(np.floor(np.sqrt(len(variables))))
        ncols = int(np.ceil(len(variables) / nrows))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2.5))
        axes = [ax for row in axes for ax in row]

        for i in range(len(variables)):
            var = variables[i]
            data = self.study_submissions[var]

            if var in continuous:
                axes[i].boxplot(data)
                x = np.random.uniform(0.8, 1.2, size=len(data))
                axes[i].plot(x, data, ".")
            else:
                categories = pd.Series(data).value_counts().to_dict()
                positions = range(len(categories.keys()))
                axes[i].bar(positions, categories.values())
                axes[i].set_xticks(positions, categories.keys(), rotation=40, ha="right")

            title_parts = var.split("_")
            cumulative_letter_count = np.cumsum([len(part) for part in title_parts])
            diff_to_mean = [np.abs(cumcount - cumulative_letter_count[-1] / 2) for cumcount in cumulative_letter_count]
            argmin = np.argmin(diff_to_mean)
            title_split = "_".join(title_parts[:argmin + 1]) + "\n" + "_".join(title_parts[argmin + 1:])

            axes[i].set_title(title_split, size=10)
            # axes[i].set_ylabel(outcomes[i], size=10)

        plt.suptitle("Demographics")
        plt.tight_layout()
        plt.savefig(os.path.join(self.RESULTS_DIR, "study_submission_descriptive_stats.png"))
        plt.close()

        study_submission_stats = []
        for i in range(len(variables)):
            var = variables[i]
            data = self.study_submissions[var]

            if var in continuous:
                study_submission_stats.append({'variable': var, 'distribution': f"{np.mean(data):.2f} +- {np.std(data):.2f}",
                                      "min": f"{np.min(data):.2f}", "max": f"{np.max(data):.2f}"})
            else:
                categories = pd.Series(data).value_counts().to_dict()
                for c in categories.keys():
                    study_submission_stats.append({'variable': f"{var}: {c}",
                                          'distribution': f"{categories[c]} ({int(categories[c] / len(data) * 100)}%)"})

        study_submission_stats_df = pd.DataFrame(study_submission_stats)
        study_submission_stats_df.to_csv(os.path.join(self.RESULTS_DIR, "study_submission_descriptive_stats.csv"), index=False)


        # prolific data
        categorical = ['sex', 'ethnicity', 'country_of_birth', 'country_of_residence', 'nationality', 'language']
        continuous = ['total_time', 'age']
        variables = categorical + continuous

        nrows = int(np.floor(np.sqrt(len(variables))))
        ncols = int(np.ceil(len(variables) / nrows))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2.5))
        axes = [ax for row in axes for ax in row]

        for i in range(len(variables)):
            var = variables[i]
            data = self.prolific[var]

            if var in continuous:
                axes[i].boxplot(data.astype(float))
                x = np.random.uniform(0.8, 1.2, size=len(data))
                axes[i].plot(x, data.astype(float), ".")
            else:
                categories = pd.Series(data).value_counts().to_dict()
                positions = range(len(categories.keys()))
                axes[i].bar(positions, categories.values())
                axes[i].set_xticks(positions, categories.keys(), rotation=40, ha="right")

            title_parts = var.split("_")
            cumulative_letter_count = np.cumsum([len(part) for part in title_parts])
            diff_to_mean = [np.abs(cumcount - cumulative_letter_count[-1] / 2) for cumcount in cumulative_letter_count]
            argmin = np.argmin(diff_to_mean)
            title_split = "_".join(title_parts[:argmin + 1]) + "\n" + "_".join(title_parts[argmin + 1:])

            axes[i].set_title(title_split, size=10)
            # axes[i].set_ylabel(outcomes[i], size=10)

        plt.suptitle("Prolific information")
        plt.tight_layout()
        plt.savefig(os.path.join(self.RESULTS_DIR, "prolific_descriptive_stats.png"))
        plt.close()

        prolific_stats = []
        for i in range(len(variables)):
            var = variables[i]
            data = self.prolific[var]

            if var in continuous:
                data = data.astype(float)
                prolific_stats.append({'variable': var, 'distribution': f"{np.mean(data):.2f} +- {np.std(data):.2f}",
                                      "min": f"{np.min(data):.2f}", "max": f"{np.max(data):.2f}"})
            else:
                categories = pd.Series(data).value_counts().to_dict()
                for c in categories.keys():
                    prolific_stats.append({'variable': f"{var}: {c}",
                                          'distribution': f"{categories[c]} ({int(categories[c] / len(data) * 100)}%)"})

        prolific_stats_df = pd.DataFrame(prolific_stats)
        prolific_stats_df.to_csv(os.path.join(self.RESULTS_DIR, "prolific_descriptive_stats.csv"), index=False)


    def _convert_webm_to_wav(self, input_file, output_file, study_submission_id):
        try:
            # Converting the .webm file to .wav format with overwrite option (-y flag)
            stream = ffmpeg.input(input_file)
            stream = ffmpeg.output(stream, output_file)
            ffmpeg.run(stream, overwrite_output=True, quiet=True)

            # Get the metadata of the output file to determine duration
            probe = ffmpeg.probe(output_file)
            duration = float(probe['streams'][0]['duration'])

            self.log(f"Conversion complete. The file '{output_file}' has been written (source: {input_file})", study_submission_id)
            return duration
        except Exception as e:
            self.log_error(f"An error occurred: {e}", study_submission_id)
            return None


    @property
    def ACS_outcomes_renaming(self):
        return {
            "O_token": "acs_token",

            # main outcomes
            "O_11_time_test_msec": "connect_the_dots_I_time_msec",
            "O_13_time_test_msec": "connect_the_dots_II_time_msec",
            "O_14_correct_t": "wordlist_correct_words",
            "O_17_RT_average_A": "avg_reaction_speed",
            "O_19_extramoves_t": "place_the_beads_total_extra_moves",
            "O_21_correct_t": "box_tapping_total_correct",
            "O_23_time_test_msec": "fill_the_grid_total_time",
            "O_25_correct_t": "wordlist_delayed_correct_words",
            "O_26_correctPandN": "wordlist_recognition_correct_words",
            "O_28_correct_t": "digit_sequence_1_correct_series",
            "O_30_correct_t": "digit_sequence_2_correct_series",
            "O_32_ANX": "questionnaire_hads_anxiety",
            "O_32_DEPR": "questionnaire_hads_depression",
            "O_33_alg_vermoeidh": "questionnaire_mfi_general_fatigue",
            "O_33_lich_vermoeidh": "questionnaire_mfi_physical_fatigue",
            "O_33_red_activit": "questionnaire_mfi_reduced_activity",
            "O_33_red_motivatie": "questionnaire_mfi_reduced_motivation",
            "O_33_ment_vermoeidh": "questionnaire_mfi_mental_fatigue",
            "O_34_computer_experience": "computer_experience",
            "O_34_type": 'mouse_type',

            # extra outcomes
            "connect_the_dots_difference": "connect_the_dots_difference",
            "connect_the_dots_fraction": "connect_the_dots_fraction",
            "digit_sequence_difference": "digit_sequence_difference",
            "digit_sequence_fraction": "digit_sequence_fraction",
            'O_14_learning_curve1': 'wordlist_correct_trial1',
            'O_14_learning_curve2': 'wordlist_correct_trial2',
            'O_14_learning_curve3': 'wordlist_correct_trial3',
            'O_14_learning_curve4': 'wordlist_correct_trial4',
            'O_14_learning_curve5': 'wordlist_correct_trial5',
            'wordlist_learning': 'wordlist_learning',
            'O_19_extramoves_per_trial': "place_the_beads_extramoves_per_trial",
            'O_19_n_perfect_solutions': "place_the_beads_n_perfect_solutions",
            'O_19_n_solved_in_max_moves': "place_the_beads_n_solved_in_max_moves",
            'O_21_span': 'box_tapping_span_2x',
            'O_21_lcs':  'box_tapping_span_1x',
            'O_28_span': 'digit_sequence_1_span_2x',
            'O_28_lcs': 'digit_sequence_1_span_1x',
            'O_30_span': 'digit_sequence_2_span_2x',
            'O_30_lcs': 'digit_sequence_2_span_1x',
            'O_7_LD': 'typeskill_levenstein_distance',
            'O_7_time_test': 'typeskill_time',
            'O_8_time_test': 'clickskill_time',
            'O_8_finished': 'clickskill_finished',
            'O_8_errors_t': 'clickskill_errors',
            'O_8_missed_t': 'clickskill_misses',
            'O_9_time_test': 'dragskill_time',
            'O_9_optimal_t': 'dragskill_optimal_moves',
            'O_9_drops_t': 'dragskill_drops',
            'O_9_finished': 'dragskill_finished',

            'O_14_time_t': 'wordlist_total_time',
            'O_17_SD_A': 'reaction_speed_std',
            'O_17_time_t': 'reaction_speed_time',
            'O_19_tot_time_10': 'place_the_beads_total_time',
            'O_21_time_t': 'box_tapping_total_time',
            'O_25_time_t': 'wordlist_delayed_total_time',
            'O_26_time_t': 'wordlist_recognition_total_time',
            'O_28_time_t': 'digit_sequence_1_total_time',
            'O_30_time_t': 'digit_sequence_2_total_time',
        }

    def prepare_cognitive_scores(self):
        print("Preparing cognitive scores")

        # keep only cognitive scores for this study, by filtering for relevant tokens
        self.cognitive_scores = self.cognitive_scores[self.cognitive_scores['O_token'].isin(self.study_submissions.acs_token)]
        print(f"... {self.cognitive_scores.shape[0]} rows in cognitive scores")

        if self.cognitive_scores.shape[0] < self.study_submissions.shape[0]:
            logging.error("Missing cognitive (ACS) scores for ")

        missing_cognitive_scores = self.study_submissions[~self.study_submissions.acs_token.isin(self.cognitive_scores['O_token'])].copy()
        missing_cognitive_scores = missing_cognitive_scores.merge(self.id_mapping, on=["study_submission_id", "acs_token"], how="left")
        if missing_cognitive_scores.shape[0] > 0:
            self.log_error(f"{missing_cognitive_scores.shape[0]} participants with missing ACS cognitive scores:")
            for _, (study_submission_id, prolific_id, acs_token) in missing_cognitive_scores[['study_submission_id', 'prolific_id', 'acs_token']].iterrows():
                self.log_error(f"Missing ACS cognitive scores for submission {study_submission_id} / {prolific_id} / {acs_token}", study_submission_id)


        # Calculate some additional computed outcomes from raw scores
        def calculate_additional_outcomes(df):
            # O_19_extramoves_t = place the beads - nr moves - nr optimal moves; sum over trials
            # Total number of additional steps required. You must calculate this outcome measure yourself and add it to the data: Calculate for each trial the difference between the number of steps taken and the lowest possible (optimal) number of steps. Take the sum of all trials (O_19_moves_1 - O_19_opt_moves_1 + O_19_moves_2 - O_19_opt_moves_2 + ... + O_19_moves_10 - O_19_opt_moves_10). Note: if not solved within the time (O_19_solved_1 = 0), or needed >20 steps, calculate a maximum score of 20 for the number of steps taken (O_19_moves_1=20).

            # CANTAB also calculates the following outcomes:
            # - % Perfect Solutions
            # - Average Excess moves per trial
            # - % Completed in Maximum moves

            # go through 10 substeps of O_19 (place the beads)
            extra_moves_columns = []
            solved_in_max_moves_columns = []
            for i in range(1, 11):
                is_solved = df[f'O_19_solved_{i}']
                moves = df[f'O_19_moves_{i}']
                optimal_moves = df[f'O_19_opt_moves_{i}']

                new_col = f"0_19_solved_in_max_moves_{i}"
                df[new_col] = np.where((is_solved == 0) | (moves > 20), 0, 1)
                solved_in_max_moves_columns.append(new_col)

                moves_processed = np.where((is_solved == 0) | (moves > 20), 20, moves)
                extra_moves = moves_processed - optimal_moves

                new_col = f'O_19_extramoves_{i}'
                df[new_col] = extra_moves
                extra_moves_columns.append(f'O_19_extramoves_{i}')

            print("     Relevant columns to calculate O_19_extramoves_t (place the beads)", extra_moves_columns)
            df["O_19_extramoves_t"] = df[extra_moves_columns].apply(sum, axis=1)
            df["O_19_extramoves_per_trial"] = df[extra_moves_columns].apply(sum, axis=1) / len(extra_moves_columns)
            df["O_19_n_perfect_solutions"] = (df[extra_moves_columns] == 0).apply(sum, axis=1)
            df["O_19_n_solved_in_max_moves"] = df[solved_in_max_moves_columns].apply(sum, axis=1)

            # O_26_correctPandN = wordlist recognition - nr correct target words + nr correct non-target words
            # You must calculate this outcome measure yourself and add it to the data: O_26_correctP_t + O_26_correctN_t
            df["O_26_correctPandN"] = df['O_26_correctP_t'] + df['O_26_correctN_t']

            # Trails B - Trails A difference. → Thought to be  purer measures of the more complex divided attention and alternating sequencing tasks required in part B
            df['connect_the_dots_difference'] = df['O_13_time_test_msec'] - df['O_11_time_test_msec']
            try:
                df['connect_the_dots_fraction'] = df['O_13_time_test_msec'].astype(float) / df['O_11_time_test_msec']
            except:
                pass

            df['digit_sequence_difference'] = df['O_30_correct_t'] - df['O_28_correct_t']
            try:
                df['digit_sequence_fraction'] = df['O_30_correct_t'].astype(float) / df['O_28_correct_t']
            except:
                pass

            df['wordlist_learning'] = df['O_14_learning_curve5'] - df['O_14_learning_curve1']

            return df

        print("... Calculate additional outputs")
        self.cognitive_scores = calculate_additional_outcomes(self.cognitive_scores)
        cognitive_scores_raw = self.cognitive_scores.copy()

        print("... Rename outcome variables")
        self.cognitive_scores = self.cognitive_scores[self.ACS_outcomes_renaming.keys()].rename(columns=self.ACS_outcomes_renaming)

        # replace acs_token by study_submission_id
        self.cognitive_scores = self.cognitive_scores.merge(self.id_mapping[['acs_token', 'study_submission_id']], on="acs_token", how="left")
        self.cognitive_scores = self.cognitive_scores.drop(columns=['acs_token'])
        self.cognitive_scores = self.cognitive_scores[['study_submission_id'] + [c for c in self.cognitive_scores if c != 'study_submission_id']]

        outcomes_filepath = os.path.join(self.RESULTS_DIR, "acs_outcomes.csv")
        print(f"Writing to {outcomes_filepath}")
        self.cognitive_scores.to_csv(outcomes_filepath, index=False)

        incomplete_cognitive_scores = self.cognitive_scores[self.cognitive_scores.isna().any(axis=1)].copy()
        incomplete_cognitive_scores = incomplete_cognitive_scores.merge(self.id_mapping, on="study_submission_id", how="left")
        if incomplete_cognitive_scores.shape[0] > 0:
            self.log_error(f"{incomplete_cognitive_scores.shape[0]} participants with incomplete ACS cognitive scores:")
            for _, (study_submission_id, prolific_id, acs_token) in incomplete_cognitive_scores[['study_submission_id', 'prolific_id', 'acs_token']].iterrows():
                self.log_error(f"Incomplete ACS cognitive scores for submission {study_submission_id} / {prolific_id} / {acs_token}", study_submission_id)

        self.calculate_norm_scores(self.cognitive_scores)

        return self.cognitive_scores

    def calculate_norm_scores(self, cognitive_scores):
        # norm data, original logic provided by ACS team
        acs_norm_data_calculator = OriginalACSNormDataCalculator(CONSTANTS=self.CONSTANTS, log_problems=True)

        print("Calculating normative scores of ACS")
        data = cognitive_scores.merge(self.demographics, on="study_submission_id", how='left')
        ACS_norm_scores = acs_norm_data_calculator.calculate_norm_scores(data)

        outcomes_filepath = os.path.join(self.RESULTS_DIR, "acs_norm_scores.csv")
        print(f"Writing to {outcomes_filepath}")
        ACS_norm_scores.to_csv(outcomes_filepath, index=False)

        self.plot_cognitive_norm_scores(ACS_norm_scores)

    def plot_cognitive_norm_scores(self, ACS_norm_scores):
        # plot the scores, to detect any outliers
        fig, ax = plt.subplots(figsize=(10, 5))

        def mean_z(row):
            with warnings.catch_warnings():
                warnings.filterwarnings(action='ignore', message='Mean of empty slice')
                nanmean = np.nanmean(row[[c for c in row.index if '_Z' in c]])
            return nanmean

        ACS_norm_scores['mean_z'] = ACS_norm_scores.apply(mean_z, axis=1)
        ACS_norm_scores['sorting'] = ACS_norm_scores['mean_z'].rank()
        ACS_norm_scores = ACS_norm_scores.sort_values(by='sorting')

        for col in [c for c in ACS_norm_scores.columns if "_Z" in c]:
            width = 0.5
            if 'wordlist' in col or 'digit_sequence' in col:
                width = 2
            ax.plot(ACS_norm_scores.sorting, ACS_norm_scores[col], ".-", label=col, linewidth=width)

        ax.set_xlabel("Study submission id")
        ax.set_ylabel("z score")
        ax.legend(ncol=3, fontsize='small')
        ax.set_xticks(ACS_norm_scores.sorting, ACS_norm_scores.study_submission_id, rotation=50, ha="right")
        ax.set_ylim([-4, 6])
        ax.set_title("ACS normative scores for participants")
        plt.tight_layout()
        filepath = os.path.join(self.RESULTS_DIR, "acs_norm_scores.png")
        print(f"Plotting norm scores to {filepath}")
        plt.savefig(filepath)
        plt.close()

    def _check_audio_chunks(self, submission_id, task, audio_id, duration):
        # check if there are more chunks than expected. This usually means there was something wrong
        # while recording, which should be looked at separately.
        if task == 'check':
            return

        chunks_base_dir = os.path.join(self.RAW_DIR, "files.uzh-speech.ch/p/assets/chunks/")
        if not os.path.exists(chunks_base_dir):
            logging.exception("Raw chunk files do not exist. Cannot do checks")
            return
        chunks_for_audio = [file for file in os.listdir(chunks_base_dir) if file.startswith(f"{audio_id}_")]
        n_chunks = len(chunks_for_audio)

        # if all chunks are in the audio, the duration should be around 10 seconds * n_chunks
        if duration * 1.02 < (n_chunks-1) * 10:
            self.log_exception(f"Audio {audio_id} (submission {submission_id}, {task}) has more raw chunks ({n_chunks}) than expected. Expecting between {(n_chunks-1)* 10}s - {n_chunks*10}s, but got {duration} seconds. ", submission_id)

        # if the hash in the chunk name is not always the same, that's also a sign of something wrong
        chunk_hashes = [re.match("[0-9]+_[0-9]+_(.*)\.[a-zA-Z]{2,5}", chunk).group(1) for chunk in chunks_for_audio]
        if len(set(chunk_hashes)) > 1:
            self.log_exception(f"Audio {audio_id} (submission {submission_id}, {task}) has multiple hashes in the chunks ({set(chunk_hashes)}) which is a sign for something going wrong while recording.", submission_id)




    def copy_recorded_data(self):
        print("Preparing data")
        audio_file_base_dir = os.path.join(self.RAW_DIR, "files.uzh-speech.ch/p/assets/public/")
        reyfigure_file_base_dir = os.path.join(self.RAW_DIR, "files.uzh-speech.ch/p/assets/public/reyFigure/")
        for id in self.study_submissions.study_submission_id.sort_values():
            self.log(f"Analyzing submission id {id}")
            data_dir = os.path.join(self.RESULTS_DIR, str(id))
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            submission = self.study_submissions.set_index("study_submission_id").loc[id]
            submission.reset_index().to_csv(os.path.join(data_dir, "submission.csv"), index=False)

            rey_figure_drawings_here = self.rey_figure_drawings.query(f"study_submission_id == {id}")
            for _, rey_figure in rey_figure_drawings_here.iterrows():
                rey_figure_id = rey_figure.id
                rey_figure_filenames = [filename for filename in os.listdir(reyfigure_file_base_dir) if
                                        re.match(str(rey_figure_id) + "\.[a-zA-Z]{3,5}", filename)]
                for filename in rey_figure_filenames:
                    extension = re.match("[0-9]+\.([a-zA-Z]{3,5})", filename).group(1)
                    src_file = os.path.join(reyfigure_file_base_dir, filename)
                    target_file = os.path.join(data_dir, "reyFigure", f"{rey_figure.task}.{extension}")
                    os.makedirs(os.path.dirname(target_file), exist_ok=True)
                    shutil.copyfile(src_file, target_file)

            audio_recordings_here = self.audio_recordings.query(f"study_submission_id == {id}")
            # assert audio_recordings_here.task.shape[0] == audio_recordings_here.task.drop_duplicates().shape[0]
            nth_task_repetition = audio_recordings_here.groupby("task")['created_at'].rank()
            audio_durations = []
            for (_, audio), nth_repetition in zip(audio_recordings_here.iterrows(), nth_task_repetition):
                task, audio_id = audio.task, audio.id
                if not audio.finished:
                    msg = f"Audio {audio_id} not finished?"
                    self.log_exception(msg, id)
                audio_file = os.path.join(audio_file_base_dir, f"{audio_id}.webm")
                audio_target_file = os.path.join(data_dir,
                                                 f"{task}{str(int(nth_repetition)) if nth_repetition > 1 else ''}.wav")
                if audio.task != "check" and nth_repetition > 1:
                    self.log_exception(f"There are {nth_repetition} versions of {task} audio for submission {audio.study_submission_id}", id)
                if not os.path.exists(audio_file):
                    msg = f"Audio file {audio_id} (task {audio.task}, submission {audio.study_submission_id}) does not exist"
                    self.log_exception(msg, id)
                    continue
                # convert to wav and copy to processed dir
                duration = self._convert_webm_to_wav(audio_file, audio_target_file, id)
                self._check_audio_chunks(id, task, audio_id, duration)
                audio_durations.append({'duration': duration, 'task': task})
            audio_durations = pd.DataFrame(audio_durations)
            audio_durations.to_csv(os.path.join(data_dir, "audio_durations.csv"), index=False)
        print("done")
    def run_speech_recognition(self):
        print("Transcribing audio files")
        whisper_transcriber = Whisper_Transcriber(use_vad=True)

        for id in self.study_submissions.study_submission_id.sort_values():
            data_dir = os.path.join(self.RESULTS_DIR, str(id))
            submission = self.study_submissions.set_index("study_submission_id").loc[id]
            # language code based on submission
            if submission['language'] in ['english_american', 'english_other']:
                language_code = 'en-US'
            elif submission['language'] in ['english_british']:
                language_code = 'en-GB'
            else:
                raise ValueError(f"Invalid submission language {submission['language']}")

            google_transcriber = GoogleSpeechTranscriber(language_code=language_code)

            # iterate through all files and transcribe
            transcriptions = []
            for file in os.listdir(data_dir):
                if file.endswith(".wav"):
                    audio_file_path = os.path.join(data_dir, file)
                    audio_duration = librosa.get_duration(path=audio_file_path)

                    google_raw = google_transcriber.transcribe_file(audio_file_path)['combined_transcript']
                    whisper_raw = whisper_transcriber.transcribe_file(audio_file_path)['text']

                    google = google_raw.strip().lower()
                    #google = re.sub('\s+', ' ', google)
                    #google = re.sub('[^A-Za-z0-9 ]+', '', google)
                    whisper = whisper_raw.strip().lower()
                    #whisper = re.sub('\s+', ' ', whisper)
                    #whisper = re.sub('[^A-Za-z0-9 ]+', '', whisper)

                    if audio_duration > 10 * 60:
                        self.log_exception(f"ATTENTION: File {file} for submission {id} is longer than 10 minutes!", id)
                        #continue

                    os.makedirs(os.path.join(data_dir, "ASR", "google_speech"), exist_ok=True)
                    os.makedirs(os.path.join(data_dir, "ASR", "whisper"), exist_ok=True)

                    task = file.split(".")[0]
                    transcription_file_path_google = os.path.join(data_dir, "ASR", "google_speech", task + ".txt")
                    transcription_file_path_whisper = os.path.join(data_dir, "ASR", "whisper", task + ".txt")

                    with open(transcription_file_path_google, "w") as f:
                        f.write(google)
                    with open(transcription_file_path_whisper, "w") as f:
                        f.write(whisper)

                    transcriptions.append({
                        'task': task,
                        'submission': id,
                        'text_google': google,
                        'text_whisper': whisper,
                        'audio_duration': audio_duration
                    })

            pd.DataFrame(transcriptions).to_csv(os.path.join(data_dir, "ASR", "transcriptions.csv"), index=False)
        print("Done transcribing audio files.")

    def prepare_data(self):
        print("Starting data preparation:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.load_raw_data()
        self.prepare_id_mappings()
        self.prepare_prolific_data()
        self.prepare_study_submission_data()
        self.check_demographics_consistency()
        self.prepare_demographics()
        self.descriptive_statistics()
        self.prepare_cognitive_scores()

        self.copy_recorded_data()

        self._run_extra_logic_processor('after_prepare_data')

        self.run_speech_recognition()

        with open(os.path.join(self.RESULTS_DIR, "..", "_status.txt"), "a") as f:
            f.write(f"Data preparatation: Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
