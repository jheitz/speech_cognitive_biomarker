import os
import pandas as pd
import numpy as np
import re
import logging
import sys
import librosa
import time
from datetime import datetime

sys.path.insert(0, '..') # to make the import from parent dir work

from data_preparation.test_scoring.fluency.string_alignment import NeedlemanWunsch

from config.config import Config
from config.constants import Constants
from config.run_parameters import RunParameters


class DataQualityChecker:
    def __init__(self, run_parameters: RunParameters, config: Config, CONSTANTS: Constants):
        self.run_parameters = run_parameters
        self.config = config
        self.CONSTANTS = CONSTANTS

        # results / preprocessed dir
        try:
            self.RESULTS_DIR = os.path.join(run_parameters.results_dir, "data")
        except:
            self.RESULTS_DIR = os.path.join(CONSTANTS.DATA_PROCESSED, config.name, "data")
        assert os.path.exists(self.RESULTS_DIR)

    def load_data(self):
        self.acs_outcomes = pd.read_csv(os.path.join(self.RESULTS_DIR, "acs_outcomes.csv"))
        self.prolific_data = pd.read_csv(os.path.join(self.RESULTS_DIR, "prolific_data.csv"))
        self.study_submissions = pd.read_csv(os.path.join(self.RESULTS_DIR, "study_submissions.csv"))

    def check_number_of_data_rows(self):
        print("Checking number of data rows...")
        n_submissions = self.study_submissions.shape[0]
        n_prolific = self.prolific_data.shape[0]
        if n_prolific != n_submissions:
            logging.error(f"Prolific data has other number of rows than study_submissions: {n_prolific} vs. {n_submissions}")

        n_acs_outcomes = self.acs_outcomes.shape[0]
        if n_acs_outcomes != n_submissions:
            logging.error(f"ACS outcomes data has other number of rows than study_submissions: {n_acs_outcomes} vs. {n_submissions}")

    def check_audio_files(self):
        print("Checking audio files...")
        expected_audio_files = ['check.wav', 'cookieTheft.wav', 'journaling.wav', 'phonemicFluency.wav',
                                'picnicScene.wav', 'pictureNaming.wav', 'semanticFluency.wav']
        for id in self.study_submissions.study_submission_id:
            id_dir = os.path.join(self.RESULTS_DIR, str(id))
            missing_files = [file for file in expected_audio_files if not os.path.exists(os.path.join(id_dir, file))]
            if len(missing_files) > 0:
                logging.warning(f"Missing audio files {missing_files} for submission {id}.")

    def check_audio_durations(self):
        print("Checking audio durations")
        expected_audio_durations = {
            'check': [1, 30],
            'cookieTheft': [45, 10*60],
            'picnicScene': [45, 10*60],
            'phonemicFluency': [58, 62],
            'semanticFluency': [58, 62],
            'pictureNaming': [30, 10*60],
            'journaling': [45, 10*60]
        }
        for id in self.study_submissions.study_submission_id:
            id_dir = os.path.join(self.RESULTS_DIR, str(id))
            audio_durations = pd.read_csv(os.path.join(id_dir, 'audio_durations.csv'))
            audio_durations = audio_durations.dropna()
            invalid_durations = [row for idx, row in audio_durations.iterrows() if not (expected_audio_durations[row.task][0] <= row.duration <= expected_audio_durations[row.task][1])]
            if len(invalid_durations) > 0:
                logging.warning(f"Submission {id} has {len(invalid_durations)} audio files with invalid durations: " + \
                                ", ".join([f"{row.task}: {row.duration} (should be {expected_audio_durations[row.task]})" for row in invalid_durations]))


    def _get_transcripts_info(self, submission_id):
        id_dir = os.path.join(self.RESULTS_DIR, str(submission_id))
        transcript_dir = os.path.join(id_dir, 'ASR')
        transcript_info = pd.read_csv(os.path.join(transcript_dir, "transcriptions.csv"))
        return transcript_info

    def check_silence(self):
        print("Checking VAD / silences in audio files")
        for id in self.study_submissions.study_submission_id:
            transcript_info = self._get_transcripts_info(id)

            # load audio durations when applied VAD (e.g. removed silence)
            VAD_file_dir = os.path.join(self.CONSTANTS.CACHE_DIR, "audio_VAD")
            VAD_audio_durations = []
            for file in os.listdir(VAD_file_dir):
                try:
                    task, submission, filetype = re.match("(.*)_([0-9]+)_VAD\.([a-zA-Z]{3,})", file).group(1,2,3)
                except AttributeError as e:
                    continue
                if filetype == 'wav':
                    audio_duration_VAD = librosa.get_duration(path=os.path.join(VAD_file_dir, file))
                    VAD_audio_durations.append({'task': task, 'submission': int(submission), 'VAD_audio_duration': audio_duration_VAD})

            VAD_audio_durations_df = pd.DataFrame(VAD_audio_durations)

            transcript_info_extended = transcript_info.merge(VAD_audio_durations_df, on=["submission", 'task'], how="left")
            # if VAD_audio_duration is None, this means that the VAD file does not exist, i.e. there were no non-silence segments detected
            transcript_info_extended['VAD_audio_duration'] = transcript_info_extended['VAD_audio_duration'].fillna(0)
            transcript_info_extended['fraction_of_silence'] = (transcript_info_extended['audio_duration'] - transcript_info_extended['VAD_audio_duration']) / transcript_info_extended['audio_duration']

            spontaneous_speech_extensive_silence = (transcript_info_extended['fraction_of_silence'] > 0.5) & (transcript_info_extended['task'].isin(['journaling', 'picnicScene', 'cookieTheft']))
            language_task_extensive_silence = (transcript_info_extended['fraction_of_silence'] > 0.8) & (transcript_info_extended['task'].isin(['phonemicFluency', 'semanticFluency', 'pictureNaming']))
            extensive_silence = transcript_info_extended[spontaneous_speech_extensive_silence | language_task_extensive_silence]
            if extensive_silence.shape[0] > 0:
                extensive_silence_desc = ", ".join([f'{task} ({frac*100:.1f}%)' for task, frac in zip(extensive_silence.task.to_list(), extensive_silence.fraction_of_silence.to_list())])
                logging.warning(f"Extensive silence for submission {extensive_silence.submission.iloc[0]}: {extensive_silence_desc}")

    def check_repeated_audio_chunks(self):
        # check if the first audio chunk is repeated, i.e. if the signal repeats after ~10sec
        print("Checking for repeated audio chunks...")
        for id in self.study_submissions.study_submission_id:
            id_dir = os.path.join(self.RESULTS_DIR, str(id))
            #print(id, end=" ")
            start_time = time.time()
            for filename in os.listdir(id_dir):
                if filename.endswith("wav") and not filename.startswith("check"):
                    audio_file_path = os.path.join(id_dir, filename)
                    signal, sample_rate = librosa.load(audio_file_path, sr=1000)

                    window_size_seconds = 8
                    window_size = sample_rate * window_size_seconds
                    first_10sec = signal[:window_size]

                    norm_first_10sec = np.linalg.norm(first_10sec)
                    def vector_similarity(window):
                        return np.dot(window, first_10sec) / (np.linalg.norm(window) * norm_first_10sec)

                    start_obervation_sec = 9
                    end_obervation_sec = 11
                    relevant_signal_section = signal[int(sample_rate * start_obervation_sec):int(sample_rate * (end_obervation_sec + window_size_seconds))]
                    similarity = pd.Series(relevant_signal_section).rolling(window=window_size).apply(vector_similarity)
                    max_similarity = np.max(np.abs(similarity))
                    if max_similarity > 0.5:
                        logging.error(f"Repeated audio chunk for {audio_file_path}?")

            #print(f" ({(time.time() - start_time)}s)")

    def check_transcripts(self):
        print("Checking transcripts for audio files")
        for id in self.study_submissions.study_submission_id.sort_values():
            transcript_info = self._get_transcripts_info(id)

            # align google and whisper transcription, to check difference as a metric of quality
            string_aligner = NeedlemanWunsch()
            def preprocess_transcription_for_alignment(transcript):
                try:
                    transcript = re.sub('\s+', ' ', transcript)
                    transcript = re.sub('[^A-Za-z0-9 ]+', '', transcript)
                except:
                    pass
                return transcript
            def string_distance_metric(row):
                whisper = preprocess_transcription_for_alignment(row.text_whisper)
                google = preprocess_transcription_for_alignment(row.text_google)
                try:
                    _, _, rx, ry, counts = string_aligner.align_strings(whisper, google, return_details=True)
                except:
                    return pd.Series([None, None])
                total_count = sum([counts[c] for c in counts if c != 'match'])
                frac = total_count / max(len(row.text_whisper), len(row.text_google))
                return pd.Series([total_count, frac])
            transcript_info[['transcript_distance', 'transcript_distance_rel']] = transcript_info.apply(string_distance_metric, axis=1)

            large_transcript_diff = transcript_info[(transcript_info.transcript_distance_rel > 0.2) & (transcript_info.task != 'check')]
            if large_transcript_diff.shape[0] > 0:
                extensive_silence_desc = ", ".join([f'{task} ({frac*100:.1f}% difference)' for task, frac in zip(large_transcript_diff.task.to_list(), large_transcript_diff.transcript_distance_rel.to_list())])
                logging.warning(f"Large difference between transcripts (low quality?) for submission {large_transcript_diff.submission.iloc[0]}: {extensive_silence_desc}")

    def check(self):
        print("Starting data quality check:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.load_data()
        self.check_number_of_data_rows()
        self.check_audio_files()
        self.check_audio_durations()
        self.check_silence()
        self.check_repeated_audio_chunks()
        self.check_transcripts()

        with open(os.path.join(self.RESULTS_DIR, "..", "_status.txt"), "a") as f:
            f.write(f"Data quality check: Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

