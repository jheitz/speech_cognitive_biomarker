import os

import numpy as np
import pandas as pd
from pydub import AudioSegment
import functools

from config.constants import Constants
from data_analysis.dataloader.dataset import Dataset, DatasetType
from data_analysis.util.decorators import cache_to_file_decorator
from util.helpers import hash_list

from data_analysis.dataloader.ADReSS_logic.dataloader import ADReSSWithPITTDataLoader as OriginalADReSSWithPITTDataLoader
from data_analysis.dataloader.ADReSS_logic.audio_cutter import AudioCutter
from util.google_speech_transcription import GoogleSpeechTranscriber

class DataLoader:
    def __init__(self, debug=False, local=None, constants=None, config=None, name="LUHA dataloader"):
        self.debug = debug
        self.name = name
        if constants is None:
            self.CONSTANTS = Constants(local=local)
        else:
            self.CONSTANTS = constants
        self.config = config

        try:
            self.transcript_version = self.config.config_data.transcript_version
        except (AttributeError, KeyError):
            self.transcript_version = 'google'
        valid_transcript_versions = ['google', 'whisper']
        assert self.transcript_version in valid_transcript_versions, f"Invalid transcript version {self.transcript_version}. Should be in {valid_transcript_versions}"

        try:
            self.task = self.config.config_data.task
        except (AttributeError, KeyError):
            self.task = 'pictureDescription'
            print(f"No task specified, loading {self.task}")
        valid_tasks = ['cookieTheft', 'journaling', 'phonemicFluency', 'picnicScene', 'semanticFluency', 'pictureDescription', 'pictureNaming']
        assert not isinstance(self.task, list), "Task is a list. This can be a valid config for run_multiple.py, which runs multiple setups with different tasks. Here (run.py / main.py), you can only have one task"
        assert self.task in valid_tasks, f"Invalid task {self.task}. Should be in {valid_tasks}"

        try:
            self.data_split = str(self.config.config_data.split)
        except (AttributeError, KeyError):
            self.data_split = 'train'
            print(f"No data split specified, loading train participants")
        valid_data_split = ['train', 'test', 'full', '1', '2']
        assert self.data_split in valid_data_split, f"Invalid task {self.data_split}. Should be in {valid_data_split}"

        try:
            self.consent_filter = self.config.config_data.consent_filter
        except (AttributeError, KeyError):
            self.consent_filter = 'full'  # full dataset
            print(f"No consent_filter specified (consent for further data use), loading all participants")
        consent_filters = ['full', 'futher_use_with_audio', 'further_use_without_audio']
        assert self.consent_filter in consent_filters, f"Invalid transcript version {self.consent_filter}. Should be in {consent_filters}"
        consent_filter_text = f", {self.consent_filter}" if self.consent_filter != 'full' else ""

        self.processed_data_dir = os.path.join(self.CONSTANTS.DATA_PROCESSED_COMBINED, "data")

        split_text = f" Split {self.data_split}" if self.data_split is not None else ""
        print(f"Initializing dataloader {self.name} (transcript_version {self.transcript_version}, task {self.task}{consent_filter_text}){split_text}")

    def _preprocess_pandas_data(self, df, sample_names):
        """
        Preprocess a dataframe with sample-level data
        - replace study_submission_id by sample_name, if present
        - sort by external sample_names list
        """
        df = df.copy().rename(columns={"study_submission_id": "sample_name"})
        df['sample_name'] = df['sample_name'].astype(str)
        assert len(sample_names) == df.shape[0], f"{len(sample_names)} vs. {df.shape[0]}: Diff: {[s for s in sample_names if s not in df.sample_name.to_list()]} / {[s for s in df.sample_name if s not in sample_names]}"
        assert len(set(sample_names)) == len(sample_names)
        assert all([s in sample_names for s in df['sample_name']])
        return df.sort_values(by="sample_name", key=lambda sn: sn.map(lambda e: sample_names.index(e))).reset_index(drop=True)

    def _prepare_cognitive_overall_score(self, acs_scores, language_task_scores):
        df = pd.merge(acs_scores, language_task_scores, how='outer', on="sample_name").copy()
        language_task_score_names = ['phonemic_fluency_score', 'semantic_fluency_score', 'picture_naming_score']
        cognitive_cols = self.CONSTANTS.ACS_MAIN_OUTCOME_VARIABLES + language_task_score_names
        for col in cognitive_cols:
            df.loc[:, col] = (df.loc[:, col] - df.loc[:, col].mean()) / df.loc[:, col].std()  # standardize cols
        mean = df[cognitive_cols].mean(axis=1)
        return pd.DataFrame({'cognition_mean': mean, 'sample_name': df.sample_name})

    def _split_transcripts(self, transcriptions_df, version):
        assert version in ['google', 'whisper']
        relevant_cols = ['task', 'sample_name', f"text_{version}"]
        df_version = transcriptions_df[relevant_cols].rename(columns={f'text_{version}': 'transcription'})
        df_version_cleaned = df_version[~df_version.task.str.contains("check")]
        pivoted = df_version_cleaned.pivot(index='sample_name', columns='task', values="transcription").reset_index()
        return pivoted

    @cache_to_file_decorator(n_days=10, verbose=False)
    def _concatenate_audio_files(self, list_of_audio_paths):
        # concatenate multiple audio files into one new audio file (with 1sec pause in between)
        # the new file is written to the cache directory

        sounds = [AudioSegment.from_file(path, format="wav") for path in list_of_audio_paths]
        concatenated = functools.reduce(lambda sound1, sound2: sound1 + AudioSegment.silent(duration=1000) + sound2, sounds)

        basenames = [os.path.basename(p).replace(".wav", "") for p in list_of_audio_paths]
        hash = hash_list(list_of_audio_paths, hash_len=10)
        new_basename = "+".join(basenames) + "_" + hash + ".wav"
        output_dir = os.path.join(self.CONSTANTS.CACHE_DIR_CENTRALIZED, "concatenated_audio")
        os.makedirs(output_dir, exist_ok=True)
        new_path = os.path.join(output_dir, new_basename)

        with open(new_path, "wb") as f:
            concatenated.export(f, format="wav")

        return new_path

    def load_data(self):
        print(f"Loading data using dataloader {self.name}")

        audio_files = []
        sample_names = []
        transcriptions = []
        for i, file_name in enumerate(sorted(os.listdir(self.processed_data_dir), key=lambda id_str: id_str.zfill(4))):
            if os.path.isdir(os.path.join(self.processed_data_dir, file_name)):
                sample_dir = os.path.join(self.processed_data_dir, file_name)
                sample_name = file_name
                sample_audio_files = {'sample_name': sample_name}
                for j, file_name in enumerate(os.listdir(sample_dir)):
                    if file_name.endswith('.wav'):
                        if 'check' in file_name:
                            continue  # ignore check files
                        task_name = file_name.replace(".wav", "")
                        file_path = os.path.join(sample_dir, file_name)
                        assert task_name not in sample_audio_files, \
                            f"Duplicate task {task_name} for sample {sample_name}?"
                        sample_audio_files[task_name] = file_path
                audio_files.append(sample_audio_files)
                sample_names.append(sample_name)

                sample_transcriptions = pd.read_csv(os.path.join(sample_dir, "ASR", "transcriptions.csv"))
                sample_transcriptions = sample_transcriptions.rename(columns={'submission': 'sample_name'})
                transcriptions.append(sample_transcriptions)

        audio_files_df = pd.DataFrame(audio_files)

        transcriptions_df = pd.concat(transcriptions)
        transcriptions_google = self._split_transcripts(transcriptions_df, 'google')
        transcriptions_google = self._preprocess_pandas_data(transcriptions_google, sample_names)
        transcriptions_whisper = self._split_transcripts(transcriptions_df, 'whisper')
        transcriptions_whisper = self._preprocess_pandas_data(transcriptions_whisper, sample_names)

        acs_outcomes_raw = pd.read_csv(os.path.join(self.processed_data_dir, "acs_outcomes_raw.csv"))
        acs_outcomes_raw = self._preprocess_pandas_data(acs_outcomes_raw, sample_names)
        acs_outcomes_imputed = pd.read_csv(os.path.join(self.processed_data_dir, "acs_outcomes_imputed.csv"))
        acs_outcomes_imputed = self._preprocess_pandas_data(acs_outcomes_imputed, sample_names)
        demographics = pd.read_csv(os.path.join(self.processed_data_dir, "demographics.csv"))
        demographics = self._preprocess_pandas_data(demographics, sample_names)
        language_task_scores = pd.read_csv(os.path.join(self.processed_data_dir, "language_task_scores.csv"))
        language_task_scores = self._preprocess_pandas_data(language_task_scores, sample_names)
        factor_scores = pd.read_csv(os.path.join(self.processed_data_dir, "factor_scores.csv"))
        factor_scores = factor_scores.rename(columns={c: f"factor_{c}" for c in factor_scores.columns if c != 'study_submission_id'})
        factor_scores = self._preprocess_pandas_data(factor_scores, sample_names)

        # versions of factor scores / composite scores: check the /factor_analysis/R/2_CFA_theory-driven.R for the logic
        try:
            factor_scores_theory_version = self.config.config_data.factor_scores_theory_version
        except:
            factor_scores_theory_version = '2025-01-07-1929'
        print(f"Using theory factor scores version {factor_scores_theory_version}")
        factor_scores_theory = pd.read_csv(os.path.join(self.CONSTANTS.RESOURCES_DIR, f"factor_scores_theory_{factor_scores_theory_version}.csv"))
        composite_score_columns = [c for c in factor_scores_theory.columns if c != 'study_submission_id']
        print("Standardizing theory factor scores to zero mean and unit variance. This makes comparison of certain evaluation metrics, beta coefficients etc. more interpretable.")
        factor_scores_theory[composite_score_columns] = factor_scores_theory[composite_score_columns].apply(lambda col: (col - np.mean(col)) / np.std(col))
        factor_scores_theory = factor_scores_theory.rename(columns={c: f"composite_{c}" for c in composite_score_columns})
        factor_scores_theory = self._preprocess_pandas_data(factor_scores_theory, sample_names)
        mean_composite_cognitive_score = pd.DataFrame({
            'mean_composite_cognitive_score': factor_scores_theory[[c for c in factor_scores_theory.columns if 'composite_' in c]].mean(axis=1),
            'sample_name': factor_scores_theory.sample_name
        })
        mean_composite_cognitive_score =  self._preprocess_pandas_data(mean_composite_cognitive_score, sample_names)
        cognitive_overall_score = self._prepare_cognitive_overall_score(acs_outcomes_imputed, language_task_scores)
        cognitive_overall_score =  self._preprocess_pandas_data(cognitive_overall_score, sample_names)

        if self.transcript_version == 'google':
            transcripts = transcriptions_google
        elif self.transcript_version == 'whisper':
            transcripts = transcriptions_whisper
        else:
            raise ValueError(f"Invalid transcript version {self.transcript_version}")

        if self.task == 'pictureDescription':
            # combine cookieTheft and picnicScene transcripts
            print("Combining cookieTheft and picnicScene to get pictureDescription...", end=" ")
            def combine_cols_transcripts(row):
                if pd.isna(row['cookieTheft']) or pd.isna(row['picnicScene']):
                    return None
                return f"{row.cookieTheft}\n {row.picnicScene}"
            transcripts_numpy = transcripts.apply(combine_cols_transcripts, axis=1).to_numpy()
            def combine_cols_audio(row):
                if pd.isna(row['cookieTheft']) or pd.isna(row['picnicScene']):
                    return None
                return self._concatenate_audio_files([row['cookieTheft'], row['picnicScene']])
            audio_files_for_task = audio_files_df.apply(combine_cols_audio, axis=1).to_numpy()
            print("... done.")
        else:
            transcripts_numpy = transcripts[self.task].to_numpy()
            audio_files_for_task = audio_files_df[self.task].to_numpy()

        # Store train / test data split into dataset, so we can retrieve it in the model phase if needed
        print(f"Loading train/test set from {self.CONSTANTS.TRAIN_TEST_DATASPLIT}")
        datasplit_assignment = pd.read_csv(self.CONSTANTS.TRAIN_TEST_DATASPLIT)
        # add sample_names without a split, because they have missing information (first 4) or missing cognitive scores (remaining)
        study_submission_ids_to_add = ['172', '488', '631', '707',
                                       '41', '43', '44', '46', '49', '50', '54', '56', '59', '61', '99', '253', '303', '1079']
        study_submission_assignments_to_add = pd.DataFrame({
                'study_submission_id': study_submission_ids_to_add,
                'split': ['' for _ in range(len(study_submission_ids_to_add))],
            })
        overlap = set(datasplit_assignment.study_submission_id).intersection(set(study_submission_assignments_to_add.study_submission_id))
        assert len(overlap) == 0, f"There already is a train/test assignment already for study_submissions {overlap}"
        datasplit_assignment = pd.concat((
            datasplit_assignment,
            study_submission_assignments_to_add
        )).reset_index(drop=True)
        datasplit_assignment = self._preprocess_pandas_data(datasplit_assignment, sample_names).split.to_numpy()


        dataset_name = f"LUHA data ({self.transcript_version} {self.task})"
        dataset = Dataset(name=dataset_name, type=DatasetType.LUHA, sample_names=np.array(sample_names),
                          config={'data_transformers': [], 'debug': self.debug},
                          audio_files=audio_files_for_task,
                          acs_outcomes_imputed=acs_outcomes_imputed, acs_outcomes_raw=acs_outcomes_raw,
                          demographics=demographics, language_task_scores=language_task_scores,
                          factor_scores=factor_scores, factor_scores_theory=factor_scores_theory,
                          cognitive_overall_score=cognitive_overall_score, transcriptions_google=transcriptions_google,
                          transcriptions_whisper=transcriptions_whisper, transcripts=transcripts_numpy,
                          data_split=datasplit_assignment,
                          mean_composite_cognitive_score=mean_composite_cognitive_score)

        if self.data_split == 'full':
            # the full data should only be used in a train/test setting, not for cross-validation
            try:
                cv_splits = self.config.config_model.cv_splits
            except:
                cv_splits = None
            assert cv_splits == 1, f"Full dataset should only be used in a train/test setup, but cv_splits={cv_splits}"

        else:
            if self.data_split in ['1', '2']:
                # Data split 1 or 2. This is a split in two stratified halves of the data
                # For the logic, check /analyses/kw26/data_split_for_factor_analysis.ipynb
                assignment = pd.read_csv(self.CONSTANTS.SPLIT1_SPLIT2_DATASPLIT)
                # Remove two submissions that are no longer part of the dataset (due to missing scores, decided after calculating the data split)
                assignment = assignment[~assignment.study_submission_id.isin([1279, 1303, 1333])]
                sample_names = assignment[assignment.split.astype(str) == self.data_split].study_submission_id
            elif self.data_split in ['train', 'test']:
                # This is a newer data split into train and test (80% / 20%)
                # Train should be used for training and model selection, test only once at the end for reporting
                # For the logic, check /analyses/kw47/train_test_split.ipynb
                assignment = pd.read_csv(self.CONSTANTS.TRAIN_TEST_DATASPLIT)
                sample_names = assignment[assignment.split == self.data_split].study_submission_id

            else:
                raise ValueError()

            dataset = dataset.subset_from_sample_names(sample_names)
            dataset.name = f"{dataset_name} - Split {self.data_split}"

        if self.consent_filter != 'full':
            # filter out samples that do not have the consent filter
            if self.consent_filter == 'futher_use_with_audio':
                sample_names = dataset.sample_names[dataset.demographics.consent_data_further_use == "Yes"]
            elif self.consent_filter == 'further_use_without_audio':
                sample_names = dataset.sample_names[dataset.demographics.consent_data_further_use.isin(["Yes", "Yes Except Recordings"])]
            else:
                raise ValueError(f"Invalid consent filter {self.consent_filter}")
            dataset = dataset.subset_from_sample_names(sample_names)
            dataset.name = f"{dataset_name} - Consent Filter {self.data_split}"



        if self.debug:
            n_samples = self.debug if self.debug > 1 else 10
            dataset = dataset.subset_from_indices(list(range(n_samples)))

        return dataset



class ADReSSDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        kwargs['name'] = 'ADReSS Dataloader'
        super().__init__(*args, **kwargs)
        self.original_dataloader = OriginalADReSSWithPITTDataLoader(constants=self.CONSTANTS)
        self.google_transcriber = GoogleSpeechTranscriber()

        # only_par => only consider participant speech, removing any interview interactions
        try:
            self.only_PAR = self.config.config_data.only_PAR
        except (AttributeError, KeyError):
            self.only_PAR = True


    def _get_google_transcript(self, path):
        google_raw = self.google_transcriber.transcribe_file(path)['combined_transcript']
        google = google_raw.strip().lower()
        return google

    def load_data(self):
        train, test = self.original_dataloader.load_data()
        combined = train.concatenate(test)

        if self.only_PAR:
            audio_cutter = AudioCutter(config=self.config, constants=self.CONSTANTS)
            combined = audio_cutter.preprocess_dataset(combined)

        paths = combined.data

        assert self.transcript_version == 'google'
        assert self.task == 'cookieTheft'
        transcripts = [self._get_google_transcript(path) for path in combined.data]

        # prepare demographics
        demographics = combined.demographics.rename(columns={'gender': 'gender_unified'}).drop(columns=['mmse'])
        demographics['country'] = 1  # usa
        demographics['education_binary'] = 0.5  # in between


        dataset_name = f"ADReSS data ({self.transcript_version} {self.task})"
        dataset = Dataset(name=dataset_name, type=DatasetType.ADReSS, sample_names=np.array(combined.sample_names),
                          config={'data_transformers': [], 'debug': self.debug},
                          audio_files=paths,
                          transcripts=np.array(transcripts),
                          demographics=demographics,
                          classification_target=combined.labels, mmse=np.array(combined.demographics.mmse))

        if self.debug:
            n_samples = self.debug if self.debug > 1 else 10
            dataset = dataset.subset_from_indices(list(range(n_samples)))

        return dataset



