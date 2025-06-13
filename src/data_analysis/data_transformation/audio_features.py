import os
import pandas as pd
import numpy as np
import opensmile

from data_analysis.data_transformation.data_transformer import DataTransformer
from data_analysis.dataloader.dataset import Dataset
from data_analysis.data_transformation.helpers.wave2vec2_phoneme_transcriber import Wave2Vec2PhonemeTranscriber
from data_analysis.data_transformation.helpers.voice_activity_detector import VoiceActivityDetector
from data_analysis.util.decorators import cache_to_file_decorator


class AudioFeatures(DataTransformer):
    """
    Features extracted from the audio
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Audio Features"
        print(f"Initializing {self.name}")

        self.voice_activity_detector = VoiceActivityDetector(debug=False)

        self.phoneme_transcriber = Wave2Vec2PhonemeTranscriber(config=self.config, constants=self.CONSTANTS)

        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

        # feature groups: which audio features to load
        valid_feature_groups = ['pause_features',
                                'phoneme_features',
                                'opensmile_features']
        try:
            self.feature_groups = self.config.config_audio_features.feature_groups
        except (AttributeError, KeyError):
            self.feature_groups = ['pause_features', 'phoneme_features', 'opensmile_features']
        assert all([fg in valid_feature_groups for fg in self.feature_groups])

        try:
            self.selected_features = self.config.config_audio_features.selected_features
        except (AttributeError, KeyError):
            self.selected_features = []  # empty list -> select all

        valid_feature_versions = ['full', 'reduced']
        try:
            self.feature_version = self.config.config_audio_features.feature_version
        except (AttributeError, KeyError):
            self.feature_version = 'full'
        assert self.feature_version in valid_feature_versions, \
            f"Invalid feature version {self.feature_version}, should be in {valid_feature_versions}"

        print(f"... using audio feature_groups {self.feature_groups}, feature_version {self.feature_version}, "
              f"selected_features {self.selected_features}")


    def _get_voice_activation_segments(self, audio_path):
        def remove_leading_and_trailing_pause_segments(df):
            first_valid_index = df[df['type'] != 'pause'].index.min()
            df = df.loc[first_valid_index:]
            last_valid_index = df[df['type'] != 'pause'].index.max()
            df = df.loc[:last_valid_index]
            return df

        segments_df = self.voice_activity_detector.get_segments(audio_path)
        segments_df = remove_leading_and_trailing_pause_segments(segments_df)

        return segments_df

    def _pause_features_one(self, segments_df):
        features = {}

        total_pause_duration = segments_df.query("type == 'pause'").duration.sum()
        total_voice_duration = segments_df.query("type == 'voice'").duration.sum()

        features['audio_length'] = (total_voice_duration + total_pause_duration)

        features['fraction_of_pause'] = total_pause_duration / (total_voice_duration + total_pause_duration)
        features['mean_pause_duration'] = segments_df.query("type == 'pause'").duration.mean()
        features['std_pause_duration'] = segments_df.query("type == 'pause'").duration.std()
        if np.isnan(features['std_pause_duration']):
            features['std_pause_duration'] = 0

        features['n_pauses'] = segments_df.query("type == 'pause'").shape[0]

        # bins according to https://www.mdpi.com/2076-3417/13/7/4244
        count_pause_lengths, _ = np.histogram(segments_df.query("type == 'pause'").duration,
                                              bins=[0, 0.5, 1, 2, 4, np.inf])
        features['n_pauses_0-05'] = count_pause_lengths[0]
        features['n_pauses_05-1'] = count_pause_lengths[1]
        features['n_pauses_1-2'] = count_pause_lengths[2]
        features['n_pauses_2-4'] = count_pause_lengths[3]
        features['n_pauses_4+'] = count_pause_lengths[4]

        # add "pause_" prefix to make clear where these features are from
        features = {f"pause_{f}": features[f] for f in features}

        # print("Extracted pause features:", [f for f in features.keys()])

        return features

    def _phoneme_features_one(self, phoneme_transcription, segments_df, pause_features):
        # features inspired by Toth et al. (2018): A Speech Recognition-based Solution for the Automatic Detection of Mild Cognitive Impairment from Spontaneous Speech

        features = {}
        features['n_phonemes'] = len(phoneme_transcription.replace(" ", ""))
        features['speech_rate'] = features['n_phonemes'] / pause_features['pause_audio_length']

        speech_length = (1-pause_features['pause_fraction_of_pause']) * pause_features['pause_audio_length']
        features['articulation_rate'] = features['n_phonemes'] / speech_length

        # add "phon_" prefix to make clear where these features are from
        features = {f"phon_{f}": features[f] for f in features}

        # print("Extracted phoneme features:", [f for f in features.keys()])

        return features

    @cache_to_file_decorator(n_days=30)
    def _smile_features_one(self, audio_path):
        print(f"Calculating Smile features for {audio_path}")
        processed = self.smile.process_file(audio_path)
        feats_df = np.array(processed)[0]
        feature_names = processed.columns.values

        features_audio = {}
        for n, name in enumerate(feature_names):
            features_audio[name] = feats_df[n]

        return features_audio

    def _postprocess_smile_features_one(self, smile_features):
        if self.feature_version == 'reduced':
            # let's select only a subset of features, based on their usefulness for the task
            # and their correlation
            # see /analyses/kw49/reduced_features/opensmile_features_reduce_number.ipynb
            features_to_keep = ['loudnessPeaksPerSec', 'spectralFlux_sma3_stddevNorm', 'MeanUnvoicedSegmentLength',
                                'loudness_sma3_percentile50.0', 'F1amplitudeLogRelF0_sma3nz_amean',
                                'spectralFlux_sma3_amean', 'spectralFluxUV_sma3nz_amean', 'VoicedSegmentsPerSec',
                                'loudness_sma3_percentile20.0', 'equivalentSoundLevel_dBp', 'mfcc3_sma3_amean',
                                'mfcc1V_sma3nz_stddevNorm', 'alphaRatioUV_sma3nz_amean',
                                'F1bandwidth_sma3nz_stddevNorm', 'loudness_sma3_percentile80.0',
                                'loudness_sma3_meanRisingSlope', 'spectralFluxV_sma3nz_stddevNorm',
                                'alphaRatioV_sma3nz_stddevNorm', 'F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope',
                                'mfcc1_sma3_stddevNorm', 'mfcc3_sma3_stddevNorm']
            assert all([f in smile_features.keys() for f in features_to_keep]), \
                f"Features {[f for f in features_to_keep if f not in smile_features.keys()]} not in {smile_features.keys()}"
            smile_features = {f: smile_features[f] for f in smile_features if f in features_to_keep}

        # add "smile_" prefix to make clear these are opensmile features
        smile_features = {f"smile_{f}": smile_features[f] for f in smile_features}

        # print("Extracted smile features:", [f for f in smile_features.keys()])

        return smile_features

    def _postprocess_pause_phoneme_features(self, features):
        if self.feature_version == 'reduced':
            # let's select only a subset of features, in order to recude high-corrolation features
            # see /analyses/kw49/reduced_features/Rationale.md
            features_to_keep = ['pause_mean_pause_duration', 'pause_std_pause_duration', 'pause_fraction_of_pause', 'phon_speech_rate', 'phon_articulation_rate']
            features = {f: features[f] for f in features if f in features_to_keep}
        return features

    def _load_features_for_audio(self, path, phoneme_transcription):
        voice_activation_segments = self._get_voice_activation_segments(path)

        features = {}
        if 'pause_features' in self.feature_groups:
            pause_features = self._pause_features_one(voice_activation_segments)
            features = {**features, **pause_features}
        if 'phoneme_features' in self.feature_groups:
            phoneme_features = self._phoneme_features_one(phoneme_transcription, voice_activation_segments, pause_features)
            features = {**features, **phoneme_features}
        features = self._postprocess_pause_phoneme_features(features)
        if 'opensmile_features' in self.feature_groups:
            opensmile_features = self._smile_features_one(path)
            opensmile_features = self._postprocess_smile_features_one(opensmile_features)
            features = {**features, **opensmile_features}

        return features

    def _load_features(self, dataset: Dataset):
        assert 'audio_files' in dataset.data_variables, "Dataset should have audio file variable for audio features"

        # phoneme transcriptions for phoneme features
        phoneme_transcriptions = self.phoneme_transcriber.transcribe_dataset(dataset)

        features = [self._load_features_for_audio(path, phoneme_transcriptions) if not pd.isna(path) else {}
                    for path, phoneme_transcriptions in zip(dataset.audio_files, phoneme_transcriptions)]

        features_df = pd.DataFrame(features)

        # select specific subset of features
        if len(self.selected_features) > 0:
            non_existing = [f for f in self.selected_features if f not in features_df.columns]
            if len(non_existing) > 0:
                print("Warning (Audio features): the following features requested in the config file have do not exist:", non_existing)
            features_df = features_df[[c for c in features_df.columns if c in self.selected_features]]

        config_without_transformers = {key: dataset.config[key] for key in dataset.config if key != 'data_transformers'}
        new_config = {
            'data_transformers': [*dataset.config['data_transformers'], self.name],
            **config_without_transformers
        }

        new_dataset = dataset.copy()
        new_dataset.name = f"{dataset.name} - {self.name}"
        new_dataset.config = new_config

        new_dataset.features = self._create_new_feature_df(new_dataset, features_df)

        return new_dataset

    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        print(f"Calculating audio features for dataset {dataset}")
        return self._load_features(dataset)

