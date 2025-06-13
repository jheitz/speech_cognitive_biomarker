import os
import pandas as pd
import numpy as np
import opensmile
import torch
import re
import torchaudio
from typing import List
import matplotlib.pyplot as plt
from torchaudio.pipelines import MMS_FA as bundle
from datetime import datetime
import matplotlib.patches as patches

from data_analysis.data_transformation.data_transformer import DataTransformer
from data_analysis.dataloader.dataset import Dataset
from data_analysis.data_transformation.helpers.voice_activity_detector import VoiceActivityDetector
from data_analysis.util.decorators import cache_to_file_decorator


class ForcedAlignmentWordTimestamps(DataTransformer):
    """
    Force align audio and transcript using torchaudio
    Based on https://pytorch.org/audio/stable/tutorials/forced_alignment_for_multilingual_data_tutorial.html
    Which is based on this paper: https://arxiv.org/abs/2305.13516
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Word Timestamps (force aligned)"
        print(f"Initializing {self.name}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device {self.device}")

        self.model = bundle.get_model()
        self.model.to(self.device)

        self.tokenizer = bundle.get_tokenizer()
        self.aligner = bundle.get_aligner()

        self.voice_activity_detector = VoiceActivityDetector(debug=False)

    def _compute_alignments(self, waveform: torch.Tensor, transcript: List[str]):
        with torch.inference_mode():
            emission, _ = self.model(waveform.to(self.device))
            token_spans = self.aligner(emission[0], self.tokenizer(transcript))
        return emission, token_spans

    def _score(self, spans):
        # Compute average score weighted by the span length
        return sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans)

    def _plot_alignments(self, waveform, token_spans, emission, transcript, sample_rate=bundle.sample_rate,
                         voice_activity_segments=None, plot_path=None):
        ratio = waveform.size(1) / emission.size(1) / sample_rate
        audio_length = waveform.size(1) / sample_rate

        fig, axes = plt.subplots(3, 1, figsize=(audio_length/90*60, 10), height_ratios=(3,3,1))
        axes[0].imshow(emission[0].detach().cpu().T, aspect="auto")
        axes[0].set_title("Emission")
        axes[0].set_xticks([])

        axes[1].specgram(waveform[0], Fs=sample_rate)
        for t_spans, chars in zip(token_spans, transcript):
            t0, t1 = t_spans[0].start, t_spans[-1].end
            axes[0].axvspan(t0 - 0.5, t1 - 0.5, facecolor="None", hatch="/", edgecolor="white")
            axes[1].axvspan(ratio * t0, ratio * t1, facecolor="None", hatch="/", edgecolor="white")
            axes[1].annotate(f"{self._score(t_spans):.2f}", (ratio * t0, sample_rate * 0.51), annotation_clip=False)

            for span, char in zip(t_spans, chars):
                t0 = span.start * ratio
                axes[1].annotate(char, (t0, sample_rate * 0.55), annotation_clip=False)

        axes[1].set_xlabel("time [second]")
        axes[1].set_xticks(range(int(waveform.size(1) / sample_rate)))

        if voice_activity_segments is not None:
            axes[2].set_xticks(range(int(waveform.size(1) / sample_rate)))
            [axes[2].add_patch(patches.Rectangle((segment.start, 0), segment.duration, 1, facecolor='blue')) for
             idx, segment in voice_activity_segments.query("type == 'voice'").iterrows()]
            axes[2].set_xlim(axes[1].get_xlim())

        fig.tight_layout()
        if plot_path is not None:
            plt.savefig(plot_path)
        plt.close()

    def _normalize_transcript(self, text):
        def _convert_number_to_words(match):
            number = int(match.group(1))
            translation = {10: 'ten', 11: 'eleven', 12: 'twelve', 20: 'twenty', 30: 'thirty', 40: 'forty', 50: 'fifty',
                           60: 'sixty', 70: 'seventy', 80: 'eighty', 90: 'ninety', 100: 'hundred',
                           1920: 'nineteen twenty', 1930: 'nineteen thirty', 1940: 'nineteen forty',
                           1950: 'nineteen fifty',
                           1960: 'nineteen sixty', 1970: 'nineteen seventy', 1980: 'nineteen eighty',
                           1990: 'nineteen ninety', 2000: 'two thousand', 2010: 'two thousand ten'}
            if number not in translation:
                print(f"Attention: Number {number} without translation to words!")
            print(f"Replacing number {number} by {translation.get(number)}")
            return translation.get(number)

        text = text.lower()
        text = text.replace("’", "'")
        text = re.sub(r"([0-9]+)", _convert_number_to_words, text)
        text = re.sub("([^a-z' ])", " ", text)
        text = re.sub(' +', ' ', text)
        return text.strip()

    def _load_audio(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert stereo to mono by averaging the channels
        if waveform.shape[0] == 2:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample the audio to 16000 Hz if it is not already at that sample rate
        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)
            sample_rate = target_sample_rate

        return waveform, sample_rate

    def _store_result(self, waveform, token_spans, emission, transcript, sample_name, timestamps_df, voice_activity_segments_df):
        transcript_version = self.config.config_data.transcript_version
        task = self.config.config_data.task
        output_basedir = os.path.join(self.CONSTANTS.DATA_INTERMEDIATES, "word_timestamps", task, transcript_version)
        output_alignment_dir = os.path.join(output_basedir, "forced_alignment")
        os.makedirs(output_alignment_dir, exist_ok=True)
        output_word_timestamps_dir = os.path.join(output_basedir, "timestamps")
        os.makedirs(output_word_timestamps_dir, exist_ok=True)

        output_voice_activity_dir = os.path.join(output_basedir, "voice_activity")
        os.makedirs(output_voice_activity_dir, exist_ok=True)

        self._plot_alignments(waveform, token_spans, emission, transcript,
                              voice_activity_segments=voice_activity_segments_df,
                              plot_path=os.path.join(output_alignment_dir, f"{sample_name}.pdf"))

        timestamps_to_store = timestamps_df.copy()
        timestamps_to_store["creation_time"] = datetime.now()
        timestamps_to_store.to_csv(os.path.join(output_word_timestamps_dir, f"{sample_name}.csv"), index=False)

        voice_activity_segments_df["creation_time"] = datetime.now()
        voice_activity_segments_df.to_csv(os.path.join(output_voice_activity_dir, f"{sample_name}.csv"), index=False)

    @cache_to_file_decorator(n_days=60)
    def _load_word_timestamps_one(self, audio_path, transcript, sample_name):
        if pd.isna(audio_path) or pd.isna(transcript):
            return None

        print(f"Calculating word timestamps for sample {sample_name}")
        waveform, sample_rate = self._load_audio(audio_path)
        audio_length = waveform.size(1) / sample_rate
        if audio_length > 20*60:
            print(f"ERROR: Audio longer than 20 minutes ({audio_length}) - this becomes a memory issue - don't calculate forced alignment here.")
            return None

        transcript = self._normalize_transcript(transcript)

        assert sample_rate == bundle.sample_rate, f"{sample_rate} != {bundle.sample_rate}"

        transcript = transcript.split()
        tokens = self.tokenizer(transcript)

        emission, token_spans = self._compute_alignments(waveform, transcript)
        num_frames = emission.size(1)

        def prepare_word_info(waveform, spans, num_frames, transcript, sample_rate=bundle.sample_rate):
            ratio = waveform.size(1) / num_frames
            return {
                'word': transcript,
                'start': int(ratio * spans[0].start) / sample_rate,
                'end': int(ratio * spans[-1].end) / sample_rate,
                'score': self._score(spans),
            }

        timestamps_df = pd.DataFrame([prepare_word_info(waveform, token_spans[i], num_frames, transcript[i]) for i in range(len(transcript))])

        # voice activity detection - to plot on the same graph
        voice_activity_segments_df = self.voice_activity_detector.get_segments(audio_path)

        self._store_result(waveform, token_spans, emission, transcript, sample_name, timestamps_df, voice_activity_segments_df)
        return timestamps_df

    def _load_word_timestamps(self, dataset: Dataset):
        assert 'audio_files' in dataset.data_variables, "Dataset should have audio file variable for audio features"
        assert 'transcripts' in dataset.data_variables, "Dataset should have transcripts"

        timestamps = [self._load_word_timestamps_one(path, transcript, sample_name) if not pd.isna(path) else {}
                    for path, transcript, sample_name in zip(dataset.audio_files, dataset.transcripts, dataset.sample_names)]

        config_without_transformers = {key: dataset.config[key] for key in dataset.config if key != 'data_transformers'}
        new_config = {
            'data_transformers': [*dataset.config['data_transformers'], self.name],
            **config_without_transformers
        }

        new_dataset = dataset.copy()
        new_dataset.name = f"{dataset.name} - {self.name}"
        new_dataset.config = new_config

        new_dataset.word_timestamps = timestamps

        return new_dataset

    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        print(f"Calculating word timestamps for {dataset}, based on force alignment")
        return self._load_word_timestamps(dataset)

