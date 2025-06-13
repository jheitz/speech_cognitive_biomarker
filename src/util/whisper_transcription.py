import torch
import pandas as pd
import numpy as np
import json
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from util.helpers import hash_from_dict
import datetime
import collections
import contextlib
import sys
import wave
import librosa
import webrtcvad
import pydub
import soundfile as sf
import random
import string
import shutil, re
import hashlib
import logging
import pickle

from config.constants import Constants


class AudioFrame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

class VoiceActivityDetector:
    """
    Voice Activity Detection algorithm, based on Google's WebRTC Voice Activity Detector (Python wrapper available in
    https://github.com/wiseman/py-webrtcvad, example code implementation in
    https://github.com/wiseman/py-webrtcvad/blob/master/example.py)
    This is used in a microsoft paper "End-to-End Speaker-Attributed ASR with Transformer": https://arxiv.org/pdf/2104.02128.pdf
    The version below is taken from jheitz/dementia/src/preprocessing/audio_features.py
    """

    def __init__(self, debug=False):
        self.debug = debug

        # aggressiveness mode, which is an integer between 0 and 3.
        # 0 is the least aggressive about filtering out non-speech, 3 is the most aggressive
        self.aggressiveness = 1


    def _read_wave(self, path):
        """Reads an audio file.

        Takes the path, and returns (PCM audio data, sample rate).
        If the file is in MP3 format, convert it to WAV.
        If the audio is stereo, convert it to mono.
        If the sample rate is not 16kHz, resample the audio to 16kHz.
        """
        _, file_extension = path.split('.', 1)

        # Convert MP3 to WAV if the file is in MP3 format.
        if file_extension.lower() == 'mp3':
            audio = pydub.AudioSegment.from_mp3(path)
            sample_rate = audio.frame_rate

            # Convert stereo to mono if needed.
            if audio.channels > 1:
                audio = audio.set_channels(1)

            audio = audio.set_frame_rate(16000)  # Set sample rate to 16kHz
            audio = audio.set_sample_width(2)  # Set sample width to 16 bits
            pcm_data = audio.raw_data
        else:  # For WAV files
            with contextlib.closing(wave.open(path, 'rb')) as wf:
                num_channels = wf.getnchannels()
                sample_rate = wf.getframerate()
                sample_width = wf.getsampwidth()
                assert sample_width == 2

                # Check if the audio is stereo; if so, convert it to mono.
                if num_channels == 2:
                    pcm_data = wf.readframes(wf.getnframes())
                    pcm_data = pydub.AudioSegment(
                        data=pcm_data, sample_width=2, frame_rate=sample_rate, channels=2
                    )
                    pcm_data = pcm_data.set_channels(1)
                    pcm_data = pcm_data.raw_data
                else:
                    pcm_data = wf.readframes(wf.getnframes())

        # Check if the sample rate is 16kHz; if not, resample the audio.
        if sample_rate != 16000:
            y, _ = librosa.load(path, sr=16000)
            pcm_data = (y * 32768).astype('<h').tobytes()
            sample_rate = 16000

        return pcm_data, sample_rate

    def _write_wave(self, sample_id, audio, sample_rate):
        """Writes a .wav file.

        Takes sample id, PCM audio data, and sample rate.
        """
        path = 'chunk-%002d.wav' % (sample_id,)
        with contextlib.closing(wave.open(path, 'wb')) as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio)

    def _frame_generator(self, frame_duration_ms, audio, sample_rate):
        """Generates audio frames from PCM audio data.

        Takes the desired frame duration in milliseconds, the PCM data, and
        the sample rate.

        Yields Frames of the requested duration.
        """
        n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
        offset = 0
        timestamp = 0.0
        duration = (float(n) / sample_rate) / 2.0
        while offset + n < len(audio):
            yield AudioFrame(audio[offset:offset + n], timestamp, duration)
            timestamp += duration
            offset += n

    def _vad_collector(self, sample_rate, frame_duration_ms,
                       padding_duration_ms, vad, frames):
        """Filters out non-voiced audio frames.

        Given a webrtcvad.Vad and a source of audio frames, yields only
        the voiced audio.

        Uses a padded, sliding window algorithm over the audio frames.
        When more than 90% of the frames in the window are voiced (as
        reported by the VAD), the collector triggers and begins yielding
        audio frames. Then the collector waits until 90% of the frames in
        the window are unvoiced to detrigger.

        The window is padded at the front and back to provide a small
        amount of silence or the beginnings/endings of speech around the
        voiced frames.

        Arguments:

        sample_rate - The audio sample rate, in Hz.
        frame_duration_ms - The frame duration in milliseconds.
        padding_duration_ms - The amount to pad the window, in milliseconds.
        vad - An instance of webrtcvad.Vad.
        frames - a source of audio frames (sequence or generator).

        Returns: A generator that yields PCM audio data. start, and end timestamp
        """
        num_padding_frames = int(padding_duration_ms / frame_duration_ms)
        if self.debug:
            print("num_padding_frames", num_padding_frames)
        # We use a deque for our sliding window/ring buffer.
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
        # NOTTRIGGERED state.
        triggered = False

        # start and end of segment
        start_timestamp, end_timestamp = None, None

        if self.debug:
            print("total length", sum([f.duration for f in frames]))

        voiced_frames = []
        for frame in frames:
            is_speech = vad.is_speech(frame.bytes, sample_rate)

            if self.debug:
                sys.stdout.write('1' if is_speech else '0')
            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                # If we're NOTTRIGGERED and more than 90% of the frames in
                # the ring buffer are voiced frames, then enter the
                # TRIGGERED state.
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    triggered = True
                    if self.debug:
                        sys.stdout.write('+(%.2f)' % (ring_buffer[0][0].timestamp,))
                    start_timestamp = ring_buffer[0][0].timestamp
                    # print(f"Detected start at {start_timestamp:.2f}s")

                    # We want to yield all the audio we see from now until
                    # we are NOTTRIGGERED, but we have to start with the
                    # audio that's already in the ring buffer.
                    for f, s in ring_buffer:
                        voiced_frames.append(f)
                    ring_buffer.clear()
            else:
                # We're in the TRIGGERED state, so collect the audio data
                # and add it to the ring buffer.
                voiced_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                # If more than 90% of the frames in the ring buffer are
                # unvoiced, then enter NOTTRIGGERED and yield whatever
                # audio we've collected.
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    if self.debug:
                        sys.stdout.write('-(%.2f)' % (frame.timestamp + frame.duration))
                    num_consecutive_unvoiced_frames = len([f for f, speech in ring_buffer if not speech])
                    end_timestamp = (frame.timestamp + frame.duration) - num_unvoiced * frame_duration_ms / 1000
                    # print(f"Detected end at {end_timestamp:.2f}s")
                    triggered = False
                    # todo: use consecutive instead of num_unvoiced
                    # yield the current segment's frames until the last voiced frame
                    yield b''.join([f.bytes for f in voiced_frames[:len(
                        voiced_frames) - num_unvoiced]]), start_timestamp, end_timestamp
                    ring_buffer.clear()
                    voiced_frames = []
        if triggered:
            if self.debug:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
        if self.debug:
            sys.stdout.write('\n')
        # If we have any leftover voiced audio when we run out of input, yield it.
        if voiced_frames:
            yield b''.join([f.bytes for f in voiced_frames]), start_timestamp, frame.timestamp + frame.duration

    def get_segments(self, audio_path):
        """ Get pause / voice segments """
        if self.debug:
            print(f"Getting segments for audio file at {audio_path}")
        audio, sample_rate = self._read_wave(audio_path)
        vad = webrtcvad.Vad(self.aggressiveness)
        frames = self._frame_generator(30, audio, sample_rate)
        frames = list(frames)
        prev_end = 0
        voiced_segments_generator = self._vad_collector(sample_rate, 30, 300, vad, frames)
        segments = []
        for i, (segment, start, end) in enumerate(voiced_segments_generator):
            pause_duration = start - prev_end
            if self.debug:
                print(f"\npause {pause_duration:.2f}, start {start:.2f}, end {end:.2f}\n")
            #self._write_wave(i, segment, sample_rate)
            assert start == 0 or pause_duration > 0
            segments.append({'type': 'pause', 'start': prev_end, 'end': start})
            segments.append({'type': 'voice', 'start': start, 'end': end})
            prev_end = end

        # if end of last segment is not end of audio, add a last pause segment
        last_frame = frames[-1]
        last_frame_end = last_frame.timestamp + last_frame.duration
        if prev_end < last_frame_end:
            pause_duration = last_frame_end - prev_end
            if self.debug:
                print(f"Pause at the end of audio, duration {pause_duration:.2f}")
            segments.append({'type': 'pause', 'start': prev_end, 'end': last_frame_end})

        segments_df = pd.DataFrame(segments)
        segments_df['duration'] = segments_df['end'] - segments_df['start']
        return segments_df



class Whisper_Transcriber():

    def __init__(self, language='english', return_timestamps=None, use_vad=True, *args, **kwargs):
        self.name = "whisper"

        assert not (return_timestamps is not None and use_vad), \
            "Whisper uses VAD but returns timestamps. This does not work, because VAD changes the timing"

        self.language = language
        assert self.language in ['english']

        self.use_vad = use_vad
        if self.use_vad:
            self.voice_activity_detector = VoiceActivityDetector(debug=False)

        self.version = 1  # version of the transcriber's code -> if significant logic changes, change this
        self.config = {'return_timestamps': return_timestamps, 'use_vad': use_vad, 'generate_kwargs': {"language": self.language}}
        self.config_hash = hash_from_dict({**self.config, 'version': self.version}, 6)

        self.current_date = str(datetime.date.today())

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "openai/whisper-large-v3"

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, use_safetensors=True,  # low_cpu_mem_usage=True,
        )
        self.model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        self.whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            torch_dtype=torch_dtype,
            device=device,
            **{'return_timestamps': return_timestamps}
        )

        self.CONSTANTS = Constants()
        self.cache_dir = self.CONSTANTS.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        self.VAD_cache_dir = os.path.join(self.cache_dir, "audio_VAD")

        print(f"Using {model_id} ({language}) with config {self.config}")

    def _remove_silence(self, file_path):
        segments = self.voice_activity_detector.get_segments(file_path)

        # Load the audio file
        audio, sr = librosa.load(file_path, sr=None)

        # get all voice segments (remove pauses)
        voice_segments = []
        for index, row in segments.iterrows():
            if row['type'] == 'voice':
                start_sample = int(sr * row['start'])
                end_sample = int(sr * row['end'])
                # Extract the segment from the audio array
                segment = audio[start_sample:end_sample]
                voice_segments.append(segment)

        # If there is no voice, return None
        if len(voice_segments) == 0:
            return None

        # Concatenate all voice segments
        voice_audio = np.concatenate(voice_segments)

        # Write the concatenated voice segments to a new file
        file_name_basis = ".".join(os.path.basename(file_path).split(".")[:-1])
        try:
            sample_name = re.match(r".*/([0-9]+)/(.+)\.wav", file_path).group(1)
        except:
            sample_name = ''.join(random.choice(string.ascii_lowercase) for i in range(4)) # random string
        os.makedirs(self.VAD_cache_dir, exist_ok=True)
        new_file_name = os.path.join(self.VAD_cache_dir, f'{file_name_basis}_{sample_name}_VAD.wav')
        sf.write(new_file_name, voice_audio, sr)
        return new_file_name

    def _get_cache_dir(self, file_path):
        # hash of actual audio content, to handle caching
        audio, sr = librosa.load(file_path, sr=None)
        audio_hash = self.name + "_" + self.config_hash + "_" + hashlib.sha256(audio.data).hexdigest()[:16]
        cache_dir = os.path.join(self.cache_dir, audio_hash)
        os.makedirs(cache_dir, exist_ok=True)
        return audio_hash, cache_dir

    def transcribe_file(self, file_path: str):
        print(f"Whisper transcribing file {file_path}.", end=" ")
        audio_hash, cache_dir = self._get_cache_dir(file_path)
        cached_results = os.path.join(cache_dir, 'results.pkl')
        if os.path.exists(cached_results):
            print(f"... from cache ({audio_hash})")
            results = pickle.load(open(cached_results, 'rb'))

            # if the VAD file is stored here in the cache dir (introduced on 2024-06-27)
            # we copy it to self.VAD_cache_dir, s.t. it's available for some of the data quality checks
            # this is necessary because the VAD file withing self.VAD_cache_dir is only written once
            # when a file gets renamed from e.g. journaling2.wav to journaling.wav, it can no longer
            # be found there, or the vad file for the previous journaling.wav is still there, resp.
            try:
                sample_name = re.match(r".*/([0-9]+)/(.+)\.wav", file_path).group(1)
            except:
                sample_name = ''.join(random.choice(string.ascii_lowercase) for i in range(4))  # random string
            file_name_basis = ".".join(os.path.basename(file_path).split(".")[:-1])
            file_path_vad = os.path.join(cache_dir, 'vad.wav')
            if os.path.exists(file_path_vad):
                dst_file = os.path.join(self.VAD_cache_dir, f'{file_name_basis}_{sample_name}_VAD.wav')
                if os.path.exists(dst_file):
                    os.remove(dst_file)
                shutil.copyfile(file_path_vad, dst_file)

                # Write the concatenated voice segments to a new file
                file_name_basis = ".".join(os.path.basename(file_path).split(".")[:-1])
                try:
                    sample_name = re.match(r".*/([0-9]+)/(.+)\.wav", file_path).group(1)
                except:
                    sample_name = ''.join(random.choice(string.ascii_lowercase) for i in range(4))  # random string
                os.makedirs(self.VAD_cache_dir, exist_ok=True)
                new_file_name = os.path.join(self.VAD_cache_dir, f'{file_name_basis}_{sample_name}_VAD.wav')


        else:
            if self.use_vad:
                print("... Using VAD model to remove pauses.", end=" ")
                file_path_vad = self._remove_silence(file_path)
                if file_path_vad is not None:
                    shutil.copyfile(file_path_vad, os.path.join(cache_dir, 'vad.wav'))
                if file_path_vad is None:
                    logging.warning(f"Silence removal (VAD) failed for file {file_path}, as there is no voice segment? We continued with the full (silent) file")
                else:
                    file_path = file_path_vad

            print(f"... transcribing.")
            results = self.whisper_pipe(file_path, generate_kwargs={"language": self.language})
            pickle.dump(results, open(cached_results, 'wb'))
            if self.use_vad:
                # delete temporary file again
                assert '_VAD' in file_path or file_path_vad is None
                #os.remove(file_path)
        return results

