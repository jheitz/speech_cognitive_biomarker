import pandas as pd
import numpy as np
import collections
import contextlib
import sys
import wave
import librosa
import webrtcvad
import pydub

from data_analysis.util.decorators import cache_to_file_decorator


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

    @cache_to_file_decorator(verbose=False)
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
