import numpy as np
import os
import datetime
import librosa
import hashlib
import pickle
from google.cloud import storage
from google.cloud import speech
from google.api_core.exceptions import GoogleAPICallError
from google.api_core.exceptions import NotFound
from google.api_core import client_options
from google.cloud import speech_v2

from util.helpers import hash_from_dict
from config.constants import Constants


class GoogleSpeechTranscriber():

    def __init__(self, language_code='en-US', model="chirp", enable_automatic_punctuation=True, max_alternatives=1,
                 enable_word_time_offsets=True, *args, **kwargs):
        self.name = "google_speech"

        self._set_auth()

        # Settings for GCS bucket and GCP project_id
        self.project_id = 'sodium-pager-388408'
        self.bucket_name = "jheitz_general"

        # 'chirp' for latest model, 'long' for more options (e.g. adaptation), 'short' for short snippets of audio
        self.model = model
        assert model in ['chirp', 'long', 'short']

        assert language_code in ['en-US', 'en-GB']

        # if not, let's get default values for what is not defined yet
        self.config = {
            'enable_automatic_punctuation': enable_automatic_punctuation,
            'max_alternatives': max_alternatives,
            'enable_word_time_offsets': enable_word_time_offsets
        }

        # Initialize the Google Cloud clients
        self.storage_client = storage.Client()

        self.recognizer_location = 'europe-west4' if self.model == 'chirp' else 'europe-west3'
        client_options_var = client_options.ClientOptions(
            api_endpoint=f"{self.recognizer_location}-speech.googleapis.com"
        )
        self.speech_client = speech_v2.SpeechClient(client_options=client_options_var)
        self.recognition_config = {
            'auto_decoding_config': speech_v2.types.cloud_speech.AutoDetectDecodingConfig(),
            'model': self.model,
            'language_codes': [language_code],
            'features': self.config,
        }

        self.version = 1  # version of the transcriber's code -> if significant logic changes, change this
        self.config_hash = hash_from_dict({**self.config, 'version': self.version}, 6)

        self.current_date = str(datetime.date.today())

        self.CONSTANTS = Constants()
        self.cache_dir = self.CONSTANTS.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        print(f"Using model {self.model} ({language_code}) with config {self.config}")

    def _request_v2(self, file_metadata):
        """
        Request for the v2 API. Note that the recognizer needs to change depending on the location used
        """
        config = speech_v2.types.cloud_speech.RecognitionConfig(**self.recognition_config)
        recognizer = f"projects/{self.project_id}/locations/{self.recognizer_location}/recognizers/_"
        request = speech_v2.types.cloud_speech.BatchRecognizeRequest(
            recognizer=recognizer,
            config=config,
            files=[file_metadata],
            recognition_output_config=speech_v2.types.cloud_speech.RecognitionOutputConfig(
                inline_response_config=speech_v2.types.cloud_speech.InlineOutputConfig(),
            ),
        )
        return request

    def _set_auth(self):
        """
        Set Google Cloud credentials.
        This is done using Service Account Keys (RSA authentification for server-to-server)
        Check the documentation
        - https://cloud.google.com/docs/authentication/provide-credentials-adc#local-key
        - Create service account key: https://console.cloud.google.com/iam-admin/serviceaccounts/create?authuser=3&walkthrough_id=iam--create-service-account-keys&project=sodium-pager-388408&supportedpurview=project
        """
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../keys/gcloud-sodium-pager-388408-95aa25d97ab0.json")

    def _get_cache_dir(self, file_path, extra_config_string):
        # hash of actual audio content, to handle caching
        audio, sr = librosa.load(file_path, sr=None)
        audio_hash = self.name + "_" + self.config_hash + "_" + hashlib.sha256(audio.data).hexdigest()[:16] + extra_config_string
        cache_dir = os.path.join(self.cache_dir, audio_hash)
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def _transcribe_file(self, file_path: str, extra_config_string: str = "") -> str:
        basename = os.path.basename(file_path)

        try:
            # Upload the audio file to Google Cloud Storage
            folder_path = f"api_access/{self.config_hash}{extra_config_string}/{self.current_date}/"
            blob_name = folder_path + basename
            blob = self.storage_client.bucket(self.bucket_name).blob(blob_name)
            if blob.exists():
                pass

            blob.upload_from_filename(file_path)

            print(f"Transcribing file {file_path} for config_string {extra_config_string}")

            # Get the GCS URI of the uploaded audio file
            gcs_uri = f"gs://{self.bucket_name}/{blob_name}"

            # Configure the speech recognition request for asynchronous recognition
            file_metadata = speech_v2.types.cloud_speech.BatchRecognizeFileMetadata(uri=gcs_uri)
            request = self._request_v2(file_metadata)
            operation = self.speech_client.batch_recognize(request=request)

            print(f"Waiting for operation to complete for {basename}...")
            operation_result = operation.result()

            # note that there's only one result (operation_result.results) because only one file is given
            operation_result = list(operation_result.results.values())[0]

            results_parsed = {
                'total_billed_time': str(operation_result.metadata.total_billed_duration),
                'output_error': str(operation_result.error),
                'results': [{
                    'alternatives': [{
                        'transcript': alt.transcript,
                        'confidence': alt.confidence,
                        'words': [{
                            'word': word.word,
                            'confidence': word.confidence,
                        } for word in alt.words]
                    } for alt in result.alternatives],
                    'language_code': result.language_code
                } for result in operation_result.transcript.results],
                'combined_transcript': "\n".join(
                    [result.alternatives[0].transcript for result in operation_result.transcript.results if
                     len(result.alternatives) > 0]),
                'combined_confidences': [result.alternatives[0].confidence for result in
                                         operation_result.transcript.results if len(result.alternatives) > 0],
                'complete_result': operation_result
            }

            print(f"Transcription completed for {basename}")

            blob.delete()

        except GoogleAPICallError as e:
            raise Exception(f"An error occurred while processing {basename}: {str(e)}")
        except NotFound as e:
            raise Exception(f"Audio file not found: {str(e)}")
        except Exception as e:
            raise Exception(f"An unexpected error occurred while processing {basename}: {str(e)}")

        return results_parsed

    def transcribe_file(self, file_path: str, extra_config_string: str = "") -> str:
        cache_dir = self._get_cache_dir(file_path, extra_config_string)
        cached_results = os.path.join(cache_dir, 'results.pkl')
        if os.path.exists(cached_results):
            print("Getting Google Speech transcription from cache")
            results = pickle.load(open(cached_results, 'rb'))
        else:
            results = self._transcribe_file(file_path, extra_config_string)
            pickle.dump(results, open(cached_results, 'wb'))

        return results


