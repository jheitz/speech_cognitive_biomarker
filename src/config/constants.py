import os, sys


class Constants:
    """
    A class of project-wise CONSTANTS.
    Directory paths can depend on being run locally or not (which is given by the --local runtime parameter)
    This local flag can also be set manually (Constants(True) or Constants(False)), or, in any case,
    the local / remote version can be accessed using Constants().LOCAL.CONSTANT_NAME / Constants().REMOTE.CONSTANT_NAME
    """
    def __init__(self, local=None, recursion=True):
        git_dir_remote = "/home/ubuntu/git/luha-prolific-study"
        git_dir_local = "/Users/jheitz/git/luha-prolific-study"

        if local is not None:
            self.local = local
        else:
            if "--local" in sys.argv:
                self.local = True
            elif os.path.exists(git_dir_local):
                self.local = True
            else:
                self.local = False

        mounted_methlab_remote = "/home/ubuntu/methlab/Students/Jonathan/"
        if self.local:
            mounted_methlab_remote = "/Volumes/methlab/Students/Jonathan/local"
        mounted_methlab_data_remote = "/home/ubuntu/methlab_data/LanguageHealthyAging/"

        # Git branch where code snapshots are committed and pushed to by src/run/run.py
        self.EXPERIMENT_BRANCH = 'experiments'

        # path to directory of project git
        self.GIT_DIR = git_dir_remote
        if self.local:
            self.GIT_DIR = git_dir_local

        self.CACHE_DIR = os.path.join(self.GIT_DIR, "cache")
        self.CACHE_DIR_CENTRALIZED = os.path.join(mounted_methlab_remote, "luha_cache")  # centralized cache (for all virtual machines)
        if self.local:
            self.CACHE_DIR_CENTRALIZED = os.path.join(self.GIT_DIR, "cache")
        self.RESOURCES_DIR = os.path.join(self.GIT_DIR, "src/resources")

        self.DATA_RAW = None
        self.DATA_PROCESSED = None
        self.DATA_PROCESSED_COMBINED = os.path.join(mounted_methlab_remote, "data", "prolificStudy", "processed_combined")
        #self.DATA_PROCESSED_COMBINED = os.path.join(mounted_methlab_data_remote)
        self.DATA_INTERMEDIATES = os.path.join(mounted_methlab_remote, "data_intermediates", "prolificStudy")
        if self.local:
            self.DATA_RAW = os.path.join(self.GIT_DIR, "data", "raw")
            self.DATA_PROCESSED = os.path.join(self.GIT_DIR, "data", "processed")
            self.DATA_PROCESSED_COMBINED = os.path.join(self.GIT_DIR, "data", "processed_combined")
            self.DATA_INTERMEDIATES = os.path.join(self.GIT_DIR, "data", "intermediates")

        self.TRAIN_TEST_DATASPLIT = os.path.join(self.RESOURCES_DIR, "train_test_split_2025-01-08_17-03.csv")  # based on demographics & cognition
        self.SPLIT1_SPLIT2_DATASPLIT = os.path.join(self.RESOURCES_DIR, "data_split_2024-07-10_16-11.csv")

        # based on analyses/kw04/cognitive_negative_outliers.ipynb - based on new data split and cognitive scores
        self.COGNITIVE_NEGATIVE_OUTLIERS = os.path.join(self.RESOURCES_DIR, "cognitive_negative_outliers_2025-01-21_13-45.csv")

        self.RESULTS_ROOT = os.path.join(mounted_methlab_remote, "results")
        self.RESULTS_ROOT_REMOTE = self.RESULTS_ROOT
        if self.local:
            self.RESULTS_ROOT = os.path.join(self.GIT_DIR, "results")
            # remote, but accessed from local machine
            self.RESULTS_ROOT_REMOTE = "/Volumes/methlab/Students/Jonathan/results"

        self.ACS_NORMATIVE_DATA = os.path.join(self.GIT_DIR, "ACS_normative_data", "csv")

        self.ACS_MAIN_OUTCOME_VARIABLES = ['connect_the_dots_I_time_msec',
                                           'connect_the_dots_II_time_msec', 'wordlist_correct_words',
                                           'avg_reaction_speed', 'place_the_beads_total_extra_moves',
                                           'box_tapping_total_correct', 'fill_the_grid_total_time',
                                           'wordlist_delayed_correct_words', 'wordlist_recognition_correct_words',
                                           'digit_sequence_1_correct_series', 'digit_sequence_2_correct_series']

        self.ACS_MAIN_OUTCOME_VARIABLES_EXTENDED = \
            self.ACS_MAIN_OUTCOME_VARIABLES + ['dragskill_time', 'clickskill_time', 'typeskill_time']

        # no norms are available for 'wordlist_recognition_correct_words'
        self.ACS_MAIN_OUTCOME_VARIABLES_FOR_ORIGINAL_NORMS = [c for c in self.ACS_MAIN_OUTCOME_VARIABLES
                                                              if c != 'wordlist_recognition_correct_words']


        # ADReSS data set
        self.DATA_ADReSS_ROOT = "/home/ubuntu/methlab/Students/Jonathan/data/dementiabank_extracted/0extra/ADReSS-IS2020-data"
        if self.local:
            self.DATA_ADReSS_ROOT = "/Users/jheitz/phd/data/dementiabank_extracted/0extra/ADReSS-IS2020-data"

        self.DATA_ADReSS_TRAIN_CONTROL = os.path.join(self.DATA_ADReSS_ROOT, "train/Full_wave_enhanced_audio/cc")
        self.DATA_ADReSS_TRAIN_AD = os.path.join(self.DATA_ADReSS_ROOT, "train/Full_wave_enhanced_audio/cd")
        self.DATA_ADReSS_TEST = os.path.join(self.DATA_ADReSS_ROOT, "test/Full_wave_enhanced_audio")
        self.DATA_ADReSS_TEST_METADATA = os.path.join(self.DATA_ADReSS_ROOT, "test", "meta_data_with_labels.csv")
        self.DATA_ADReSS_TRAIN_CONTROL_METADATA = os.path.join(self.DATA_ADReSS_ROOT, "train", "cc_meta_data.txt")
        self.DATA_ADReSS_TRAIN_AD_METADATA = os.path.join(self.DATA_ADReSS_ROOT, "train", "cd_meta_data.txt")

        self.ADReSS_PREPROCESSED_DATA = "/home/ubuntu/methlab/Students/Jonathan/data_preprocessed"
        self.ADReSS_PREPROCESSED_DATA_REMOTE = self.ADReSS_PREPROCESSED_DATA
        if self.local:
            self.ADReSS_PREPROCESSED_DATA = "/Users/jheitz/phd/data_preprocessed"
            self.ADReSS_PREPROCESSED_DATA_REMOTE = "/Volumes/methlab/Students/Jonathan/data_preprocessed"

        self.ADReSS_ORIGINAL_PITT_FILES = os.path.join(self.ADReSS_PREPROCESSED_DATA, "ADReSS_original_PITT_files")

        # Segmentation files (timing for INV & PAR)
        self.DATA_ADReSS_SEGMENTATION = os.path.join(self.ADReSS_PREPROCESSED_DATA, "ADReSS_segmentation")



        # create an explicit attribute LOCAL & REMOTE to get the constants of the local or remote environment
        if recursion:
            # make sure this is not done recursively forever (only once -> recursion=False here)
            self.LOCAL = Constants(local=True, recursion=False)
            self.REMOTE = Constants(local=False, recursion=False)






