import types
import argparse

class RunParameters(types.SimpleNamespace):
    """ Class to hold information on the current run """

    def __init__(self, **params):
        super().__init__(**params)

    @classmethod
    def from_command_line_args(cls):
        # read command line arguments
        arg_parser = argparse.ArgumentParser(description="Read in configuration")
        arg_parser.add_argument("--config", help="config file", required=True)
        arg_parser.add_argument("--local", help="is local setup?")
        arg_parser.add_argument("--name", help="run name")
        arg_parser.add_argument("--results_dir", help="results directory", required=True)
        arg_parser.add_argument("--results_base_dir", help="results base directory", required=False)
        args = vars(arg_parser.parse_args())

        return cls(**args)


