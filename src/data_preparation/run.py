import os, argparse, time, random, string, sys, yaml, subprocess, shutil, datetime

sys.path.append(sys.path[0] + '/..')  # to make the import from parent dir util work

from config.constants import Constants

constants = Constants()

all_modules = ['data_preparation', 'quality_check', 'test_scoring']

arg_parser = argparse.ArgumentParser(description="Read in configuration")
arg_parser.add_argument("--config", help="config file", required=True)
arg_parser.add_argument("--modules", help="modules to run", nargs="*", default=all_modules)
args = arg_parser.parse_args()

f = open(os.path.join(os.getcwd(), args.config), 'r')
RUN_PARAMETERS = yaml.load(f, Loader=yaml.FullLoader)

try:
    NAME = RUN_PARAMETERS['name']
except KeyError:
    raise ValueError("Required config for name")

modules = args.modules if args.modules is not None else all_modules
invalid_modules = [m for m in modules if m not in all_modules]
assert len(invalid_modules) == 0, f"Invalid modules {invalid_modules}, should be in {all_modules}"
print(f"Running modules {modules} for config file {args.config}")

RESULTS_DIR = os.path.join(constants.DATA_PROCESSED, NAME)
os.makedirs(RESULTS_DIR, exist_ok=True)

def export_conda():
    """
    Export current conda environment to file, so it can be persisted in git
    """
    os.system("conda env export > '../conda/environment.yaml'")

export_conda()

def copy_config_file():
    config_file = os.path.join(args.config)
    assert os.path.isfile(config_file), f"Config file does not exist: {config_file}"
    config_file_name = os.path.basename(config_file)
    destination_path = os.path.join(RESULTS_DIR, config_file_name)
    shutil.copy2(config_file, destination_path)

copy_config_file()

# prepare command line arguments to pass to python script
arguments = {key: args.__dict__[key] for key in args.__dict__ if key != 'modules'}
python_commandline_arguments = ' '.join(["--{} {}".format(k, arguments[k]) for k in arguments if arguments[k] is not None])


if 'data_preparation' in modules:
    stdout_file = os.path.join(RESULTS_DIR, "stdout.txt")
    errorlog_file = os.path.join(RESULTS_DIR, "errorlog.txt")
    command = f"python -u prepare_data.py --results_dir {RESULTS_DIR} {python_commandline_arguments} 2> {errorlog_file} > {stdout_file}"
    print(f"Preparing data: {command}")
    os.system(command)

if 'quality_check' in modules:
    stdout_file = os.path.join(RESULTS_DIR, "datacheck_stdout.txt")
    errorlog_file = os.path.join(RESULTS_DIR, "datacheck_errorlog.txt")
    command = f"python -u check_data_quality.py --results_dir {RESULTS_DIR} {python_commandline_arguments} 2> {errorlog_file} > {stdout_file}"
    print(f"Checking data: {command}")
    os.system(command)

if 'test_scoring' in modules:
    stdout_file = os.path.join(RESULTS_DIR, "test_scoring_stdout.txt")
    errorlog_file = os.path.join(RESULTS_DIR, "test_scoring_errorlog.txt")
    command = f"python -u score_language_tests.py --results_dir {RESULTS_DIR} {python_commandline_arguments} 2> {errorlog_file} > {stdout_file}"
    print(f"Scoring language tests: {command}")
    os.system(command)