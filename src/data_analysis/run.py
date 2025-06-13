import os, argparse, time, random, string, sys, yaml, subprocess, shutil, datetime, socket

sys.path.append(sys.path[0] + '/..')  # to make the import from parent dir util work

from config.constants import Constants
from util.helpers import hash_from_dict

constants = Constants()

arg_parser = argparse.ArgumentParser(description="Read in configuration")
arg_parser.add_argument("--config", help="config file", required=True)
arg_parser.add_argument("--name", help="run name", required=True)
arg_parser.add_argument("--results_base_dir", help="base directory to write results to")
args = arg_parser.parse_args()

f = open(os.path.join(os.getcwd(), args.config), 'r')
RUN_PARAMETERS = yaml.load(f, Loader=yaml.FullLoader)

if args.results_base_dir is not None:
    RESULTS_BASE_DIR = args.results_base_dir  # + "_" + time.strftime("%Y-%m-%d_%H%M")
else:
    RESULTS_BASE_DIR = ""


try:
    NAME = args.name
except KeyError:
    raise ValueError("Required config for name")


year = datetime.datetime.today().isocalendar().year
calendar_week = datetime.datetime.today().isocalendar().week
year_week = f"{year}_kw{calendar_week}"
results_subdir = time.strftime("%Y%m%d_%H%M") + "_" + NAME + '_' + ''.join(random.choices(string.ascii_lowercase, k=4))
RESULTS_DIR = os.path.join(constants.RESULTS_ROOT, "runs_luha", year_week, RESULTS_BASE_DIR, results_subdir)
os.makedirs(RESULTS_DIR, exist_ok=True)
stdout_file = os.path.join(RESULTS_DIR, "stdout.txt")
errorlog_file = os.path.join(RESULTS_DIR, "errorlog.txt")


def export_conda():
    """
    Export current conda environment to file, so it can be persisted in git
    """
    os.system("conda env export > '../../conda/environment.yaml'")

export_conda()


def persist_current_code():
    """
    commit and push current state of code to constants.EXPERIMENT_BRANCH in git
    """
    def git_command(command):
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error executing Git command: {result.stderr}")
            raise Exception()
        return result.stdout.strip()

    # Change to EXPERIMENT_BRANCH branch (plus vm-dependent postfix, to avoid conflicts), commit, and push
    git_command(["git", "checkout", f"{constants.EXPERIMENT_BRANCH}_{hash_from_dict({'user': os.getlogin(), 'host': socket.gethostname()}, hash_len=4)}"])
    git_command(["git", "add", "../.."])

    # check if changes to commit
    status = git_command(["git", "status", "--porcelain"])
    if status:
        git_command(["git", "commit", "-m", f"Run {os.path.join(year_week, RESULTS_BASE_DIR, results_subdir)}"])

    # Get the commit hash of the latest commit
    commit_hash = git_command(["git", "rev-parse", "HEAD"])

    # Write the commit hash to a file
    with open(os.path.join(RESULTS_DIR, "git_commit_hash.txt"), "w") as file:
        file.write(commit_hash)

    if status:
        # push only here because this can fail - we still want the hash to fix it later on the VM
        git_command(["git", "push"])


try:
    persist_current_code()
except Exception as e:
    print(f"An exception occurred persisting the current code to git:\n\nException message: {str(e)}")
    print("Continuing without pushing to git...")



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


stdout_file = os.path.join(RESULTS_DIR, "stdout.txt")
errorlog_file = os.path.join(RESULTS_DIR, "errorlog.txt")
command = f"python -u main.py --results_dir {RESULTS_DIR} {python_commandline_arguments} 2> {errorlog_file} > {stdout_file}"
print(f"Running: {command}")
os.system(command)
