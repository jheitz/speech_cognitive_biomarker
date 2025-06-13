import os, argparse, time, random, string, sys, yaml, subprocess, shutil, datetime, copy

sys.path.append(sys.path[0] + '/..')  # to make the import from parent dir util work

from config.constants import Constants
from util.helpers import hash_from_dict


# sometimes we want to try different combinations of settings systematically (e.g. task / target combinations)
# instead of creating all combinations as different config files, we can specify a list in one config file
# and create them here on the spot

constants = Constants()

arg_parser = argparse.ArgumentParser(description="Read in configuration")
arg_parser.add_argument("--config", help="config file", required=True)
arg_parser.add_argument("--name", help="run name", required=True)
arg_parser.add_argument("--results_base_dir", help="base directory to write results to", required=True)
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


try:
    tasks = RUN_PARAMETERS['config_data']['task']
    if not isinstance(tasks, list):
        tasks = [tasks]
except:
    raise ValueError("A task should be specified in the config file")

try:
    targets = RUN_PARAMETERS['config_model']['target_variable']
    if not isinstance(targets, list):
        targets = [targets]
except:
    raise ValueError("A target variable should be specified in the config file")


config_cache_dir = os.path.join(constants.CACHE_DIR, "config_yaml")
os.makedirs(config_cache_dir, exist_ok=True)

for task in tasks:
    for target in targets:
        new_run_parameters = copy.deepcopy(RUN_PARAMETERS)
        new_run_parameters['config_data']['task'] = task
        new_run_parameters['config_model']['target_variable'] = target
        config_base_filename = os.path.splitext(os.path.basename(args.config))[0]
        config_hash = hash_from_dict({'base': args.config, 'target': target, 'task': task}, hash_len=6)
        yaml_file = os.path.join(config_cache_dir, f"{config_base_filename}_{task}_{target}_{config_hash}.yaml")
        new_name = f"{args.name}_{task}_{target}"
        with open(yaml_file, 'w') as outfile:
            yaml.dump(new_run_parameters, outfile)
            print("Writing to {}".format(yaml_file))

        # prepare command line arguments to pass to python script
        arguments = {key: args.__dict__[key] for key in args.__dict__ if key not in ['config', 'name']}
        python_commandline_arguments = ' '.join(["--{} {}".format(k, arguments[k]) for k in arguments if arguments[k] is not None])

        command = f"python -u run.py --name {new_name} --config {yaml_file} {python_commandline_arguments} "
        print(f"Running: {command}")
        os.system(command)

