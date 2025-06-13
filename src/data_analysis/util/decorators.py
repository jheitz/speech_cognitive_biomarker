from config.constants import Constants
import re, os, pickle, time
import numpy as np
import pandas as pd
from util.helpers import create_directory, hash_from_dict

CONSTANTS = Constants()

def cache_to_file_decorator(n_days=3, verbose=True):
    # n_days = number of days the cache should be considered: If older, recalculate

    def cache_to_file_decorator_inner(func):
        def make_str_safe(val):
            """ Make any string safe to use as part of a filename """
            val = re.sub(r'[^\w\d\s_-]', '', str(val))
            val = re.sub(r'[\s_-]', '-', val)
            return val

        def cache_to_file_wrapper(class_self, *args, **kwargs):
            # calculate a fingerprint of the function call: what function has been called on what arguments?
            # class_self is the self argument in a class method, should be ignored
            function_name = func.__name__
            class_name = class_self.__class__.__name__
            assert class_name is not None
            string_args = [make_str_safe(val) for val in args]
            string_kwargs = [make_str_safe(key) + "=" + make_str_safe(val) for key, val in kwargs.items()]
            arguments_string = "|".join(string_args + string_kwargs)[:50]
            complete_config = {'args': args, 'kwargs': kwargs}
            hash = hash_from_dict(complete_config)

            # based on this, calculate a file_path where the cache is kept
            dirname_base = f"{class_name}_{function_name}"
            dirname = os.path.join(CONSTANTS.CACHE_DIR, dirname_base)
            create_directory(dirname)
            filename = f"{function_name}__{arguments_string}__{hash}.pkl"
            file_path = os.path.join(dirname, filename)

            use_cache = False

            # check if file path exists = result has been cached
            # if so, check the date it was written and use cache if recent enough
            if os.path.exists(file_path):
                mod_time = os.path.getmtime(file_path)
                # Calculate number of days since file was modified
                elapsed_days = (time.time() - mod_time) / (24 * 3600)

                # Check if file was modified less than n days ago
                if elapsed_days <= n_days:
                    use_cache = True
                else:
                    if verbose:
                        print(f"     [Cache-To-File] Cache exists but is too old --> recalculating for {class_name}.{function_name}")

            if use_cache:
                # if result has been cached already, load it from disk
                def shorten_arg(arg):
                    if isinstance(arg, list):
                        if len(arg) > 3:
                            return [shorten_arg(a) for a in arg[:3]] + ["..."]
                        else:
                            return [shorten_arg(a) for a in arg[:3]]
                    elif isinstance(arg, dict):
                        return {key: shorten_arg(arg[key]) for key in arg}
                    elif isinstance(arg, np.ndarray):
                        return f"Array[{arg.shape}]({arg.flatten()[:3]}...)"
                    elif isinstance(arg, (pd.Series, pd.DataFrame)):
                        return shorten_arg(arg.to_numpy())
                    elif isinstance(arg, str):
                        if len(arg) > 50:
                            return arg[:50] + "..."
                        else:
                            return arg
                    else:
                        return arg
                args_shortened = [shorten_arg(arg) for arg in args]
                kwargs_shortened = {key: shorten_arg(kwargs[key]) for key in kwargs}
                if verbose:
                    print(f"     [Cache-To-File] Getting value for {class_name}.{function_name} from disk using arguments: {args_shortened} and {kwargs_shortened}")
                return_val = pickle.load(open(file_path, 'rb'))
                #print(f"... {return_val}")
            else:
                # otherwise, actually run the function
                return_val = func(class_self, *args, **kwargs)
                # and store to disk
                pickle.dump(return_val, open(file_path, 'wb'))

            return return_val

        return cache_to_file_wrapper

    return cache_to_file_decorator_inner