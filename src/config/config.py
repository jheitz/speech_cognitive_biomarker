import yaml, types

class Config(types.SimpleNamespace):
    """Config class"""

    def __init__(self, **params):
        super().__init__(**params)

    @classmethod
    def from_yaml(cls, config_file):
        """Creates config from yaml"""
        f = open(config_file, 'r')
        params = yaml.load(f, Loader=yaml.FullLoader)

        return cls.from_dict(params)

    @classmethod
    def from_dict(cls, params):
        """Creates config from Python dict"""

        def convert_to_namespace_recursive(val):
            # Recursively convert to namespace, so you can access it by attribute, e.g. config.config_data.keep_terminators
            if isinstance(val, dict):
                return types.SimpleNamespace(**{key: convert_to_namespace_recursive(val[key]) for key in val})
            elif isinstance(val, (list, tuple)):
                t = type(val)
                return t([convert_to_namespace_recursive(elem) for elem in val])
            else:
                return val

        params_converted = {key: convert_to_namespace_recursive(params[key]) for key in params}
        return cls(**params_converted)