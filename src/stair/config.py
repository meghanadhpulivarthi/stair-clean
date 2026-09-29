from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"


def load_yaml_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.is_dir():
        raise OSError(f"Config path is a directory, not a file: {path}")
    with open(path, "r") as config_file:
        try:
            loaded = yaml.safe_load(config_file)
        except yaml.YAMLError as parse_error:
            raise ValueError(f"Could not parse config file {path}: {parse_error}")
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Config file {path} must be a mapping of keys to values, "
            f"got {type(loaded).__name__}"
        )
    return loaded


def deep_merge(base_dict, override_dict):
    merged = dict(base_dict)
    for key, override_value in override_dict.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(override_value, dict):
            merged[key] = deep_merge(base_value, override_value)
        else:
            merged[key] = override_value
    return merged


def find_unknown_keys(base_dict, override_dict, path_prefix=""):
    unknown_keys = []
    for key, override_value in override_dict.items():
        full_path = f"{path_prefix}.{key}" if path_prefix else key
        if key not in base_dict:
            unknown_keys.append(full_path)
            continue
        base_value = base_dict[key]
        if isinstance(override_value, dict) and isinstance(base_value, dict):
            unknown_keys.extend(find_unknown_keys(base_value, override_value, full_path))
    return unknown_keys


def resolve_config(override_path=None):
    base_config = load_yaml_file(BASE_CONFIG_PATH)

    if override_path is None:
        return base_config

    override_config = load_yaml_file(override_path)

    unknown_keys = find_unknown_keys(base_config, override_config)
    if unknown_keys:
        joined_keys = ", ".join(unknown_keys)
        raise ValueError(
            f"Override config has keys not present in base config: {joined_keys}. "
            f"Check configs/base.yaml for valid keys."
        )

    return deep_merge(base_config, override_config)
