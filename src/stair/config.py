from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"


def load_yaml_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as config_file:
        loaded = yaml.safe_load(config_file)
    if loaded is None:
        return {}
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
