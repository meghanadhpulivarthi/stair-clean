import json
import sys
from pathlib import Path

from stair.config import resolve_config
from stair.core.train.args_builder import build_training_cli_args


def args_list_to_dict(args_list):
    args_as_dict = {}
    index = 0
    while index < len(args_list):
        key = args_list[index].lstrip("-")
        value = args_list[index + 1]
        args_as_dict[key] = value
        index += 2
    return args_as_dict


def test_build_training_cli_args_has_an_even_number_of_entries():
    config = resolve_config(override_path=None)
    cli_args = build_training_cli_args(config, data_dir="data/example_book", run_dir="runs/example")
    assert len(cli_args) % 2 == 0


def test_build_training_cli_args_sets_required_fields():
    config = resolve_config(override_path=None)
    cli_args = build_training_cli_args(config, data_dir="data/example_book", run_dir="runs/example")
    args_as_dict = args_list_to_dict(cli_args)

    assert args_as_dict["model_name"] == config["model"]["name"]
    assert args_as_dict["save_path"] == "runs/example"
    assert args_as_dict["batch_size_per_gpu"] == str(config["training"]["batch_size_per_gpu"])
    # "none" (not "no"): transformers 5.x maps report_to="none" to [] (no
    # reporting) and rejects "no" as an unknown integration name. It stays a
    # scalar string, kept intact through the vendored parser via NO_PARSE_KEYS.
    assert args_as_dict["report_to"] == "none"
    assert args_as_dict["padding_side"] == config["training"]["padding_side"]
    assert args_as_dict["load_best_model_at_end"] == str(config["training"]["load_best_model_at_end"])
    assert args_as_dict["metric_for_best_model"] == config["training"]["metric_for_best_model"]
    assert args_as_dict["greater_is_better"] == str(config["training"]["greater_is_better"])


def test_build_training_cli_args_datasets_field_is_valid_json_with_one_dataset():
    config = resolve_config(override_path=None)
    cli_args = build_training_cli_args(config, data_dir="data/example_book", run_dir="runs/example")
    args_as_dict = args_list_to_dict(cli_args)

    datasets = json.loads(args_as_dict["datasets"])
    assert len(datasets) == 1
    assert datasets[0]["data_class"] == "JSONLinesDataset"
    assert datasets[0]["data_path"] == "data/example_book"
    assert datasets[0]["files"]["train"] == ["train.jsonl"]
    assert datasets[0]["files"]["val"] == ["val.jsonl"]
    assert datasets[0]["files"]["test"] == ["test.jsonl"]


def test_build_training_cli_args_dataset_post_process_function_points_at_io_functions():
    config = resolve_config(override_path=None)
    cli_args = build_training_cli_args(config, data_dir="data/example_book", run_dir="runs/example")
    args_as_dict = args_list_to_dict(cli_args)

    datasets = json.loads(args_as_dict["datasets"])
    post_process_functions = datasets[0]["post_process_functions"]
    assert len(post_process_functions) == 1
    assert post_process_functions[0]["class_path"] == "stair.core.train.io_functions.prepare_input_output_stair"
    assert post_process_functions[0]["init_args"]["toc_json"] == "data/example_book/toc.json"
    assert post_process_functions[0]["init_args"]["output_col"] == "{reference}"


def test_build_training_cli_args_lora_settings_come_from_config():
    config = resolve_config(override_path=None)
    cli_args = build_training_cli_args(config, data_dir="data/example_book", run_dir="runs/example")
    args_as_dict = args_list_to_dict(cli_args)

    assert args_as_dict["lora_rank"] == str(config["model"]["lora"]["rank"])
    assert args_as_dict["lora_alpha"] == str(config["model"]["lora"]["alpha"])
    assert args_as_dict["lora_dropout"] == str(config["model"]["lora"]["dropout"])
    lora_target_modules = json.loads(args_as_dict["lora_target_modules"])
    assert lora_target_modules == config["model"]["lora"]["target_modules"]


def test_build_training_cli_args_optimizer_field_carries_configured_learning_rate():
    config = resolve_config(override_path=None)
    cli_args = build_training_cli_args(config, data_dir="data/example_book", run_dir="runs/example")
    args_as_dict = args_list_to_dict(cli_args)

    optimizer = json.loads(args_as_dict["optimizer"])
    assert optimizer["lr"] == config["training"]["learning_rate"]


def test_build_training_cli_args_produces_a_valid_training_args_object(tmp_path):
    # this feeds build_training_cli_args's real output through the vendored
    # parser's actual value-parsing and pydantic construction, so a future
    # regression like a str field getting a "none"/"null" value, or a
    # required trainer attribute never being supplied, gets caught here
    # instead of requiring a human to construct TrainingArgs by hand
    vendor_silt_dir = Path(__file__).resolve().parents[1] / "src" / "stair" / "vendor" / "silt"
    sys.path.insert(0, str(vendor_silt_dir))
    try:
        from arguments import parse_value
        from arguments import TrainingArgs

        config = resolve_config(override_path=None)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        run_dir = tmp_path / "run"

        cli_args = build_training_cli_args(config, data_dir=str(data_dir), run_dir=str(run_dir))

        parsed_config = {}
        index = 0
        while index < len(cli_args):
            key = cli_args[index].lstrip("-")
            value = cli_args[index + 1]
            parsed_config[key] = parse_value(value, key)
            index += 2

        # TrainingArgs.__init__ -> _post_init -> DatasetArgs._post_init does a
        # bare `import data` (a sibling module of arguments.py in the vendored
        # silt dir), so vendor_silt_dir must still be on sys.path for this
        # call, not just for the `from arguments import ...` statements above
        training_args = TrainingArgs(**parsed_config)
    finally:
        sys.path.remove(str(vendor_silt_dir))

    assert training_args.save_path == str(run_dir)
