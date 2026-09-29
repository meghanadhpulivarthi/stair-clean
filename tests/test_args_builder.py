import json

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
    assert args_as_dict["report_to"] == "none"


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
