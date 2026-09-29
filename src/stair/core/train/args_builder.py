import json

from stair.core.prompts import SYSTEM_PROMPT, USER_PROMPT


def build_dataset_config(config, data_dir):
    system_prompt = SYSTEM_PROMPT
    user_prompt = USER_PROMPT

    dataset_config = {
        "data_class": "JSONLinesDataset",
        "data_name": "stair",
        "data_path": data_dir,
        "files": {
            "train": ["train.jsonl"],
            "val": ["val.jsonl"],
            "test": ["test.jsonl"],
        },
        "post_process_functions": [
            {
                "class_path": "stair.core.train.io_functions.prepare_input_output_stair",
                "init_args": {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "toc_json": data_dir + "/toc.json",
                    "output_col": "{reference}",
                },
            }
        ],
    }
    return dataset_config


def build_optimizer_config(training_config):
    optimizer_config = {
        "lr": training_config["learning_rate"],
        "weight_decay": 0.05,
        "betas": [0.9, 0.95],
        "eps": 1.0e-10,
    }
    return optimizer_config


def build_training_cli_args(config, data_dir, run_dir):
    model_config = config["model"]
    lora_config = model_config["lora"]
    training_config = config["training"]

    dataset_config = build_dataset_config(config, data_dir)

    field_values = {
        "model_name": model_config["name"],
        "save_path": run_dir,
        "batch_size_per_gpu": training_config["batch_size_per_gpu"],
        "gradient_accumulation_steps": training_config["gradient_accumulation_steps"],
        "num_train_epochs": training_config["num_train_epochs"],
        "max_seq_length": training_config["max_seq_length"],
        "eval_strategy": training_config["eval_strategy"],
        "save_strategy": training_config["eval_strategy"],
        "eval_interval": training_config["eval_interval"],
        "save_interval": training_config["save_interval"],
        "save_total_limit": training_config["save_total_limit"],
        "early_stopping": training_config["early_stopping"],
        "early_stopping_patience": training_config["early_stopping_patience"],
        "num_gpus": training_config["num_gpus"],
        "lora_rank": lora_config["rank"],
        "lora_alpha": lora_config["alpha"],
        "lora_dropout": lora_config["dropout"],
        "lora_target_modules": lora_config["target_modules"],
        "optimizer": build_optimizer_config(training_config),
        "lr_schedule": training_config["lr_schedule"],
        "warmup_steps": training_config["warmup_steps"],
        # "no" (not "none") — the vendored parser's parse_value() converts any
        # CLI value that case-insensitively equals "null"/"none" into Python
        # None before it reaches pydantic, but TrainingArgs.report_to is a
        # plain str field, so "none" would crash with a ValidationError.
        # "no" survives parse_value() unchanged and is what HuggingFace's
        # underlying training-arguments machinery accepts to mean "no
        # reporting integrations".
        "report_to": "no",
        "padding_side": training_config["padding_side"],
        "load_best_model_at_end": training_config["load_best_model_at_end"],
        "metric_for_best_model": training_config["metric_for_best_model"],
        "greater_is_better": training_config["greater_is_better"],
        "datasets": [dataset_config],
    }

    cli_args = []
    for field_name, field_value in field_values.items():
        cli_args.append("--" + field_name)
        if isinstance(field_value, (list, dict)):
            cli_args.append(json.dumps(field_value))
        else:
            cli_args.append(str(field_value))

    return cli_args
