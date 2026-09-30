import json
from argparse import ArgumentParser
import os
import sys
from typing import Any, List, Union
import numpy as np
import torch
import transformers
import yaml
from peft import PromptTuningInit
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from IPython.core.debugger import Pdb
import pdb
from pydantic import BaseModel, ConfigDict
import ast
import re
import warnings

# from src.utils import print_rank_0

from constants import (
    DatasetConfigKeys,
    LearningRateScheduler,
    Mode,
    DatasetSplit,
)


class BaseArgs(BaseModel):
    def __init__(__pydantic_self__, **data: Any) -> None:
        super().__init__(**data)
        #__pydantic_self__.__config__.extra = Extra.allow
        __pydantic_self__._post_init()

    def _post_init(self) -> None:
        return


class ModelArgs(BaseArgs):
    # model name on huggingface hub
    model_name: str = None
    # model class on huggingface hub, for example: AutoModelForCausalLM, AutoModelForSeq2SeqLM
    model_class: str = None
    # dtype to use for training / inference
    #dtype: str = "float32"
    # add special tokens to the tokenizer
    # trust remote code for models that are not directly supported by HuggingFace yet
    # padding side
    #padding_side: PaddingSide = None
    # LoRA peft configurations
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    # Quantization method, applicable only for QLoRA
    # attention implementation (only works with GPTMegatronForCausalLM)
    #attention_implementation: AttentionImplementation = None
    weight_log_path: str = None
    resume_from_checkpoint: Union[bool, str] = False
    def _post_init(self) -> None:
        # model_name
        assert self.model_name is not None, "model_name cannot be None"
        # model_class


class InitializationArgs(BaseArgs):
    # random seed
    seed: int = 42
    deterministic: bool = False
    
    # path to load checkpoints
    load_paths: list = [] 
    load_suffixes: list = [] 
    load_path: str = None
    init_path: str = None
    suffix: str = ''
    num_gpus: int = 1
    task: str = None
    debug: bool = False

    def _post_init(self) -> None:
        pass 



class DatasetArgs(BaseArgs):
    # list of datasets to use

    data_dump_keys: List[str] = None
    datasets: List[dict] = []
    select_datasets_at_index: List[int] = []
    #data_config = None
    interleave_stopping_strategy: str = "all_exhausted"
    def _post_init(self) -> None:
        # datasets
        assert (
            self.datasets is not None and len(self.datasets) != 0
        ), "datasets cannot be None or an empty list"
        self._check_each_dataset_and_set_defaults()

        assert self.interleave_stopping_strategy in ['first_exhausted', 'all_exhausted']

    def _check_each_dataset_and_set_defaults(self) -> None:
        """checks whether the arguments specified in the config are valid"""

        import data as data_classes

        for i, data_config in enumerate(self.datasets):
            assert (
                DatasetConfigKeys.data_class.value in data_config
            ), f"{DatasetConfigKeys.data_class.value} is not specified for dataset at index {i}"
            # convert to string to the actual class type
            data_config[DatasetConfigKeys.data_class.value] = getattr(
                data_classes, data_config[DatasetConfigKeys.data_class.value]
            )

            # check data_sampling_proportion
            #assert (
            #    DatasetConfigKeys.data_sampling_proportion.value in data_config
            #    and isinstance(
            #        data_config[DatasetConfigKeys.data_sampling_proportion.value], int
            #    )
            #    and data_config[DatasetConfigKeys.data_sampling_proportion.value] > 0
            #), f"{DatasetConfigKeys.data_sampling_proportion.value} is not specified for dataset at index {i}"


class OptimizationArgs(BaseArgs):
    # optimizer
    optimizer: dict = {
        "lr": 1e-5,
        "weight_decay": 0.1,
        "betas": [0.9, 0.95],
        "eps": 1e-10,
    }
    # learning rate schedule
    lr_schedule: LearningRateScheduler = LearningRateScheduler.cosine
    # warmup steps
    warmup_steps: int = 200
    max_grad_norm: float = 1.0

    def _post_init(self) -> None:
        # optimizer
        pass



class LoggingArgs(BaseArgs):
    # logging directory for experiments
    logdir: str = None
    # aim repo, experiment logs are saved here
    aim_repo: str = None
    # name of the experiment
    experiment_name: str = None
    # which logger to use
    report_to: str = "tensorboard"


class DebuggingArgs(BaseArgs):
    # steps per print for memory logging etc for deepspeed
    steps_per_print: int = np.inf


class TrainingArgs(
    ModelArgs,
    InitializationArgs,
    DatasetArgs,
    OptimizationArgs,
    LoggingArgs,
    DebuggingArgs,
):
    model_config = ConfigDict(extra='allow')
    # path to save checkpoints
    save_path: str = None
    # number of training steps
    num_training_steps: int = -1 
    # gradient accumulation steps
    gradient_accumulation_steps: int = 1
    # interval for evaluation
    eval_interval: int = None
    # interval for checkpointing
    save_interval: float = None
    # batch size per GPU for ZeRO-DP
    batch_size_per_gpu: int = None
    # whether to use val dataset for validation during training
    force: int = 1
    batch_size_per_gpu_val: int = 1
    num_train_epochs: float = 3.0 
    logging_interval: float = 0.1 
    eval_strategy: str = 'epoch'
    save_strategy: str = 'epoch'
    early_stopping: bool = False
    early_stopping_patience: int = 5
    use_sampling_ratios: bool = False
    use_iterator: bool = False
    max_seq_length: int = 20000 
    packing: bool = False 

    def _post_init(self) -> None:
        ModelArgs._post_init(self)
        InitializationArgs._post_init(self)
        DatasetArgs._post_init(self)
        # OptimizationArgs._post_init(self)
        LoggingArgs._post_init(self)
        DebuggingArgs._post_init(self)
        #if self.packing > 0:
        #    self.packing = True
        #else:
        #    self.packing = False

        # save_path
        assert self.save_path is not None, "save_path cannot be None"

        # num_training_steps
        #assert ((self.num_training_steps > 0) or (self.num_train_epochs is not None)), "both num_training_steps and num_train_epochs cannot be None simultaneously"
        if self.num_training_steps > 0 and self.num_train_epochs > 0:
            print(f"Both num_training_steps {self.num_training_steps} and num_train_epochs {self.num_train_epochs} are non zero. num_training_steps will take precedence")
        # save_interval
        #assert self.save_interval is not None, "save_interval cannot be None"

        # eval_interval


        # batch_size_per_gpu
        assert self.batch_size_per_gpu is not None, "batch_size_per_gpu cannot be None"


class InferenceArgs(ModelArgs, InitializationArgs, DatasetArgs):
    model_config = ConfigDict(extra='allow')
    #model_config['ignored_types'] = True
    # batch size
    batch_size: int = None
    # sample or greedy
    do_sample: bool = None
    # max new tokens to generate
    max_new_tokens: int = None
    # temperature
    temperature: float = None
    # top k
    top_k: int = None
    # top p
    top_p: float = None
    # output dir
    output_dir: str = None
    stop_strings: List[str] = None
    split: DatasetSplit = DatasetSplit.test
    # whether to force overwrite of inference output
    force: int = 1
    def _post_init(self) -> None:
        ModelArgs._post_init(self)
        InitializationArgs._post_init(self)
        DatasetArgs._post_init(self)


        # batch_size
        assert self.batch_size is not None, "batch_size cannot be None"

        # max_new_tokens
        assert self.max_new_tokens is not None, "max_new_tokens cannot be None"

        # output_dir
        assert self.output_dir is not None, "output_dir cannot be None"


def add_args(arg_string = None):
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="path for the config")
    parser.add_argument(
        "--data_config", type=str, default=None, help="path for the data config"
    )
    parser.add_argument(
        "--model_config", type=str, default=None, help="path for the model config"
    )
    parser.add_argument(
        "--inference_config", type=str, default=None, help="path for the model config"
    )

    if arg_string is None:
        args, unknown_args = parser.parse_known_args()
    else:
        args, unknown_args = parser.parse_known_args(arg_string.split())

    return args, unknown_args


def replace_variables_and_parse(config_path, defaults):
    config_str = open(config_path, "r").read()

    for k, v in defaults.items():
        config_str = config_str.replace("${" + k + "}", str(v))
    #

    if config_path.endswith((".yaml", ".yml")):
        config = yaml.safe_load(config_str)
    elif config_path.endswith(".json"):
        config = json.loads(config_str)
    else:
        raise f"Unknown config file format {config_path}"

    return config

# report_to must reach TrainingArgs as the literal string "none": that is the
# value transformers maps to [] (reporting disabled). Without this exemption,
# parse_value() below would turn "none" into Python None, which makes
# transformers fall back to *all* installed reporting integrations instead.
NO_PARSE_KEYS = {"report_to"}

def parse_value(val, key=None):
    """Convert a value (usually a string) into a Python value if possible.
    Handles literals, JSON, comma-separated lists, and bracketed lists with or without quotes.
    """
    # if it's already not a string, return as-is
    if not isinstance(val, str):
        return val
    if key in NO_PARSE_KEYS:
        return val 

    s = val.strip()

    # common booleans as some people pass "true"/"false" or "True"/"False"
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    if s.lower() in ("null", "none"):
        return None

    # try ast.literal_eval first (safe for python literals like [], {}, 0.1, "str", 1)
    try:
        return ast.literal_eval(s)
    except Exception:
        pass

    # try JSON (handles true/false/null and quoted strings/lists)
    try:
        return json.loads(s)
    except Exception:
        pass

    # handle bracketed lists without quotes: [up_proj,gate_proj] or [ up_proj , gate_proj ]
    m = re.match(r"^\[(.*)\]$", s)
    if m:
        inner = m.group(1).strip()
        if inner == "":
            return []
        # split on commas and strip whitespace
        parts = [p.strip() for p in inner.split(",")]
        return parts

    # handle simple comma-separated list without brackets: "a,b,c"
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
        return parts

    # fallback: return original string
    return val


def read_defaults(args, unknown_args):
    defaults = {}

    # load defaults from config files (unchanged)
    for config_path in [
        args.config,
        args.data_config,
        args.model_config,
        args.inference_config,
    ]:
        if config_path is not None:
            if config_path.endswith((".yaml", ".yml")):
                file_defaults = yaml.safe_load(open(config_path)).get("defaults", {}) or {}
                defaults.update(file_defaults)
            elif config_path.endswith(".json"):
                file_defaults = json.load(open(config_path)).get("defaults", {}) or {}
                defaults.update(file_defaults)

    # parse pairwise unknown args into defaults (with type conversion)
    ind = 0
    print(unknown_args)
    cli_args = {}
    while ind < len(unknown_args) - 1:
        key, value = unknown_args[ind], unknown_args[ind + 1]
        key = key.lstrip("-")
        parsed_value = parse_value(value, key)
        cli_args[key] = parsed_value
        ind += 2

    return defaults, cli_args



def get_config(args, unknown_args, mode: Mode):
    defaults, cli_args = read_defaults(args, unknown_args)

    defaults.update(cli_args)
    config = {}

    if args.config is not None:
        config = replace_variables_and_parse(args.config, defaults)
        # return config

    if args.data_config is not None:
        data_config = replace_variables_and_parse(args.data_config, defaults)
        config.update(data_config)

    if args.model_config is not None:
        model_config = replace_variables_and_parse(args.model_config, defaults)
        config.update(model_config)

    if mode == Mode.inference:
        if args.inference_config is not None:
            inference_config = replace_variables_and_parse(
                args.inference_config, defaults
            )
            config.update(inference_config)

    for k, v in cli_args.items():
        if k in config and config[k] != v:
            warnings.warn(
                f"CLI argument '{k}={v}' overrides config value '{config[k]}'"
            )
        config[k] = v
    
    if "defaults" in config:
        del config["defaults"]
    # print_rank_0("############ .     CONFIG : ###############")
    # yaml.dump(config, sys.stdout)
    # print_rank_0("################ .   #########################")
    if mode == Mode.training:
        if not os.path.exists(config["save_path"]):
            os.makedirs(config["save_path"], exist_ok=True)
        #
        yaml.dump(
            config,
            open(os.path.join(config["save_path"], f"{mode.value}_config.yaml"), "w"),
        )
        json.dump(
            config,
            open(os.path.join(config["save_path"], f"{mode.value}_config.json"), "w"),
            indent=4,
        )

    return config


def print_args(args) -> None:
    """prints args

    Args:
        args : args
    """

    print("------------------------ arguments ------------------------")

    kv_list = []
    for k, v in vars(args).items():
        dots = "." * (48 - len(k))
        kv_list.append(f"{k} {dots} " + str(v))

    kv_list.sort(key=lambda x: x.lower())

    for kv in kv_list:
        print(kv)

    print("-------------------- end of arguments ---------------------")



def get_args(mode: Mode, arg_string=None) -> Union[TrainingArgs, InferenceArgs]:
    """get args for training / inference

    Args:
        mode (Mode): training / inference mode for running the program

    Returns:
        Union[TrainingArgs, InferenceArgs]: args for training / inference
    """

    args, unknown_args = add_args(arg_string)

    config = get_config(args, unknown_args, mode)

    if mode == Mode.training or mode == Mode.dump:
        args = TrainingArgs(**config)
    else:
        args = InferenceArgs(**config)


    print_args(args)

    return args

