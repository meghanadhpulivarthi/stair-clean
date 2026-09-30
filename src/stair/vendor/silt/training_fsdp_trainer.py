import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, AutoConfig
from peft import LoraConfig, PeftModel, get_peft_model
import torch.nn.functional as F
import json
import pandas as pd
import datasets
from datasets import Dataset, interleave_datasets
import random
from accelerate import Accelerator
from arguments import get_args
from constants import DatasetKeys, DatasetSplit, Mode
from data import ConcatenatedDatasets
from IPython.core.debugger import Pdb
import train_utils
from trainers import StackLoraTrainer
from accelerate.utils import KwargsHandler, InitProcessGroupKwargs
from transformers import EarlyStoppingCallback, TrainerCallback

from datetime import timedelta
datasets.disable_caching()
kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
accelerator = Accelerator(kwargs_handlers=[kwargs])


def get_datasets_from_concat_dataset(concat_dataset, tokenizer):
    return_dataset = []

    for this_dataset in concat_dataset.datasets:
        return_dataset.extend(this_dataset.examples)
    #
    random.shuffle(return_dataset)

    return return_dataset


def interleave_using_ratios(train_dataset, args):
    ratios = train_dataset.data_sampling_proportion
    total = sum(ratios)
    ratios = [x / total for x in ratios]
    print("sampling ratios: ", ratios)
    individual_datasets = []

    for this_dataset in train_dataset.datasets:
        this_dataset = [x for x in this_dataset]
        random.shuffle(this_dataset)
        individual_datasets.append(Dataset.from_pandas(pd.DataFrame(this_dataset)))
    #
    print("Interleave strategy: ", args.interleave_stopping_strategy)
    dataset = interleave_datasets(
        individual_datasets,
        stopping_strategy=args.interleave_stopping_strategy,
        # stopping_strategy="all_exhausted",
        probabilities=ratios,
    )
    print("Length of interleave dataset:", len(dataset))

    return dataset

class GenerateTextCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, train_dataset):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.train_dataset = train_dataset

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        # Pick one example from the eval and train sets
        eval_sample = self.eval_dataset[0]  # or random.choice(...) for variety
        train_sample = self.train_dataset[0]
        
        # Tokenize the input text if it exists
        if "prompt" in eval_sample:
            eval_input_text = eval_sample["prompt"]
        else:
            raise ValueError("Eval sample must contain either 'text' or 'prompt' field")
        
        if "prompt" in train_sample:
            train_input_text = train_sample["prompt"]
        else:
            raise ValueError("Train sample must contain either 'text' or 'prompt' field")
            
        eval_input_ids = self.tokenizer.encode(
            eval_input_text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
            padding="max_length"
        ).to(model.device)

        train_input_ids = self.tokenizer.encode(
            train_input_text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
            padding="max_length"
        ).to(model.device)

        # Generate output
        eval_generated_ids = model.generate(
            input_ids=eval_input_ids,
            max_new_tokens=args.max_length,
            do_sample=False,
        )

        train_generated_ids = model.generate(
            input_ids=train_input_ids,
            max_new_tokens=args.max_length,
            do_sample=False,
        )

        eval_generated_text = self.tokenizer.decode(eval_generated_ids[0][eval_input_ids.shape[1]:], skip_special_tokens=True)
        train_generated_text = self.tokenizer.decode(train_generated_ids[0][train_input_ids.shape[1]:], skip_special_tokens=True)

        # mention the training step
        print("\n=== EVAL GENERATION ===")
        print("Training Step: ", state.global_step)
        print("Input: ", eval_input_text)
        print("Output:", eval_generated_text)
        print("Gold:", eval_sample["completion"])
        print("=======================\n")
        
        print("\n=== TRAIN GENERATION ===")
        print("Input: ", train_input_text)
        print("Output:", train_generated_text)
        print("Gold:", train_sample["completion"])
        print("=======================\n")


def main():

    mode = Mode.training
    args = get_args(mode)
    set_seed(args.seed, args.deterministic)
    run_name = args.experiment_name
    intial_checkpoint = args.model_name
    ckpt_base_dir = args.save_path
    intial_lr = args.optimizer["lr"]
    lora_r = args.lora_rank
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    target_modules = args.lora_target_modules
    batch_size_train = args.batch_size_per_gpu
    batch_size_val = args.batch_size_per_gpu_val
    gradient_accumulation_steps = args.gradient_accumulation_steps
    num_train_epochs = args.num_train_epochs
    logging_steps = args.logging_interval
    padding_side = args.padding_side
    load_suffixes = args.load_suffixes
    load_paths = args.load_paths
    debug = args.debug
    
    force = args.force
    suffix = args.suffix

    if accelerator.is_main_process:
        print("Initial Model: {}".format(intial_checkpoint))
        print("Checkpoint base dir path: {}".format(ckpt_base_dir))
        print(f"Training batch size: {batch_size_train}")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"Validation batch size: {batch_size_val}")
        print(f"Initial learning rate: {intial_lr}")
        print(f"LoRA rank: {lora_r}")
        print(f"LoRA alpha: {lora_alpha}")
        print(f"LoRA dropout rate: {lora_dropout}")
        print(f"Target modules: {target_modules}")
        print(f"num epochs: {num_train_epochs}")
        print(f"logging_step_ratio: {logging_steps}")
        print(f"padding_side: {padding_side}")
        print(f"suffix: {suffix}")
        print(f"load_suffixes: {load_suffixes}")
        print(f"LoRA Load Paths: {load_paths}")
        print(f"Debug: {debug}")
    
    if os.path.isdir(ckpt_base_dir) and any('checkpoint-' in dir for dir in os.listdir(ckpt_base_dir)) and force != 1:
        print("This stage has already been completed and --force is False. Starting next stage...")
        print(f"LoRA Load Paths {load_paths}")

    if (
        os.path.isdir(ckpt_base_dir)
        and any("checkpoint-" in dir for dir in os.listdir(ckpt_base_dir))
        and force != 1
    ):
        print(
            "This stage has already been completed and --force is False. Starting next stage..."
        )

        return

    tokenizer = AutoTokenizer.from_pretrained(intial_checkpoint)

    if tokenizer.pad_token_id is None:
        # tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = tokenizer.unk_token_id
        print(f"pad token id: {tokenizer.pad_token_id}")

    num_added_tokens = 0

    if (tokenizer.pad_token_id == tokenizer.eos_token_id) or (
        tokenizer.pad_token_id is None
    ):
        print(
            "Pad token id same as eos. Will interfare with training. Lets try to reuse the last added_token."
        )
        # select the largest token id
        selected = -1

        for token_id, token_detail in tokenizer.added_tokens_decoder.items():
            selected = max(selected, token_id)
        #
        tokenizer.pad_token = "[PAD]"
        tokenizer.pad_token_id = selected
        print("using the following token id as a pad token: ", tokenizer.pad_token_id)

        # special_tokens_dict = {"pad_token": "[PAD]"}
        # num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
        # print("We have added", num_added_tokens, "tokens")
        # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
        # model.resize_token_embeddings(len(tokenizer))

    # if padding_side == 'right':
    # tokenizer.padding_side = 'right'

    # tokenizer.add_bos_token = False

    model_config = AutoConfig.from_pretrained(intial_checkpoint, trust_remote_code=True)
    train_dataset = ConcatenatedDatasets(
        args,
        split=DatasetSplit.train,
        mode=mode,
        tokenizer=tokenizer,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    val_dataset = ConcatenatedDatasets(
        args,
        split=DatasetSplit.val,
        mode=mode,
        tokenizer=tokenizer,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )

    if not args.use_sampling_ratios:
        train_dataset = get_datasets_from_concat_dataset(train_dataset, tokenizer)
        train_dataset = Dataset.from_list(train_dataset)
    else:
        print("Interleave using ratios")
        train_dataset = interleave_using_ratios(train_dataset, args)

    val_dataset = get_datasets_from_concat_dataset(val_dataset, tokenizer)
    val_dataset = Dataset.from_list(val_dataset)
    columns = set(val_dataset.column_names)
    #target_columns = set(['preprocessed_input', 'preprocessed_output'])

    if debug:
        train_dataset = train_dataset.select(range(10))
        val_dataset = val_dataset.select(range(10))

    train_columns = set(train_dataset.column_names)
    target_columns = set(["prompt", "completion", "text"])
    columns_to_remove = list(train_columns - target_columns)
    train_dataset = train_dataset.remove_columns(columns_to_remove)

    val_columns = set(val_dataset.column_names)
    columns_to_remove = list(val_columns - target_columns)
    val_dataset = val_dataset.remove_columns(columns_to_remove)

    print("Train Set {}".format(train_dataset))
    print("Val Set {}".format(val_dataset))

    # max_length = 0

    # for i in val_dataset['preprocessed_input']:
    #    max_length = max(max_length, len(i))

    model = AutoModelForCausalLM.from_pretrained(
        intial_checkpoint,
        torch_dtype=torch.bfloat16,
        # use PyTorch's built-in scaled-dot-product attention instead of
        # flash_attention_2: flash-attn is not a declared dependency of this
        # project (not in pyproject.toml / uv.lock), so forcing FA2 made
        # `stair train` crash at model load out-of-the-box. sdpa ships with
        # torch, needs no extra package, and runs on the CUDA GPUs here.
        attn_implementation="sdpa",
    )

    if num_added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # if args.resume_from_checkpoint and isinstance(args.resume_from_checkpoint, str):
    #     print(f"Loading LoRA adapter from: {args.resume_from_checkpoint}")
    #     model = PeftModel.from_pretrained(
    #         model, 
    #         args.resume_from_checkpoint,
    #         is_trainable=True,
    #         adapter_name="default",
    #     )

    assert len(load_suffixes) == len(load_paths)

    if len(load_suffixes) == 0:
        if (
            (args.init_path is not None)
            and (args.init_path.lower() != "none")
            and (args.init_path.strip() != "")
        ):
            print("Initialize model from: ", args.init_path)
            model.load_adapter(args.init_path)
            train_utils.prepare_adapters(
                model, active_adapters=["default"], freeze_adapters=[]
            )
            # print(model.active_adapters)
            # TODO Check if load_adapter also results in setting active adpater and grads will flow
        else:
            model = get_peft_model(model, peft_config)
        print("Model loaded successfully!")
    else:
        for this_load_suffix, this_load_path in zip(load_suffixes, load_paths):
            print("Loading ", this_load_suffix, " at path ", this_load_path)
            # this_load_path = load_path + this_load_suffix
            # latest_ckpt = get_latest_checkpoint(this_load_path)
            # latest_ckpt = this_load_path
            # model = PeftModel.from_pretrained(model, latest_ckpt, is_trainable=False, adapter_name=this_load_suffix)
            # model = model.merge_and_unload(adapter_names=[this_load_suffix])
            model.load_adapter(this_load_path, adapter_name=this_load_suffix)
            print(f"Loaded {this_load_suffix} weights!")

        # model = get_peft_model(model, peft_config=peft_config, adapter_name=suffix)
        # model.set_adapter(suffix)
        model.add_adapter(peft_config, adapter_name="sft_lora")
        print("New adapter loaded successfully!")
        train_utils.prepare_adapters(model, load_suffixes + ["sft_lora"], load_suffixes)
        # model.set_adapter(load_suffixes + ['sft_lora'])
        # for name, param in model.named_parameters():
        #    if any([freeze_adapter in name for freeze_adapter in load_suffixes]):
        #        param.requires_grad = False

    # optimizer = AdamW(model.parameters(), lr=intial_lr)

    model.to("cuda")
    model.config.use_cache = False
    output_dir = f"{ckpt_base_dir}"

    max_length = tokenizer.model_max_length
    max_length = min(args.max_seq_length, tokenizer.model_max_length)

    print(f"Max length after truncating due to  model's max length: {max_length}")

    training_args = train_utils.get_training_args(
        args,
        max_length=max_length,
        packing=args.packing,
        eval_packing=False,
        load_best_model_at_end=args.load_best_model_at_end,
        max_grad_norm=args.max_grad_norm,
        completion_only_loss="prompt" in train_columns,
        ddp_timeout=18000,
    )
    
    if debug:
        training_args.eval_strategy = "steps"
        training_args.save_strategy = "steps"
        training_args.save_total_limit = None
        training_args.eval_steps = 2
        training_args.save_steps = 2
        training_args.max_steps = 5
        training_args.report_to = None

    callbacks = [GenerateTextCallback(tokenizer=tokenizer, eval_dataset=val_dataset, train_dataset=train_dataset)]
    if args.early_stopping:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
    trainer = StackLoraTrainer(
        frozen_adapters=load_suffixes,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
        # peft_config=peft_config,
        # max_seq_length=max_length,
        # packing=False,
        # dataset_kwargs={"skip_prepare_dataset": True}
    )
    # trainer.model.print_trainable_parameters()
    train_result = trainer.train(args.resume_from_checkpoint)

    trainer.accelerator.print(f"{trainer.model}")
    train_result.metrics
    # saving final model

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(output_dir + "/checkpoint-best")

    # args_dict = vars(args)
    ## Path to the JSON file where the configuration will be saved
    # config_file_name = "training_config_fsdp_trainer.json"
    # config_file_path = os.path.join(output_dir, config_file_name)

    ## Write the configuration dictionary to the JSON file
    # with open(config_file_path, "w") as json_file:
    #    json.dump(args_dict, json_file, indent=4)

    print(f"Checkpoint save directory: {output_dir}")


if __name__ == "__main__":
    main()
