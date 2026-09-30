
from transformers import  TrainingArguments
from trl import SFTConfig

from transformers import AutoModelForCausalLM

def prepare_adapters(model: AutoModelForCausalLM, active_adapters, freeze_adapters):
    """Set active adapters and freeze some adapters."""
    model.set_adapter(active_adapters)
    if len(freeze_adapters) > 0:
        for name, param in model.named_parameters():
            if any([freeze_adapter in name for freeze_adapter in freeze_adapters]):
                param.requires_grad = False

def get_training_args(args, **kwargs):
    #max_length, packing=False):
    
    ckpt_base_dir = args.save_path
    output_dir = f"{ckpt_base_dir}"
    intial_lr = args.optimizer['lr']
    batch_size_train = args.batch_size_per_gpu
    batch_size_val = args.batch_size_per_gpu_val  
    gradient_accumulation_steps = args.gradient_accumulation_steps
    num_train_epochs = args.num_train_epochs
    logging_steps = args.logging_interval
    training_args = SFTConfig(
        #packing=packing,
        output_dir=output_dir,
        dataloader_drop_last=False,
        num_train_epochs=num_train_epochs,
        max_steps=args.num_training_steps,
        save_steps=args.save_interval,
        eval_steps=args.eval_interval,
        eval_strategy=args.eval_strategy, #'epoch',
        save_strategy=args.save_strategy, #'epoch',
        save_total_limit=args.save_total_limit,
        logging_steps=logging_steps,
        per_device_train_batch_size=batch_size_train,
        per_device_eval_batch_size=batch_size_val,
        optim="adamw_torch_fused",
        learning_rate=intial_lr,
        lr_scheduler_type=args.lr_schedule,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        fp16=False,
        bf16=True,
        weight_decay=args.optimizer['weight_decay'],
        run_name=args.experiment_name,
        report_to=args.report_to,
        ddp_find_unused_parameters=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_first_step=True,
        log_level="info",
        # save_safetensors removed: transformers 5.x (the version locked in
        # uv.lock) dropped this TrainingArguments/SFTConfig kwarg — safetensors
        # is now the default save format — and passing it raises TypeError at
        # config construction. Kept behaviour identical (still saves safetensors).
        save_only_model=False,
        save_on_each_node=True,
        eval_on_start=True,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        **kwargs
        #max_seq_length=max_length
    )

    return training_args
