import torch


def choose_device_and_dtype(cuda_is_available):
    # matches training's own dtype choice for CUDA
    # (see src/stair/vendor/silt/training_fsdp_trainer.py, torch_dtype=torch.bfloat16);
    # eval falls back to float32 on CPU since eval (unlike training) does
    # support a CPU path, just a slow one, so it should not force bfloat16
    # onto a CPU that may not handle it well
    if cuda_is_available:
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def build_generate_call(model, tokenizer, max_new_tokens):
    def generate_call(messages):
        rendered_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokenized_input = tokenizer(rendered_prompt, return_tensors="pt")
        input_ids = tokenized_input["input_ids"]
        # a real HF tokenizer's output has no "prompt_length" key; the
        # prompt's token length is the sequence-length dimension of the
        # input_ids tensor itself (shape = [batch_size, sequence_length])
        prompt_length = input_ids.shape[1]

        # read the model's device at call time (works for both a plain
        # AutoModelForCausalLM and a PeftModel wrapping one) and move the
        # tokenized input onto it, so generation runs on GPU when the model
        # was loaded there instead of silently staying on CPU
        model_device = model.device
        input_ids = input_ids.to(model_device)

        attention_mask = tokenized_input.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model_device)

        # many causal-LM tokenizers have no distinct pad token; fall back to
        # eos_token_id so generate() has a defined pad_token_id for
        # single-sequence generation instead of warning/behaving oddly
        if tokenizer.pad_token_id is not None:
            pad_token_id = tokenizer.pad_token_id
        else:
            pad_token_id = tokenizer.eos_token_id

        generate_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            # explicit do_sample=False makes generation greedy/deterministic
            # regardless of what the loaded model's own generation_config.json
            # says (the bundled default model, Qwen2.5-0.5B-Instruct, ships
            # with do_sample=True) - eval output must be exact
            # Python-list-of-strings syntax, so sampling only hurts
            "do_sample": False,
            "pad_token_id": pad_token_id,
        }
        if attention_mask is not None:
            generate_kwargs["attention_mask"] = attention_mask

        generated_token_ids = model.generate(**generate_kwargs)

        new_token_ids = generated_token_ids[0][prompt_length:]
        generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
        return generated_text

    return generate_call


def load_finetuned_model(base_model_name, adapter_path):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, dtype = choose_device_and_dtype(torch.cuda.is_available())
    print(f"Loading base model: {base_model_name}")
    print(f"Device: {device}, dtype: {dtype}")

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype)
    base_model = base_model.to(device)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    return model, tokenizer
