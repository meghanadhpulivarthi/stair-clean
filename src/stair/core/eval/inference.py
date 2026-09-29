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

        generated_token_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
        )

        new_token_ids = generated_token_ids[0][prompt_length:]
        generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
        return generated_text

    return generate_call


def load_finetuned_model(base_model_name, adapter_path):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    return model, tokenizer
