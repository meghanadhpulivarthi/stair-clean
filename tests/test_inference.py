from stair.core.eval.inference import build_generate_call, choose_device_and_dtype


class FakeTensor:
    # Minimal stand-in for a real torch.Tensor's batch-of-token-ids shape.
    # A real HF tokenizer returns input_ids as a tensor with a .shape
    # attribute (shape[0] = batch size, shape[1] = sequence length), not a
    # plain nested list, and generate_call relies on that .shape access to
    # compute prompt_length. Wrapping the nested list here keeps the test
    # dependency-free (no real torch import) while still exercising the
    # same interface generate_call actually uses.
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))
        # a real torch.Tensor's .to(device) call is a no-op for tests that
        # don't care about device placement, but generate_call always calls
        # it, so the fake must support it
        self.to_calls = []

    def __getitem__(self, index):
        return self.rows[index]

    def to(self, device):
        self.to_calls.append(device)
        return self


class FakeTokenizer:
    def __init__(self, prompt_token_ids=None, pad_token_id=None, eos_token_id=99):
        self.applied_messages = None
        self.decode_calls = []
        # defaults to a 3-token prompt; tests that care about prompt-length
        # slicing pass a different length explicitly
        self.prompt_token_ids = prompt_token_ids if prompt_token_ids is not None else [1, 2, 3]
        # many real causal-LM tokenizers have no distinct pad token
        # (pad_token_id is None) and generate_call must fall back to
        # eos_token_id in that case
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.applied_messages = messages
        return "RENDERED_PROMPT"

    def __call__(self, text, return_tensors):
        return {"input_ids": FakeTensor([self.prompt_token_ids])}

    def decode(self, token_ids, skip_special_tokens):
        self.decode_calls.append(token_ids)
        return '["1 Introduction"]'


class FakeModel:
    def __init__(self):
        self.generate_calls = []
        self.device = "cpu"

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return [[1, 2, 3, 4, 5]]


def test_build_generate_call_renders_chat_template_from_messages():
    fake_model = FakeModel()
    fake_tokenizer = FakeTokenizer()
    generate_call = build_generate_call(fake_model, fake_tokenizer, max_new_tokens=64)

    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    generate_call(messages)

    assert fake_tokenizer.applied_messages == messages


def test_build_generate_call_passes_max_new_tokens_to_generate():
    fake_model = FakeModel()
    fake_tokenizer = FakeTokenizer()
    generate_call = build_generate_call(fake_model, fake_tokenizer, max_new_tokens=64)

    generate_call([{"role": "user", "content": "hi"}])

    assert fake_model.generate_calls[0]["max_new_tokens"] == 64


def test_build_generate_call_returns_decoded_text():
    fake_model = FakeModel()
    fake_tokenizer = FakeTokenizer()
    generate_call = build_generate_call(fake_model, fake_tokenizer, max_new_tokens=64)

    result = generate_call([{"role": "user", "content": "hi"}])

    assert result == '["1 Introduction"]'


def test_build_generate_call_computes_prompt_length_from_input_ids_shape():
    # Regression test for the first-draft bug: generate_call must derive
    # prompt_length from input_ids.shape[1], not from a "prompt_length"
    # dict key (a real HF tokenizer output has no such key). Using a
    # 5-token prompt here means decode() only sees the 2 genuinely new
    # tokens if shape-based slicing is used correctly.
    fake_model = FakeModel()
    fake_tokenizer = FakeTokenizer(prompt_token_ids=[1, 2, 3, 4, 5])
    fake_model.generate = lambda input_ids, max_new_tokens, **kwargs: [[1, 2, 3, 4, 5, 6, 7]]

    generate_call = build_generate_call(fake_model, fake_tokenizer, max_new_tokens=64)
    generate_call([{"role": "user", "content": "hi"}])

    assert fake_tokenizer.decode_calls[0] == [6, 7]


def test_build_generate_call_passes_do_sample_false_for_deterministic_output():
    # Regression test for the critical finding: without an explicit
    # do_sample=False, generate() falls back to the loaded model's own
    # generation_config.json, which for the bundled default model
    # (Qwen2.5-0.5B-Instruct) has sampling enabled, making eval
    # non-deterministic and actively hurting exact-syntax output.
    fake_model = FakeModel()
    fake_tokenizer = FakeTokenizer()
    generate_call = build_generate_call(fake_model, fake_tokenizer, max_new_tokens=64)

    generate_call([{"role": "user", "content": "hi"}])

    assert fake_model.generate_calls[0]["do_sample"] is False


def test_build_generate_call_passes_tokenizer_pad_token_id_when_present():
    fake_model = FakeModel()
    fake_tokenizer = FakeTokenizer(pad_token_id=7, eos_token_id=99)
    generate_call = build_generate_call(fake_model, fake_tokenizer, max_new_tokens=64)

    generate_call([{"role": "user", "content": "hi"}])

    assert fake_model.generate_calls[0]["pad_token_id"] == 7


def test_build_generate_call_falls_back_to_eos_token_id_when_no_pad_token():
    # many causal-LM tokenizers have no distinct pad token at all
    fake_model = FakeModel()
    fake_tokenizer = FakeTokenizer(pad_token_id=None, eos_token_id=99)
    generate_call = build_generate_call(fake_model, fake_tokenizer, max_new_tokens=64)

    generate_call([{"role": "user", "content": "hi"}])

    assert fake_model.generate_calls[0]["pad_token_id"] == 99


def test_build_generate_call_moves_input_ids_to_model_device():
    fake_model = FakeModel()
    fake_model.device = "cuda"
    fake_tokenizer = FakeTokenizer()
    generate_call = build_generate_call(fake_model, fake_tokenizer, max_new_tokens=64)

    generate_call([{"role": "user", "content": "hi"}])

    moved_input_ids = fake_model.generate_calls[0]["input_ids"]
    assert moved_input_ids.to_calls == ["cuda"]


def test_choose_device_and_dtype_uses_cuda_and_bfloat16_when_available():
    import torch

    device, dtype = choose_device_and_dtype(cuda_is_available=True)

    assert device == "cuda"
    assert dtype == torch.bfloat16


def test_choose_device_and_dtype_falls_back_to_cpu_and_float32():
    import torch

    device, dtype = choose_device_and_dtype(cuda_is_available=False)

    assert device == "cpu"
    assert dtype == torch.float32
