import ast
import json


def load_toc(toc_json_path):
    with open(toc_json_path, "r") as toc_file:
        toc = json.load(toc_file)

    title = toc["title"]
    toc_lines = []
    id_to_title_map = {}
    for node in toc["table_of_contents"]:
        toc_lines.append(f"{node['section_num']} {node['title']}")
        id_to_title_map[node["id"]] = f"{node['section_num']} {node['title']}"

    return title, "\n".join(toc_lines), id_to_title_map


def prepare_input_output_stair(
    data_config,
    examples,
    tokenizer,
    split,
    system_prompt,
    toc_json,
    user_prompt,
    output_col,
    num_samples=1.0,
    repetitions=1,
    **kwargs
):
    book_title, toc_text, id_to_title_map = load_toc(toc_json)

    if num_samples <= 1.0:
        keep_until_index = int(num_samples * len(examples))
    else:
        keep_until_index = int(num_samples)

    prepared_examples = []
    for example_index, example in enumerate(examples):
        if example_index >= keep_until_index:
            break

        prompt_fields = dict(example)
        prompt_fields.update(kwargs)
        prompt_fields["title"] = book_title
        prompt_fields["toc"] = toc_text

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(**prompt_fields)},
        ]

        reference_ids_text = output_col.format(**prompt_fields).lstrip("\n")
        reference_ids = ast.literal_eval(reference_ids_text)

        reference_titles = []
        for reference_id in reference_ids:
            if reference_id not in id_to_title_map:
                raise KeyError(
                    f"reference id {reference_id!r} not found in table of contents at {toc_json}"
                )
            reference_titles.append(id_to_title_map[reference_id])

        quoted_titles = []
        for reference_title in reference_titles:
            quoted_titles.append(f'"{reference_title}"')
        completion = "\n[" + ", ".join(quoted_titles) + "]"

        prepared_example = {}
        prepared_example["messages"] = messages
        prepared_example["prompt"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prepared_example["completion"] = completion
        prepared_example["prompt_type"] = prompt_fields.get("prompt_type", "generic")
        prepared_example["meta_data"] = prompt_fields
        prepared_examples.append(prepared_example)

    if repetitions > 1:
        repeated_examples = []
        for prepared_example in prepared_examples:
            for _ in range(repetitions):
                repeated_examples.append(dict(prepared_example))
        prepared_examples = repeated_examples

    return prepared_examples
