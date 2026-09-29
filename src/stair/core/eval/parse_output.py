import ast


def parse_predicted_sections(raw_text, title_to_id_map):
    try:
        parsed_entries = ast.literal_eval(raw_text.strip())
    except (ValueError, SyntaxError):
        return [], 1

    if not isinstance(parsed_entries, list):
        return [], 1

    predicted_ids = []
    hallucination_count = 0
    for entry in parsed_entries:
        if not isinstance(entry, str):
            hallucination_count += 1
            continue
        if entry not in title_to_id_map:
            hallucination_count += 1
            continue
        predicted_ids.append(title_to_id_map[entry])

    return predicted_ids, hallucination_count
