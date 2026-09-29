import ast


SYSTEM_PROMPT = """
You are an AI assistant tasked with generating question and answer pairs for the given context.
Return only a Python list of dicts, no other text. Each dict has a "question" key and an "answer" key.
You should create the following number of question/answer pairs: {number_of_pairs}.

Guidelines:
- Answers should be accurate, clear, and directly based on the context.
- Do not repeat or rephrase the same question in multiple ways.
- Questions must be self-contained and understandable without external context.
"""

USER_PROMPT = """
Format:
[{{"question": "...", "answer": "..."}}]

Book: {book_title}
Section: {section_title}
Context:
{context}
"""


def build_messages(chunk, num_pairs, book_title):
    section_title = chunk["metadata"]["section_title"]
    context = chunk["text"]
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(number_of_pairs=num_pairs)},
        {
            "role": "user",
            "content": USER_PROMPT.format(book_title=book_title, section_title=section_title, context=context),
        },
    ]


def parse_qa_response(raw_content):
    try:
        parsed = ast.literal_eval(raw_content.strip())
    except (ValueError, SyntaxError):
        return None

    if not isinstance(parsed, list):
        return None

    return parsed


def generate_qa_pairs_for_chunk(chunk, llm_call, num_pairs, book_title, max_retries=3):
    section_num = chunk["doc_id"][0]
    messages = build_messages(chunk, num_pairs, book_title)

    for attempt in range(1, max_retries + 1):
        raw_content = llm_call(messages)
        parsed = parse_qa_response(raw_content)

        if parsed is not None:
            complete_items = [
                item
                for item in parsed
                if "question" in item and "answer" in item
            ]
            # note: only keep items with both question and answer; drop the rest
            # rather than raising, since a single malformed entry shouldn't sink
            # an otherwise-usable batch of QA pairs
            qa_pairs = [
                {"question": item["question"], "answer": item["answer"], "reference": [section_num]}
                for item in complete_items
            ]
            return qa_pairs

        # not raising here because a single malformed LLM response is expected
        # occasionally; retrying is cheap and usually succeeds
        print(f"QA generation attempt {attempt}/{max_retries} failed to parse for section {section_num}")

    print(f"QA generation gave up after {max_retries} attempts for section {section_num}")
    return []


def build_openai_compatible_llm_call(api_base, api_key, model_name):
    from openai import OpenAI

    client = OpenAI(base_url=api_base, api_key=api_key)

    def llm_call(messages):
        completion = client.chat.completions.create(model=model_name, messages=messages)
        return completion.choices[0].message.content

    return llm_call
