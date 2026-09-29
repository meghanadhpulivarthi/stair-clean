import re
from math import floor
import os
import random
import numpy as np
import pandas as pd
from IPython.core.debugger import Pdb

# from tqdm.notebook import tqdm
import pdb as pdb
from .config import DatasetConfig
from tqdm import tqdm

import copy
from transformers import AutoTokenizer
import yaml
from typing import List

import re
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
import importlib
import sys
import warnings

def remove_markdown_comments(md_text: str) -> str:
    """Removes all HTML-style comments from a markdown text."""

    return re.sub(r"<!--.*?-->", "", md_text, flags=re.DOTALL)



def create_chunks(
    book,
    tokenizer,
    chunk_size_list=None,
    chunk_overlap_list=None,
    title_name="Document title: ",
    title=None,
    data_identifier=None,
    **kwargs,
):
    if chunk_size_list is None:
        assert chunk_overlap_list is None
        chunk_size_list = [2048, 1024, 512, 256, 128]
        chunk_overlap_list = [2048 - 512, 1024 - 256, 512 - 128, 256 - 64, 128 - 32]

    chunk2node = {}

    for chunk_size, chunk_overlap in zip(chunk_size_list, chunk_overlap_list):
        node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        book_doc = Document(text=book)
        chunk2node[(chunk_size, chunk_overlap)] = node_parser.get_nodes_from_documents(
            [book_doc]
        )

    # cols = ['id','text','chunk_size','chunk_overlap','local_chunk_num','global_chunk_num','start_char_idx','end_char_idx']
    # tab = {}
    # for col in cols:
    #     tab[col] = []

    global_count = 0
    chunks = []
    min_tokens = min(chunk_size_list) / 2

    for k, v in chunk2node.items():
        for local_count, node in enumerate(v):
            num_tokens = len(tokenizer(node.text)["input_ids"])

            if num_tokens > min_tokens:
                this_text = node.text
                prefix = ""

                if data_identifier is not None:
                    prefix = f"{data_identifier}\n"

                if title is not None:
                    prefix = f"{prefix}\n\n{title_name} {title}\n\n"
                this_text = f"{prefix}{this_text}".strip()
                chunks.append(
                    {
                        "id": node.id_,
                        "text": this_text,
                        "input": "",
                        "output": "",
                        "chunk_size": k[0],
                        "chunk_overlap": k[1],
                        "local_chunk_num": local_count,
                        "global_chunk_num": global_count,
                        "start_char_idx": node.start_char_idx,
                        "end_char_idx": node.end_char_idx,
                        "num_tokens_wo_prefix": num_tokens,
                    }
                )

                for kwk, kwv in kwargs.items():
                    chunks[-1][kwk] = kwv
                # tab['id'].append(node.id_)
                # tab['text'].append(node.text)
                # tab['chunk_size'].append(k[0])
                # tab['chunk_overlap'].append(k[1])
                # tab['local_chunk_num'].append(local_count)
                # tab['global_chunk_num'].append(global_count)
                # tab['start_char_idx'].append(node.start_char_idx)
                # tab['end_char_idx'].append(node.end_char_idx)
                global_count += 1
    # return tab

    return chunks


class _PostProcessData:

    @classmethod
    def rename_columns(
        cls,
        data_config,
        examples: List[dict],
        tokenizer,
        split,
        rename=None,
        delete_columns=None,
    ):
        ex = pd.DataFrame(examples)

        if delete_columns is None:
            delete_columns = {}

        if rename is None:
            rename = {}

        for col in delete_columns:
            if col in ex.columns:
                del ex[col]

        ex = ex.rename(columns=rename)

        return ex.to_dict(orient="records")

    @classmethod
    def prepare_cpt_data(
        cls,
        data_config,
        examples: List[dict],
        tokenizer,
        split,
        text_column="md_document",
        title_name=None,
        title_column=None,
        data_identifier=None,
        chunk_size_list=None,
        chunk_overlap_list=None,
    ):
        if title_name is None:
            title_name = "Document title: "
        chunks = []

        for ex in tqdm(examples):
            text = ex[text_column]
            text = remove_markdown_comments(text)
            title = None

            if title_column is not None:
                title = ex[title_column]

            chunks.extend(
                create_chunks(
                    text,
                    tokenizer,
                    chunk_size_list,
                    chunk_overlap_list,
                    title_name,
                    title,
                    data_identifier,
                    document_id=ex.get("document_id", "0"),
                )
            )
        return chunks

    @classmethod
    def convert_to_cpt(
    cls,
        data_config,
        examples: List[dict],
        tokenizer,
        split
    ):
        for ex in examples:
            ex['text'] = f"{ex['prompt']}{ex['completion']}"
            del ex['prompt']
            del ex['completion']
        return examples

    @classmethod
    def apply_chat_template(
        cls,
        data_config,
        examples: List[dict],
        tokenizer,
        split,
        instruct_tokenizer=None,
        output_col = 'output'
    ):
        #TODO - should we use add_generation_prompt or not??

        use_tokenizer = tokenizer
        if tokenizer.chat_template is None:
            use_tokenizer = AutoTokenizer.from_pretrained(instruct_tokenizer)

        for ex in examples:
            ex['prompt'] = use_tokenizer.apply_chat_template(ex['messages'], tokenize=False, add_generation_prompt=True)
            ex['completion'] = '\n'+ex[output_col].lstrip('\n') 
            del ex['messages']

        print("Sample input: ")
        print(ex['prompt'])
        print("Sample output: ")
        print(ex['completion'])
        
        return examples

    @classmethod
    def create_auxilary_data(
        cls,
        data_config,
        examples: List[dict],
        tokenizer,
        split,
        data_processors,
        num_samples=None,
        num_repeat=1,
    ):
        return_examples = []

        for this_function in data_processors:
            function_name = getattr(_PostProcessData, this_function["class_path"])
            function_args = this_function.get("init_args", {})
            this_output = function_name(examples, **function_args)
            return_examples.extend(copy.deepcopy(this_output))

        if num_samples is not None:
            return_examples = return_examples[:num_samples]
        #
        return_examples = num_repeat * return_examples

        return return_examples


    @classmethod
    def remove_header_from_passages(
        cls,
        data_config,
        examples: List[dict],
        tokenizer,
        split,
        passages_col: str, 
    ):
        for ex in examples:
            ex[passages_col] = ['\n\n'.join(x.split('\n\n')[1:]) for x in ex[passages_col]]
        #
        return examples

    @classmethod
    def add_source_info(
        cls,
        data_config,
        examples: List[dict],
        tokenizer,
        split,
        retrieved_ids_col: str, 
        gold_ids_col: str,
        top_k: int = 5,
        source_info_templates: str = None
    ):

        source_info_templates_dict = {
            "NO_OVERLAP": ["The retrieved context does not contain information to answer the question. Answering from internal parametric knowledge."],
            "PARTIAL_OVERLAP": ["The information in the retrieved passages is not sufficient. Using information from internal knowledge as well."],
            "FULL_OVERLAP": ["Generating response using the information in the retrieved passages."],
        }
        source_counts = {"NO_OVERLAP": 0, "PARTIAL_OVERLAP": 0, "FULL_OVERLAP": 0}


        if source_info_templates is not None and source_info_templates.strip() != '':
            print("Reading source info templates from ", source_info_templates)
            source_info_templates_dict = yaml.safe_load(open(source_info_templates))

        print("Source info dict: ", source_info_templates_dict)
        # Pdb().set_trace()
        for ex in examples:
            retrieved_ids = ex[retrieved_ids_col][:top_k]
            gold_ids = ex[gold_ids_col]
            intersection = list(set(retrieved_ids).intersection(set(gold_ids)))
            difference = list(set(gold_ids).difference(set(intersection)))

            if len(intersection) == 0:
                src = random.choice(source_info_templates_dict["NO_OVERLAP"])
                source_counts["NO_OVERLAP"] += 1
            elif len(difference) == 0:
                src = random.choice(source_info_templates_dict["FULL_OVERLAP"])
                source_counts["FULL_OVERLAP"] += 1
            else:
                src = random.choice(source_info_templates_dict["PARTIAL_OVERLAP"])
                source_counts["PARTIAL_OVERLAP"] += 1
            #
            ex['source_info'] = src
        
        return examples

    @classmethod
    def rag_data(
        cls,
        data_config,
        examples: List[dict],
        tokenizer,
        split,
        input_output_data: List[dict],
        question_col: str,
        data_identifier: str = None,
        passages_col: str = None, # type: ignore
        top_k: int = 5,
        instruct_tokenizer=None,
    ):
        use_tokenizer = tokenizer
        if tokenizer.chat_template is None:
            use_tokenizer = AutoTokenizer.from_pretrained(instruct_tokenizer)
        
        if data_identifier is None:
            data_identifier = ''

        return_examples = []
        for ind, this_example in enumerate(examples):
            for io_format in input_output_data:
                messages = []
                documents = []
                this_prompt = io_format["prompt"].strip()

                if this_prompt != "":
                    messages.append({"role": "system", "content": this_prompt})

                if passages_col is not None:
                    doc_string = '\n\n'.join([f"<passage {x}>\n{y}\n</passage {x}>" for x,y in zip(range(1,top_k+1), this_example[passages_col][:top_k])])
                    messages[-1]['content'] = f"{messages[-1]['content']}\n{doc_string}"
                    
                messages.append(
                    {
                        "role": "user",
                        "content": f"{data_identifier} {this_example[question_col]}".strip(),
                    }
                )
                
                output_list = []

                if len(io_format.get("output_tags", [])) == 0:
                    for col in io_format["output_cols"]:
                        output_list.append(this_example[col])
                else:
                    for tag, col in zip(
                        io_format["output_tags"], io_format["output_cols"]
                    ):
                        output_list.append(f"<{tag}>\n{this_example[col]}\n</{tag}>")

                output = "\n\n".join(output_list)
                ex = copy.deepcopy(this_example) 
                ex['prompt'] = use_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                ex['completion'] = '\n'+output.lstrip('\n') 
                ex['prompt_type'] = io_format['prompt_type']
                #ex['source'] =this_example.get('source_info','')
                return_examples.append(ex)
        return return_examples

    @classmethod
    def generic_prepare_input_output(
        cls,
        data_config,
        examples: List[dict],
        tokenizer,
        split,
        input_format,
        output_format,
        num_samples=1.0,
        repetitions=1,
        **kwargs,
    ):
        examples = copy.deepcopy(examples)

        random.Random(42).shuffle(examples)

        if num_samples <= 1.0:
            keep_till = num_samples * len(examples)
        else:
            keep_till = num_samples

        selected_examples = []

        for ind, this_ex in enumerate(examples):
            if ind > keep_till:
                break
            this_ex.update(kwargs)
            ex = {}
            ex["prompt"] = input_format.format(**this_ex)
            ex["completion"] = '\n'+output_format.format(**this_ex).lstrip('\n')
            ex["prompt_type"] = this_ex.get("prompt_type", "generic")
            selected_examples.append(ex)

        if repetitions > 1:
            selected_examples = [
                copy.deepcopy(item)

                for item in selected_examples

                for _ in range(repetitions)
            ]
            random.Random(420).shuffle(selected_examples)

        return selected_examples

    @classmethod
    def subselect(
        cls,
        data_config,
        examples: List[dict],
        tokenizer,
        split,
        start_fraction=0,
        end_fraction=100,
    ):

        if (start_fraction == 0) and (end_fraction == 100):
            return examples
        #
        start_ind = floor(start_fraction * len(examples) / 100.0)
        end_ind = floor(end_fraction * len(examples) / 100.0)
        print("Start ind: ", start_ind, " End ind: ", end_ind)

        return examples[start_ind:end_ind]

    @classmethod
    def prepare_input_output_with_chat_template(
        cls,
        data_config,
        examples: List[dict],
        tokenizer,
        split,
        system_prompt,
        user_prompt,
        output_format,
        num_samples=1.0,
        repetitions=1,
        **kwargs,
    ):
        examples = copy.deepcopy(examples)

        random.Random(42).shuffle(examples)

        if num_samples <= 1.0:
            keep_till = num_samples * len(examples)
        else:
            keep_till = num_samples

        selected_examples = []

        use_tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        if use_tokenizer.chat_template is None:
            print("Chat template not found for tokenizer: ", tokenizer)
            

        for ind, this_ex in enumerate(examples):
            if ind > keep_till:
                break
            this_ex.update(kwargs)
            ex = {}
            if system_prompt != "None":
                ex['messages'] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt.format(**this_ex)}
                ]
            else:
                ex['messages'] = [
                    {"role": "user", "content": user_prompt.format(**this_ex)}
                ]
            ex['prompt'] = use_tokenizer.apply_chat_template(ex['messages'], tokenize=False, add_generation_prompt=True)
            ex["completion"] = '\n'+output_format.format(**this_ex).lstrip('\n')
            ex["prompt_type"] = this_ex.get("prompt_type", "generic")
            ex["meta_data"] = this_ex
            selected_examples.append(ex)

        if repetitions > 1:
            selected_examples = [
                copy.deepcopy(item)

                for item in selected_examples

                for _ in range(repetitions)
            ]
            random.Random(420).shuffle(selected_examples)

        return selected_examples


def post_process_data(
    data_config: DatasetConfig, tokenizer, split, examples: List[dict]
) -> List[dict]:
    """filters the examples based on the filter_functions in data_config

    Args:
        data_config (DatasetConfig): dataset config
        examples (List[dict]): list of examples

    Returns:
        List[dict]: filtered list of examples
    """
    post_process_functions = data_config.post_process_functions

    if post_process_functions is not None:
        for this_function in post_process_functions:
            func_path = this_function["class_path"]
            function_args = this_function.get("init_args", {})

            # First check built-in PostProcessData
            if hasattr(_PostProcessData, func_path):
                function_name = getattr(_PostProcessData, func_path)
            else:
                # Split into module and function/class name
                if "." not in func_path:
                    raise ValueError(
                        f"External class_path must be 'module.submodule.function', got '{func_path}'"
                    )

                module_name, func_name = func_path.rsplit(".", 1)
                
                # Ensure the root project directory is in sys.path
                root_dir = os.path.abspath(os.getcwd())  # or specify your project root explicitly
                if root_dir not in sys.path:
                    sys.path.append(root_dir)

                # Import the module
                user_module = importlib.import_module(module_name)

                # Get the function/class from the module
                if hasattr(user_module, func_name):
                    function_name = getattr(user_module, func_name)
                else:
                    raise ValueError(
                        f"Function/Class '{func_name}' not found in module '{module_name}'"
                    )
            
            # Check for duplicates with positional args
            positional_param_names = ["data_config", "examples", "tokenizer", "split"]
            duplicates = [k for k in positional_param_names if k in function_args]
            if duplicates:
                warnings.warn(
                    f"Function '{func_path}' init_args contains keys {duplicates} "
                    f"which conflict with positional arguments. These keys will be removed.",
                    RuntimeWarning
                )
                for k in duplicates:
                    function_args.pop(k)


            # Call the function
            examples = function_name(data_config, examples, tokenizer, split, **function_args)


    return examples
