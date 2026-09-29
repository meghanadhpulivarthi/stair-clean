import copy
import csv
import json
import os
import sys
import random
from ast import literal_eval
from functools import partial
from typing import Callable, List, Union

import numpy as np
import pandas as pd
from IPython.core.debugger import Pdb
from tqdm import tqdm

from .config import DatasetConfig

csv.field_size_limit(sys.maxsize)

class _LoadData:

    @classmethod
    def read_rag_data(
        cls,
        data_config: DatasetConfig,
        data_path: str, 
        tokenizer,
        split: str="",
        prompt: str="",
        num_passages: int=5
    ):
        def format_passages(passages: Union[List[str], str]) -> str:
            formatted_passages = []
            for i, passage in enumerate(passages):
                formatted_passages.append(f"<passage_{i}>{passage}</passage_{i}>")
            return "\n".join(formatted_passages)

        files = data_config.files[split] # type: ignore
        if isinstance(files, str):
            files = [files]
        data_files = [os.path.join(data_path, x) for x in files]
        all_examples = []
        for this_file in data_files:
            data = pd.read_json(this_file, lines=True, orient='records')
            for i, example in data.iterrows():
                ex = {}
                ex['id'] = i
                formatted_passages = format_passages(example['retrieved_passages'][:num_passages])
                if 'llama' not in tokenizer.name_or_path:
                    ex['messages'] = [{'role': 'user', 'content': f"{prompt}\n{formatted_passages}\nUser: {example['question']}"}]
                else:
                    ex['messages'] = [
                        {'role': 'system', 'content': f"{prompt}"},
                        {'role': 'user', 'content': f"User: {example['question']}"}
                        ]
                ex['output'] = example['answer'].strip()
                ex['question'] = example['question']
                ex['int_id_list'] = example['int_id_list']
                ex['retrieved_ids'] = example['retrieved_ids']
                ex['retrieved_passages'] = example['retrieved_passages'][:num_passages]
                ex['corrupt_retrieved_ids'] = example['corrupt_retrieved_ids']
                ex['overlap'] = "SOME_OVERLAP" if example['in_top5']==1 else "NO_OVERLAP"
                all_examples.append(ex)

        return all_examples


    @classmethod
    def read_qa_data(
        cls,
        data_config: DatasetConfig,
        data_path: str, 
        tokenizer,
        split: str="",
        prompt: str="",
    ):
        files = data_config.files[split] # type: ignore
        if isinstance(files, str):
            files = [files]
        data_files = [os.path.join(data_path, x) for x in files]
        all_examples = []
        for this_file in data_files:
            data = pd.read_json(this_file, lines=True, orient='records')
            for i, example in data.iterrows():
                ex = {}
                ex['id'] = i
                if 'llama' not in tokenizer.name_or_path:
                    ex['messages'] = [{'role': 'user', 'content': f"{prompt}\nUser: {example['question']}"}]
                else:
                    ex['messages'] = [
                        {'role': 'system', 'content': f"{prompt}"},
                        {'role': 'user', 'content': f"User: {example['question']}"}
                        ]
                ex['question'] = example['question']
                ex['output'] = example['answer'].strip()
                ex['int_id_list'] = example['int_id_list']
                ex['retrieved_ids'] = ""                         # No retrieved ids in this dataset                        
                ex['retrieved_passages'] = []                         # No retrieved ids in this dataset
                ex['corrupt_retrieved_ids'] = []                         # No retrieved ids in this dataset
                ex['overlap'] = "SOME_OVERLAP"
                all_examples.append(ex)

        return all_examples
        


    @classmethod
    def read_unanswerablility_data(
        cls,
        data_config,
        data_path,
        tokenizer,
        split="",
        variants: List[str] = None,
        prompts: List[str] = None, 
        unanswerable_response: dict = None
    ):

        if unanswerable_response is None:
            unanswerable_response = {
                'variant1': 'unanswerable',
                'variant2': 'IDK',
                'variant3': 'N/A' 
            }

        if variants is None: 
            variants = ["variant1"]
        if prompts is None:
            prompts = ["CoT-Prompt"]

        files = data_config.files[split]
        if isinstance(files,str):
            files = [files]
        data_files = [os.path.join(data_path,x) for x in files]
        #all_examples = {'variant': [], 'gold_class': [], 'prompt_type': [], 'id': [], 'messages': [],'output': []} 
        all_examples = []
        for this_file in data_files:
            data = json.load(open(this_file))
            #Pdb().set_trace()
            for variant in variants:
                this_variant = data[variant]
                for gold_class, examples in this_variant.items():
                    for this_example in examples:
                        for prompt_type in prompts:
                            ex = {}
                            #'variant': [], 'gold_class': [], 'prompt_type': [], 'id': [], 'messages': [],'output': []} 
                            ex['variant'] = variant
                            ex['gold_class'] = gold_class
                            ex['prompt_type'] = prompt_type
                            ex['id'] = f"{this_example['id']}-{prompt_type}"
                            messages = [{'role': 'user', 'content': this_example[prompt_type]}]
                            #messages.append({'role': 'assistant', 'content': })
                            ex['messages'] = messages
                            this_output = this_example['answer'].strip()
                            if this_output == '':
                                this_output = unanswerable_response[variant]

                            ex['output'] = this_output
                            all_examples.append(ex)

        return all_examples
        #
    @classmethod
    def read_text_files_for_cpt(
        cls,
        data_config,
        data_path,
        tokenizer,
        split,
    ):
        files = data_config.files[split]
        if isinstance(files,str):
            files = [files]
        data_files = [os.path.join(data_path,x) for x in files]
        all_files = []
        for this_file in data_files:
            if os.path.isdir(this_file):
                all_files.extend([os.path.join(this_file,x) for x in os.listdir(this_file)])
                #read all the files in it
            else:
                all_files.append(this_file)
        #        
        #
        examples = []
        for ind, this_file in enumerate(all_files):
            basename =os.path.basename(this_file) 
            examples.append(
                {
                    'fname': basename,
                    'document_id': basename,
                    'text': open(this_file).read(),
                    'title': open(this_file).readlines()[0].strip("#").strip()
                }
            )
        #
        return examples
    
    
def load_data(data_config: DatasetConfig, data_path, tokenizer, split) -> List[dict]:
    """filters the examples based on the filter_functions in data_config

    Args:
        data_config (DatasetConfig): dataset config
        examples (List[dict]): list of examples

    Returns:
        List[dict]: filtered list of examples
    """
    # Pdb().set_trace()
    data_function = data_config.read_raw_data # type: ignore
    function_name = getattr(_LoadData, data_function["class_path"])
    function_args = data_function.get("init_args",{})
    examples = function_name(data_config, data_path, tokenizer, split, **function_args)
    #
    return examples
