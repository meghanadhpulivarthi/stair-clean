from collections import defaultdict
import glob
import os
from copy import deepcopy
from typing import List, Set, Tuple, Type, Union
from uuid import uuid4

import jsonlines
import json
import numpy as np
import torch
from transformers import AutoTokenizer

from arguments import InferenceArgs, TrainingArgs
from constants import DUMMY, DatasetConfigKeys, DatasetKeys, DatasetSplit, Mode
from .post_process_data import post_process_data
import pdb
from tqdm import tqdm
#from IPython.core.debugger import Pdb
import pandas as pd
from .config import JSONLinesConfig
from .load_data import load_data
from IPython.core.debugger import Pdb

class BaseDataset(torch.utils.data.Dataset):
    """BaseDataset class to be implemented by all the datasets"""

    def __init__(
        self,
        args: Union[TrainingArgs, InferenceArgs],
        split: DatasetSplit,
        mode: Mode,
        tokenizer: AutoTokenizer,
        is_encoder_decoder: bool,
    ) -> None:
        super().__init__()

        self.split = split
        self.mode = mode

        self.tokenizer = tokenizer

        data_config: dict = args.data_config

        self.data_name: str = data_config.get(DatasetConfigKeys.data_name.value)
        self.data_path: str = data_config.get(DatasetConfigKeys.data_path.value)

        #self.data_config: dict = drop_common_args(deepcopy(data_config))
        self.data_config: dict = deepcopy(data_config)
        self.data_config_dict: dict = deepcopy(data_config)

        self.examples = []

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        if self.data_name is not None:
            example[DatasetConfigKeys.data_name.value] = self.data_name
        return example

    def __len__(self) -> int:
        return len(self.examples)

class JSONLinesDataset(BaseDataset):
    """A dataset for loading JSON lines files"""

    def __init__(
        self,
        args: Union[TrainingArgs, InferenceArgs],
        split: DatasetSplit,
        mode: Mode,
        tokenizer: AutoTokenizer,
        is_encoder_decoder: bool,
    ) -> None:
        super().__init__(args, split, mode, tokenizer, is_encoder_decoder)
        config_class = JSONLinesConfig
        self.num_samples = self.data_config.get('num_samples', -1)
        self.data_config = config_class(**self.data_config)
        #self.data_config is no longer a dict after this. It is an object of class JSONLinesConfig 
        self.examples = self.prepare_examples()
        print(f"{len(self.examples)} examples in {self.split.value} split")
        dump_file = os.path.join(f"{self.data_config.dump_dir}", f"{self.split.value}_{self.data_name}.json")

        self.dump_data(dump_file)

        # Pdb().set_trace()
        
        print(f"Done reading {self.data_name}")

    def dump_data(self, dump_file):
        
        if (self.data_config.dump_dir is not None) and (self.mode != Mode.inference):
            dirname = os.path.dirname(dump_file)
            os.makedirs(dirname, exist_ok=True)
            with open(dump_file, "w") as fh:
                dump_examples = []
                #dump_keys = ["id", "type", "context", "response", "document", "evidence"]
                for ex in self.examples:
                    #dump_ex = {}
                    dump_ex = ex 
                    #for key in dump_keys:
                    #    dump_ex[key] = ex.get(key, "NA")
                    #
                    #dump_ex[DatasetKeys.preprocessed_input.value] = self.tokenizer.decode(
                    #    ex[DatasetKeys.preprocessed_input.value]
                    #)
                    #dump_ex[DatasetKeys.preprocessed_output.value] = self.tokenizer.decode(
                    #    ex[DatasetKeys.preprocessed_output.value]
                    #)

                    #processed_cols = [DatasetKeys.preprocessed_input.value, DatasetKeys.preprocessed_output.value, DatasetKeys.preprocessed_base_model_output.value]
                    #for col in processed_cols:
                    #    if col in ex:
                    #        dump_ex[col] = self.tokenizer.decode(ex[col])
                    # print(ex['nlogprobs_idk'])
                    dump_examples.append(dump_ex)

                json.dump(dump_examples, fh, indent=5)



    def prepare_examples(self) -> List[dict]:
        #data_file = os.path.join(self.data_path, self.data_config.files[self.split.value])
        # pdb.set_trace()
        if self.split.value in self.data_config.files:
            files = self.data_config.files[self.split.value]
        else:
            return [] 

        if self.data_config.read_raw_data is not None:
            raw_examples = load_data(self.data_config, self.data_path, self.tokenizer, self.split.value)
        else:
            if isinstance(self.data_config.files[self.split.value],str):
                files = [files]
            
            data_files = [os.path.join(self.data_path,x) for x in files]
            raw_examples = []
            for data_file in data_files: 
                print(f"Loading data from {data_file}")
                if data_file.endswith('.csv'):
                    this_examples = pd.read_csv(data_file)
                else:
                    print(data_file)
                    this_examples = pd.read_json(data_file,lines=data_file.endswith('.jsonl'))
                this_examples = this_examples.to_dict(orient='records')
                raw_examples.extend(this_examples)
                #with open(data_file, "r") as f:
                #    this_raw_examples = [json.loads(x) for x in f]
                #    raw_examples.extend(this_raw_examples)

        print(f"No. of examples before post process = {len(raw_examples)}")
        raw_examples = post_process_data(self.data_config, self.tokenizer, self.split.value, raw_examples)
        print(f"No. of examples AFTER post process = {len(raw_examples)}")
        

        for ind, raw_example in enumerate(tqdm(raw_examples)):
            check_raw_example(raw_example, self.mode)
            if DatasetKeys.id.value not in raw_example:
                raw_example[DatasetKeys.id.value] = generate_random_id(self.__class__)
            #
            if (self.num_samples != -1) and (ind >= self.num_samples):
                return raw_examples[:ind] 
        #
        return raw_examples


class ConcatenatedDatasets(torch.utils.data.Dataset):
    """Concatenated list of datasets for training or inference"""

    def __init__(
        self,
        args: Union[TrainingArgs, InferenceArgs],
        split: DatasetSplit,
        mode: Mode,
        tokenizer: AutoTokenizer,
        is_encoder_decoder: bool,
    ) -> None:
        super().__init__()

        self.split = split
        self.mode = mode

        self.tokenizer = tokenizer
        self.is_encoder_decoder = is_encoder_decoder

        self.datasets, self.data_sampling_proportion = self.get_datasets_list(args)

        num_examples_in_each_dataset = self.get_num_examples_in_each_dataset()
        self.num_examples = sum(num_examples_in_each_dataset)
        self.start_indices = np.cumsum([0] + num_examples_in_each_dataset[:-1]).tolist()

        self.datasets_key_value_to_add = self.get_dataset_keys()

        self.print_dataset_stats()

    def get_dataset_keys(self) -> Set[str]:
        """gets the combined set of keys in each dataset

        Returns:
            Set[str]: set of keys in all the datasets
        """

        dataset_keys = []
        dataset_key_value_to_add = []
        all_keys = set()

        for dataset in self.datasets:
            example = dataset[0]
            dataset_keys.append(example.keys())
            all_keys.update(example.keys())

        for keys in dataset_keys:
            keys_to_add = all_keys.difference(keys)
            dataset_key_value_to_add.append({k: DUMMY for k in keys_to_add})

        return dataset_key_value_to_add

    def get_datasets_list(self, args: Union[TrainingArgs, InferenceArgs]) -> Tuple[List[BaseDataset], List[int]]:
        """prepare all the datasets

        Args:
            args (Union[TrainingArgs, InferenceArgs]): arguments based on training / inference mode
            split (DatasetSplit): dataset split to use
            mode (Mode): training / inference mode
            tokenizer (AutoTokenizer): tokenizer to use
            is_encoder_decoder (bool): whether the model is decoder-only or encoder-decoder

        Returns:
            Tuple[List[BaseDataset], List[int]]: list of all datasets, data sampling proportion
        """

        datasets = []
        data_sampling_proportion = []

        for ind, data_config in enumerate(args.datasets):
            if (len(args.select_datasets_at_index) == 0) or (ind in args.select_datasets_at_index):
                args_copy = deepcopy(args)
                args_copy.data_config = deepcopy(data_config)
                del args_copy.datasets

                dataset = args_copy.data_config[DatasetConfigKeys.data_class.value](
                    args_copy, self.split, self.mode, self.tokenizer, self.is_encoder_decoder
                )

                if len(dataset) > 0:
                    datasets.append(dataset)
                    data_sampling_proportion.append(data_config.get(DatasetConfigKeys.data_sampling_proportion.value,1))
            else:
                print("Skipping dataset at ind", ind) 
        return datasets, data_sampling_proportion

    def get_num_datasets(self) -> int:
        """returns the number of datasets in the mixture

        Returns:
            int: number of datasets in the mixture
        """

        return len(self.datasets)

    def get_num_examples_in_each_dataset(self) -> List[int]:
        """returns the number of examples in each dataset component

        Returns:
            List[int]: the number of examples in each dataset component
        """

        return [len(dataset) for dataset in self.datasets]

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, index: int) -> dict:
        num_datasets = self.get_num_datasets()

        # get the dataset the example belongs to
        dataset_index = num_datasets - 1
        for i in range(num_datasets):
            if index < self.start_indices[i]:
                dataset_index = i - 1
                break

        # get the position of the example in the specific dataset
        index -= self.start_indices[dataset_index]

        # get the example
        example = self.datasets[dataset_index][index]
        example[DatasetKeys.data_class_index.value] = get_data_class_index(
            self.datasets[dataset_index].__class__, dataset_index
        )

        # update the example with dummy values if the some specific keys are absent in a dataset
        example.update(self.datasets_key_value_to_add[dataset_index])

        return example

    def print_dataset_stats(self) -> None:
        """prints the statistics of all the datasets"""

        print(f"{'-' * 25} {self.split.value} {'-' * 25}")

        print(f"number of datasets = {self.get_num_datasets()}")
        print(f"total examples in the entire dataset mixture = {len(self)}\n")

        for dataset in self.datasets:
            print(f"examples in {dataset.__class__.__name__} ({dataset.data_name}) = {len(dataset)}")

        print("-" * 57)


def generate_random_id(data_class: Type[BaseDataset]) -> str:
    """generates a random unique ID for every example

    Args:
        data_class (Type[BaseDataset]): dataset class to which the example belongs

    Returns:
        str: randomly generated ID
    """

    return f"{uuid4()}:{data_class.__name__}"


def get_data_class_index(data_class: Type[BaseDataset], index: int) -> str:
    """get dataset label in the list (concatnated name and index in list)

    Args:
        data_class (Type[BaseDataset]): specific data class
        index (int): position in the list

    Returns:
        str: unique dataset label in the list
    """

    return f"{data_class.__name__}:{index}"

def check_raw_example(raw_example: dict, mode: Mode) -> None:
    """checks whether the dataset has conflicting fields

    Args:
        raw_example (dict): example to check
        mode (Mode): training / inference mode for running the program
    """

    assert (
        DatasetKeys.data_class_index.value not in raw_example
    ), "data_class_index found in the dataset, please drop this field"

    if mode == Mode.inference:
        assert (
            DatasetKeys.generated_text.value not in raw_example
        ), "generated_text found in the dataset, please drop this field"

def drop_common_args(data_config: dict) -> dict:
    for key in DatasetConfigKeys:
        if key.value in data_config:
            del data_config[key]

    return data_config
