from enum import Enum


DUMMY = "<DUMMY>"


class DatasetSplit(str, Enum):
    """dataset split"""

    train = "train"
    val = "val"
    test = "test"


class Mode(str, Enum):
    """training / inference mode/ create data dump"""

    training = "training"
    inference = "inference"
    dump = "dump"


#class PaddingSide(str, Enum):
#    """padding side for the tokenizer"""
#
#    left = "left"
#    right = "right"

class LearningRateScheduler(str, Enum):
    """learning rate schedule"""

    linear = "linear"
    cosine = "cosine"
    constant_with_warmup = "constant_with_warmup"

class DatasetConfigKeys(str, Enum):
    """standard keys in the dataset"""

    data_class = "data_class"
    data_name = "data_name"
    data_path = "data_path"
    data_sampling_proportion = "data_sampling_proportion"
    

class DatasetKeys(str, Enum):
    """standard keys in the dataset"""

    id = "id"
    #input = "input"
    #output = "output"
    #preprocessed_input = "preprocessed_input"
    #preprocessed_output = "preprocessed_output"
    generated_text = "generated_text"
    num_generated_tokens = "num_generated_tokens"
    data_class_index = "data_class_index"
    #preprocessed_base_model_output = "preprocessed_base_model_output"
    #base_model_output = "base_model_output"
    #idk_response = "idk_response"
    #idk_input = "idk_input"


#class TrainingInferenceType(str, Enum):
#    """training method"""
#
#    full_finetuning = "full_finetuning"
#    prompt_tuning = "prompt_tuning"
#    lora_finetuning = "lora_finetuning"
#    qlora_finetuning = "qlora_finetuning"
#
#    # Convenience functions
#    @staticmethod
#    def isLoRA(ops):
#        return ops in [TrainingInferenceType.lora_finetuning, TrainingInferenceType.qlora_finetuning]
#
#    @staticmethod
#    def isFinetuning(ops):
#        return ops in [
#            TrainingInferenceType.full_finetuning,
#            TrainingInferenceType.lora_finetuning,
#            TrainingInferenceType.qlora_finetuning,
#        ]
#

#class QuantizationType(str, Enum):
#    """training method"""
#
#    int8 = "int8"
#    fp4 = "fp4"
#    nf4 = "nf4"
#
#    # Convenience functions
#    @staticmethod
#    def is4bit(ops):
#        return ops in [QuantizationType.fp4, QuantizationType.nf4]
#
#    @staticmethod
#    def isInt8(ops):
#        return ops == QuantizationType.int8
#
#
#class AttentionImplementation(str, Enum):
#    math = "math"
#    flash = "flash"
#    sdpa = "sdpa"
#