from collections import defaultdict
from typing import Dict, Literal

import torch
import torch.nn as nn
from transformers import AutoTokenizer, PreTrainedTokenizer
from trl import SFTTrainer
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
import train_utils
from IPython.core.debugger import Pdb
class StackLoraTrainer(SFTTrainer):
    def __init__(self, *args, frozen_adapters=None,  **kwargs):
        
        super().__init__(*args, **kwargs)
        self.frozen_adapters = frozen_adapters
        if frozen_adapters is None:
            self.frozen_adapters = []

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        all_adapters = list(self.model.peft_config.keys())
        if len(all_adapters) == 1:
            super().save_model(os.path.join(output_dir), _internal_call)
            return 
        active_adapters = self.model.active_adapters()
        for this_adapter in all_adapters:
            self.model.set_adapter(this_adapter)
            super().save_model(os.path.join(output_dir,this_adapter), _internal_call)
        #
        #self.model.set_adapter(active_adapters)
        train_utils.prepare_adapters(self.model, active_adapters, self.frozen_adapters)
        