from typing import Any

from pydantic import BaseModel

from typing import List

class DatasetConfig(BaseModel):
    def __init__(__pydantic_self__, **data: Any) -> None:
        super().__init__(**data)
        __pydantic_self__._post_init()

    def _post_init(self) -> None:
        return


class JSONLinesConfig(DatasetConfig):
    read_raw_data: dict = None
    post_process_functions: List[dict] = None
    
    files: dict = None
    data_name: str = None
    dump_dir: str = None
    #batch_keys: List = [] 
    #output_col: str = 'output' 
    def _post_init(self) -> None:
        return

