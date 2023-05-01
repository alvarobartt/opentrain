from typing import Dict, Union

from opentrain.dataset import Dataset, File

DatasetType = Union[
    str, Dict[str, str], Dataset, Dict[str, Dataset], File, Dict[str, File]
]
