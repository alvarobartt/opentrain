from dataclasses import dataclass
from typing import List, Union

try:
    from pydantic import BaseModel

    has_pydantic = True
except ImportError:
    has_pydantic = False

if has_pydantic:

    class _HyperParams(BaseModel):
        batch_size: int
        learning_rate_multiplier: float
        n_epochs: int
        prompt_loss_weight: float

    class _File(BaseModel):
        bytes: int
        created_at: int
        filename: str
        id: str
        object: str
        purpose: str
        status: str
        status_details: Union[str, None]

    class FineTune(BaseModel):
        created_at: int
        fine_tuned_model: Union[str, None]
        hyperparams: _HyperParams
        id: str
        model: str
        object: str
        organization_id: str
        result_files: list
        status: str
        training_files: List[_File]
        updated_at: int
        validation_files: List[_File]

    class PromptCompletion(BaseModel):
        prompt: str
        completion: str

else:

    @dataclass
    class _HyperParams:
        batch_size: int
        learning_rate_multiplier: float
        n_epochs: int
        prompt_loss_weight: float

    @dataclass
    class _File:
        bytes: int
        created_at: int
        filename: str
        id: str
        object: str
        purpose: str
        status: str
        status_details: Union[str, None]

    @dataclass
    class FineTune:
        created_at: int
        fine_tuned_model: Union[str, None]
        hyperparams: _HyperParams
        id: str
        model: str
        object: str
        organization_id: str
        result_files: list
        status: str
        training_files: List[_File]
        updated_at: int
        validation_files: List[_File]

    @dataclass
    class PromptCompletion:
        prompt: str
        completion: str
