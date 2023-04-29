import pytest

try:
    from pydantic import ValidationError

    has_pydantic = True
except ImportError:
    has_pydantic = False

from opentrain.schemas import PromptCompletion


def test_prompt_completion_schema() -> None:
    valid_schema = {"prompt": "Hello", "completion": "World"}
    assert PromptCompletion(**valid_schema)

    invalid_schema = {"not_prompt": "Hello", "not_completion": "World"}
    with pytest.raises(TypeError):
        PromptCompletion(**invalid_schema)

    if has_pydantic:
        invalid_values = {"prompt": 1, "completion": 1}
        with pytest.raises(ValidationError):
            PromptCompletion(**invalid_values)
