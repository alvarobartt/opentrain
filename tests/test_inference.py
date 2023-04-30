import pytest

try:
    from pydantic import BaseModel

    has_pydantic = True
except ImportError:
    has_pydantic = False

from opentrain.inference import Inference, list_fine_tunes


@pytest.mark.usefixtures("fine_tuned_model", "prompt")
def test_inference(fine_tuned_model: str, prompt: str) -> None:
    inference = Inference(fine_tuned_model)
    completion = inference(prompt, temperature=0.0, max_tokens=1)
    assert isinstance(completion, str)
    assert completion in ["pos", "neg"]


@pytest.mark.usefixtures("fine_tune_id", "prompt")
def test_inference_from_fine_tune_id(fine_tune_id: str, prompt: str) -> None:
    inference = Inference.from_fine_tune_id(fine_tune_id)
    completion = inference(prompt, temperature=0.0, max_tokens=1)
    assert isinstance(completion, str)
    assert completion in ["pos", "neg"]


def test_list_fine_tunes() -> None:
    fine_tunes = list_fine_tunes()
    assert isinstance(fine_tunes, list)
    if has_pydantic:
        assert isinstance(fine_tunes[0], BaseModel)
    else:
        assert hasattr(fine_tunes[0], "__dataclass_fields__")
