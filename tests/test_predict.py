import pytest

from opentrain.predict import OpenAIPredict


@pytest.mark.usefixtures("fine_tune_id", "prompt")
def test_openai_predict_with_fine_tune_id(fine_tune_id: str, prompt: str) -> None:
    predict = OpenAIPredict(fine_tune_id=fine_tune_id)
    completion = predict(prompt)
    assert isinstance(completion, str)
    assert completion in ["pos", "neg"]


@pytest.mark.usefixtures("fine_tuned_model", "prompt")
def test_openai_predict_with_model(fine_tuned_model: str, prompt: str) -> None:
    predict = OpenAIPredict(model=fine_tuned_model)
    completion = predict(prompt)
    assert isinstance(completion, str)
    assert completion in ["pos", "neg"]
