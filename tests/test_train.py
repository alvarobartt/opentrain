import pytest

from opentrain.train import OpenAITrainer


@pytest.mark.usefixtures("training_data")
def test_openai_trainer(training_data: list) -> None:
    trainer = OpenAITrainer(model="ada")
    with pytest.warns(UserWarning):
        fine_tune_id = trainer.train(training_data, epochs=1, batch_size=1)
    assert isinstance(fine_tune_id, str)
    assert fine_tune_id.startswith("ft-")
