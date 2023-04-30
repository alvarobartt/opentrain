import pytest

from opentrain.train import Train


@pytest.mark.usefixtures("file_id")
def test_train(file_id: str) -> None:
    trainer = Train(model="ada")
    with pytest.warns(UserWarning):
        fine_tune_id = trainer.train(file_id, epochs=1, batch_size=1)
    assert isinstance(fine_tune_id, str)
    assert fine_tune_id.startswith("ft-")
