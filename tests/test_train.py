import pytest

from opentrain.train import Train


@pytest.mark.usefixtures("file_id")
def test_train(file_id: str) -> None:
    trainer = Train(model="ada")
    with pytest.warns(UserWarning):
        trainer.train(file_id, n_epochs=1, batch_size=1)
    assert isinstance(trainer.fine_tune_id, str)
    assert trainer.fine_tune_id.startswith("ft-")
