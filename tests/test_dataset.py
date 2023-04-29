import json
import tempfile

import pytest

from opentrain.dataset import Dataset


class TestDatasetFromRecords:
    @pytest.fixture(autouse=True)
    @pytest.mark.usefixtures("training_data")
    def setup_method(self, training_data: list) -> None:
        self.dataset = Dataset.from_records(
            records=training_data, file_name="opentrain-test-dataset"
        )

    def teardown_method(self) -> None:
        self.dataset.delete()
        del self.dataset

    def test_info(self) -> None:
        info = self.dataset.info
        assert isinstance(info, dict)
        assert info["id"] == self.dataset.file_id
        assert info["object"] == "file"
        assert info["purpose"] == "fine-tune"


class TestDatasetFromFile:
    @pytest.fixture(autouse=True)
    @pytest.mark.usefixtures("training_data")
    def setup_method(self, training_data: dict) -> None:
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as f:
            with open(f.name, "w") as f:
                for record in training_data:
                    json.dump(record, f)
                    f.write("\n")
            self.dataset = Dataset.from_file(
                file_path=f.name, file_name="opentrain-test-dataset"
            )

    def teardown_method(self) -> None:
        self.dataset.delete()
        del self.dataset

    def test_info(self) -> None:
        info = self.dataset.info
        assert isinstance(info, dict)
        assert info["id"] == self.dataset.file_id
        assert info["object"] == "file"
        assert info["purpose"] == "fine-tune"
