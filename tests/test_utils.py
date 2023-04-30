import json
import tempfile

from opentrain.utils import (
    prepare_openai_dataset,
    validate_openai_dataset,
)


def test_prepare_openai_dataset() -> None:
    data = [{"prompt": "Hello", "completion": "World"}]
    output_path = prepare_openai_dataset(data)
    assert isinstance(output_path, str)


def test_validate_openai_dataset() -> None:
    data = [{"prompt": "Hello", "completion": "World"}]
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as f:
        with open(f.name, "w") as f:
            for entry in data:
                json.dump(entry, f)
                f.write("\n")
        assert validate_openai_dataset(f.name)

    data = [{"not_prompt": "Hello", "not_completion": "World"}]
    with tempfile.NamedTemporaryFile(suffix=".jsonl") as f:
        with open(f.name, "w") as f:
            for entry in data:
                json.dump(entry, f)
                f.write("\n")
        assert validate_openai_dataset(f.name) is False
