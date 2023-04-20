import json
from pathlib import Path
from typing import List, Union
from uuid import uuid4

import openai


def list_fine_tunes(just_succeeded: bool = True) -> List[str]:
    """List all fine-tuned models in your OpenAI account.

    Args:
        just_succeeded: If True, only return models that have succeeded.

    Returns:
        A list of fine-tuned model IDs.
    """
    fine_tunes = openai.FineTune.list()["data"]
    if just_succeeded:
        return [
            fine_tune["fine_tuned_model"]
            for fine_tune in fine_tunes
            if fine_tune["status"] == "succeeded"
        ]
    return [fine_tune["fine_tuned_model"] for fine_tune in fine_tunes]


def prepare_openai_dataset(
    data: list, output_path: Union[str, Path, None] = None
) -> str:
    """Prepare the training data for OpenAI, and save it to a JSONL file.

    Args:
        data: A list of dictionaries containing the training data, which MUST be
            formatted as prompt-completion pairs.
        output_path: The path to the JSONL file. Defaults to None, which means that
            the $HOME/.cache will be used instead.

    Returns:
        The path to the JSONL file.
    """
    if output_path is None:
        output_path = Path.home() / ".cache" / "opentrain" / f"{uuid4()}.jsonl"
    if not isinstance(output_path, Path):
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")
    return output_path.as_posix()


def validate_openai_dataset(file_path: Union[str, Path]) -> bool:
    """Validate the training data for OpenAI.

    Args:
        file_path: The path to the training data.

    Returns:
        True if the training data is valid, False otherwise.
    """
    if isinstance(file_path, Path):
        file_path = file_path.as_posix()
    with open(file_path, "r") as f:
        for line in f:
            try:
                json_line = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {line} is not a JSON object.") from e
            if not ["prompt", "completion"] == list(json_line.keys()):
                return False
    return True
