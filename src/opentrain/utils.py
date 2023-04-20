import json
from pathlib import Path
from typing import List

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


def prepare_openai_dataset(data: list) -> str:
    """Prepare the training data for OpenAI, and save it to a JSONL file.

    Args:
        data: A list of dictionaries containing the training data, which MUST be
            formatted as prompt-completion pairs.

    Returns:
        The path to the JSONL file.
    """
    file_path = Path.cwd() / str(uuid4()) / "training_data.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")
    return file_path.as_posix()

