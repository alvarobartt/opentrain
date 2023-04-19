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
