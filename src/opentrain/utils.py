from typing import List

import openai


def list_fine_tunes(just_succeeded: bool = True) -> List[str]:
    fine_tunes = openai.FineTune.list()["data"]
    if just_succeeded:
        return [
            fine_tune["fine_tuned_model"]
            for fine_tune in fine_tunes
            if fine_tune["status"] == "succeeded"
        ]
    return [fine_tune["fine_tuned_model"] for fine_tune in fine_tunes]
