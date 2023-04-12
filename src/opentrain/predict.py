from typing import Union

import openai


class OpenAIPredict:
    def __init__(
        self, model: Union[str, None] = None, fine_tune_id: Union[str, None] = None
    ) -> None:
        assert (
            model or fine_tune_id
        ), "You must provide either a `model` or a `fine_tune_id`."
        assert not (
            model and fine_tune_id
        ), "You must provide either a `model` or a `fine_tune_id`, not both."

        self.model = (
            model if model is not None else self._model_from_fine_tune_id(fine_tune_id)
        )

    def __call__(self, prompt: str, max_tokens: int = 1) -> str:
        response = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        return response.choices[0].text

    def _model_from_fine_tune_id(self, fine_tune_id: str) -> str:
        return openai.FineTune.retrieve(fine_tune_id).fine_tuned_model
