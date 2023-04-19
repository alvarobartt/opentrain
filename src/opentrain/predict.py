from typing import Union

import openai


class OpenAIPredict:
    """A class for generating text from a prompt using OpenAI.

    Attributes:
        model: The model to use for prediction.

    Example:
        >>> from opentrain import OpenAIPredict
        >>> predict = OpenAIPredict(fine_tune_id="<FINE_TUNE_ID>")
        >>> predict("This text should be classified as -> ")
        "neutral"
    """

    def __init__(
        self, model: Union[str, None] = None, fine_tune_id: Union[str, None] = None
    ) -> None:
        """Initialize `OpenAIPredict`.

        Args:
            model: The model to use for prediction, that can be either an existing OpenAI
                model or a fine-tuned model.
            fine_tune_id: The fine-tune ID to use for prediction.
        """
        assert (
            model or fine_tune_id
        ), "You must provide either a `model` or a `fine_tune_id`."
        assert not (
            model and fine_tune_id
        ), "You must provide either a `model` or a `fine_tune_id`, not both."

        self.model = (
            self._model_from_fine_tune_id(fine_tune_id) if fine_tune_id else model
        )

    def __call__(self, prompt: str, max_tokens: int = 1) -> str:
        """Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from.
            max_tokens: The maximum number of tokens to generate.

        Returns:
            The generated text.
        """
        response = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        return response.choices[0].text

    def _model_from_fine_tune_id(self, fine_tune_id: str) -> str:
        """Get the model from a fine-tune ID.

        Args:
            fine_tune_id: The fine-tune ID.

        Returns:
            The name of the fine-tuned model.

        Raises:
            ValueError: If the model is not fine-tuned yet.
        """
        model = openai.FineTune.retrieve(fine_tune_id).fine_tuned_model
        if model is None:
            raise ValueError(
                "The model is not yet ready. Please wait a few minutes and try again."
            )
        return model
