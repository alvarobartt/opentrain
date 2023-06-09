import warnings
from typing import Any, Dict, Iterator

import openai

from opentrain.dataset import Dataset
from opentrain.typing import DatasetType

warnings.simplefilter("once", category=UserWarning)

DEFAULT_OPENAI_MODELS = ["ada", "babbage", "curie", "davinci"]


class Train:
    """The `Train` class is a wrapper around OpenAI's FineTune API, and it also
    uses `opentrain.Dataset` to use OpenAI files for training/fine-tuning.

    Args:
        model: the OpenAI model name to be used for training/fine-tuning.

    Attributes:
        model: the OpenAI model name to be used for training/fine-tuning.

    Examples:
        >>> from opentrain import Train, Dataset
        >>> trainer = Train(model="curie")
        >>> dataset = Dataset(file_id="file-1234")
        >>> trainer.train(dataset, n_epochs=5, batch_size=32)
        >>> trainer.track()

        >>> from opentrain import Train
        >>> trainer = Train(model="curie")
        >>> trainer.train(
            {
                "train": "file-1234",
                "eval": "file-5678",
            },
            n_epochs=5,
            batch_size=32
        )
        >>> trainer.track()
    """

    def __init__(self, model: str) -> None:
        """Initializes the `Train` class.

        Args:
            model: the OpenAI model name to be used for training/fine-tuning.
        """
        assert model in DEFAULT_OPENAI_MODELS, (
            "Invalid OpenAI model, it must be one of the following:"
            f" {','.join(DEFAULT_OPENAI_MODELS)}."
        )
        self.model = model

    def train(
        self,
        dataset: DatasetType,
        **kwargs,
    ) -> None:
        """Trains/Fine-tunes the OpenAI model with the given dataset/s.

        Args:
            dataset: the dataset/s to be used for training/fine-tuning and/or evaluating it.
            **kwargs: the keyword arguments to be passed to the OpenAI FineTune API. See
                https://platform.openai.com/docs/api-reference/fine-tunes
        """
        if isinstance(dataset, str):
            train_file_id = dataset
            validation_file_id = None
        elif isinstance(dataset, Dataset):
            train_file_id = dataset.file_id
            validation_file_id = None
        elif isinstance(dataset, dict):
            train_file = dataset.get("train")
            if train_file:
                train_file_id = (
                    train_file.file_id
                    if isinstance(train_file, Dataset)
                    else train_file
                    if isinstance(train_file, str)
                    else None
                )
            else:
                train_file_id = None
            validation_file = dataset.get("eval")
            if validation_file:
                validation_file_id = (
                    validation_file.file_id
                    if isinstance(validation_file, Dataset)
                    else validation_file
                    if isinstance(validation_file, str)
                    else None
                )
            else:
                validation_file_id = None

        if not train_file_id:
            raise ValueError(
                "You must provide at least a training dataset to be used for training"
                " the model."
            )

        fine_tune_args = {
            "training_file": train_file_id,
            "model": self.model,
            **kwargs,
        }

        if validation_file_id:
            fine_tune_args["validation_file"] = validation_file_id

        fine_tune_response = openai.FineTune.create(**fine_tune_args)
        self.fine_tune_id = fine_tune_response.id

        warnings.warn(
            "Since the OpenAI API may take from minutes to hours depending on the size"
            " of the training data, then from now on, you'll be able to check its"
            " progress via the following command: `openai api fine_tunes.follow -i"
            f" {self.fine_tune_id}`. Once the training is completed, then you'll be"
            " able to use `Inference` with the either the fine tune id returned,"
            " or from the model name generated by OpenAI linked to your account.",
            stacklevel=2,
        )

    def fine_tune(self, dataset: DatasetType, **kwargs) -> None:
        """This function is just a wrapper around `train` with the same
        functionality. It's just here to keep the same naming convention as OpenAI.

        Args:
            dataset: the dataset/s to be used for training/fine-tuning and/or evaluating it.
            **kwargs: the keyword arguments to be passed to the OpenAI FineTune API. See
                https://platform.openai.com/docs/api-reference/fine-tunes
        """
        self.train(dataset, **kwargs)

    def track(self) -> Iterator[Dict[str, Any]]:
        """Tracks the progress of the training/fine-tuning process.

        Returns:
            A list of events containing the progress of the training/fine-tuning
            process.

        Raises:
            ValueError: if the model training/fine-tuning hasn't started yet.
        """
        if not self.fine_tune_id:
            raise ValueError(
                "You must call `train` before `track`, since nothing will be tracked as"
                " the training/fine-tuning hasn't started yet."
            )
        return openai.FineTune.stream_events(self.fine_tune_id)


class FineTune(Train):
    pass
