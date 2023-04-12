import json
import sys
from pathlib import Path
from typing import Union

import openai


class OpenAITrainer:
    def __init__(self, model: str = "ada") -> None:
        self.model = model

    def train(
        self,
        path_or_buf: Union[str, Path, list],
        epochs: int = 10,
        batch_size: int = 32,
    ) -> dict:
        if isinstance(path_or_buf, list):
            file_path = self._prepare_training_data(path_or_buf)
        elif isinstance(path_or_buf, Path):
            file_path = path_or_buf.as_posix()
        else:
            file_path = path_or_buf

        upload_response = openai.File.create(
            file=open(file_path, "rb"),
            purpose="fine-tune",
        )
        file_id = upload_response.id
        fine_tune_response = openai.FineTune.create(
            training_file=file_id,
            model=self.model,
            n_epochs=epochs,
            batch_size=batch_size,
        )
        self.fine_tune_id = fine_tune_response.id
        for event in openai.FineTune.stream_events(id=self.fine_tune_id):
            sys.stdout.write(f"\r{event['event']}")
        return openai.FineTune.retrieve(id=self.fine_tune_id).fine_tuned_model

    def _prepare_training_data(self, buf: list) -> str:
        file_path = Path.cwd() / "training_data.jsonl"
        with open(file_path, "w") as f:
            for entry in buf:
                json.dump(entry, f)
                f.write("\n")
        return file_path.as_posix()
