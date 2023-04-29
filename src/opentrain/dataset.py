import json
import warnings
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Union
from uuid import uuid4

import openai
from openai.error import TryAgain

FILE_SIZE_WARNING = 500 * 1024 * 1024


class Dataset:
    def __init__(self, file_id: str, organization: Union[str, None] = None) -> None:
        self.file_id = file_id
        self.organization = organization

    @cached_property
    def info(self) -> Dict[str, Any]:
        return openai.File.retrieve(id=self.file_id, organization=self.organization)

    def download(self) -> bytes:
        warnings.warn(
            "Dataset.download() is just available for paid/pro accounts, so bear in"
            " mind that this will fail if you're using a free tier.",
            stacklevel=2,
        )
        return openai.File.download(id=self.file_id, organization=self.organization)

    def to_file(self, output_path: str) -> None:
        content = self.download()
        with open(output_path, "wb") as f:
            f.write(content)
        del content

    def delete(self) -> bool:
        try:
            openai.File.delete(
                sid=self.file_id, organization=self.organization, request_timeout=30
            )
            return True
        except TryAgain:
            return False

    @classmethod
    def from_file(
        cls,
        file_path: str,
        file_name: Union[str, None] = None,
        organization: Union[str, None] = None,
    ) -> "Dataset":
        upload_response = openai.File.create(
            file=open(file_path, "rb"),
            organization=organization,
            purpose="fine-tune",
            user_provided_filename=file_name,
        )
        return cls(file_id=upload_response.id, organization=organization)

    @classmethod
    def from_records(
        cls,
        records: List[Dict[str, str]],
        file_name: Union[str, None] = None,
        organization: Union[str, None] = None,
    ) -> "Dataset":
        local_path = (
            Path.home() / ".cache" / "opentrain" / f"{file_name or uuid4()}.jsonl"
        )
        local_path.parent.mkdir(parents=True, exist_ok=True)

        with open(local_path.as_posix(), "w") as f:
            for record in records:
                json.dump(record, f)
                f.write("\n")

        if local_path.stat().st_size > FILE_SIZE_WARNING:
            warnings.warn(
                f"Your file is larger than {FILE_SIZE_WARNING / 1024 / 1024} MB, and"
                " the maximum total upload file size in OpenAI is 1GB, so please be"
                " aware that if you already have uploaded files to OpenAI this might"
                " fail. If you need to upload larger files or require more space,"
                " please contact OpenAI as suggested at"
                " https://platform.openai.com/docs/api-reference/files/upload.",
                stacklevel=2,
            )

        upload_response = openai.File.create(
            file=open(local_path.as_posix(), "rb"),
            organization=organization,
            purpose="fine-tune",
            user_provided_filename=file_name,
        )
        return cls(file_id=upload_response.id, organization=organization)


class File(Dataset):
    pass


def list_datasets(organization: Union[str, None] = None) -> List[Dataset]:
    return [
        Dataset(file_id=file.id, organization=organization)
        for file in openai.File.list(organization=organization)
    ]


def list_files(organization: Union[str, None] = None) -> List[File]:
    return [
        File(file_id=file.id, organization=organization)
        for file in openai.File.list(organization=organization)
    ]
