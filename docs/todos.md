# 🔮 v0.2.0 - TODOs

- [ ] Add `Typer` CLI e.g. `opentrain train ...`
- [ ] Add `Dataset` validation before actually uploading a `Dataset`/`File` to OpenAI.
- [ ] Add `Dataset.from_datasets`, `Dataset.to_datasets`, and `Dataset.to_records`.
- [ ] Add `fsspec` support for `Dataset.from_file`, and `Dataset.to_file`.
- [ ] Allow different input paths such as `pathlib.Path` or `os.path` in `Dataset.from_file`.
- [ ] Explore https://github.com/openai/openai-python/blob/c556584eff3b36c92278e6af62cfe02ebb68fb65/openai/api_resources/file.py#L218 to avoid uploading duplicated files to OpenAI.
- [ ] Add `Trainer.for_text_classification`, `Trainer.for_question_answering`, and more if applicable.
