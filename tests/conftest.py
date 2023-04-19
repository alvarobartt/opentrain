import os

import openai


def pytest_sessionstart():
    openai.api_key = os.getenv("OPENAI_API_KEY")
