from dataclasses import dataclass

try:
    from pydantic import BaseModel

    has_pydantic = True
except ImportError:
    has_pydantic = False

if has_pydantic:

    class PromptCompletion(BaseModel):
        prompt: str
        completion: str

else:

    @dataclass
    class PromptCompletion:
        prompt: str
        completion: str
