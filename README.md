<div align="center">
  <h1>opentrain</h1>
  <p>
    <em>🚂 Fine-tune OpenAI models for text classification, question answering, and more</em>
  </p>
</div>

---

`opentrain` is a simple Python package to fine-tune OpenAI models for task-specific purposes such as text classification, token classification, or question answering.

## Usage

```python
import openai
from opentrain.train import OpenAITrainer
openai.api_key = "<ADD_OPENAI_API_KEY_HERE>"
trainer = OpenAITrainer(model="ada")
trainer.train(
    [
        {
            "prompt": "I love to play soccer ->",
            "completion": " soccer",
        },
        {
            "prompt": "I love to play basketball ->",
            "completion": " basketball",
        },
    ],
)
```
