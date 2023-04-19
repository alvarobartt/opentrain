# 💻 Usage

## 🦾 Fine-tune

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