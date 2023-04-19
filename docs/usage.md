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

## 🤖 Predict

```python
import openai
from opentrain.predict import OpenAIPredict

openai.api_key = "<ADD_OPENAI_API_KEY_HERE>"

predict = OpenAIPredict(model="ada:ft-personal-2021-03-01-00-00-01")
predict.predict("I love to play ->")
```
