# 💻 Usage

## 📦 Data management

```python
import openai
from opentrain import Dataset

openai.api_key = "<ADD_OPENAI_API_KEY_HERE>"

dataset = Dataset.from_file("data.jsonl")
dataset.info
dataset.download(output_path="downloaded-data.jsonl")
```

## 🦾 Fine-tune

```python
import openai
from opentrain import Train

openai.api_key = "<ADD_OPENAI_API_KEY_HERE>"

trainer = Train(model="ada")
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
from opentrain import Inference

openai.api_key = "<ADD_OPENAI_API_KEY_HERE>"

predict = Inference(model="ada:ft-personal-2021-03-01-00-00-01")
predict.predict("I love to play ->")
```
