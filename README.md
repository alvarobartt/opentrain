<div align="center">
  <h1>opentrain</h1>
  <p>
    <em>🚂 Fine-tune OpenAI models for text classification, question answering, and more</em>
  </p>
</div>

---

`opentrain` is a simple Python package to fine-tune OpenAI models for task-specific purposes such as text classification, token classification, or question answering.

More information about OpenAI Fine-tuning at https://platform.openai.com/docs/guides/fine-tuning.

## 💻 Usage

### 🦾 Fine-tune

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

### 🤖 Predict

```python
import openai
from opentrain.predict import OpenAIPredict

openai.api_key = "<ADD_OPENAI_API_KEY_HERE>"

predict = OpenAIPredict(model="ada:ft-personal-2021-03-01-00-00-01")
predict.predict("I love to play ->")
```

## ⚠️ Warning

Fine-tuning OpenAI models via their API may take too long, so please be patient. Also, bear in mind
that in some cases you just won't need to fine-tune an OpenAI model for your task.

To keep track of all the models you fine-tuned, you should visit https://platform.openai.com/account/usage, 
and then in the "Daily usage breakdown (UTC)" you'll need to select the date where you triggered the
fine-tuning and click on "Fine-tune training" to see all the fine-tune training requests that you sent.

Besides that, in the OpenAI Playground at https://platform.openai.com/playground, you'll see a dropdown
menu for all the available models, both the default ones and the ones you fine-tuned. Usually, in the 
following format `<MODEL>:ft-personal-<DATE>`, e.g. `ada:ft-personal-2021-03-01-00-00-01`.
