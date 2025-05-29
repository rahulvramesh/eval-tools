# Bud Model

`opencompass/configs/models/bud/bud-model.py`

```python
from opencompass.models import OpenAI

models = [
    dict(
        abbr='meta-llama/Llama-3.2-3B-Instruct',
        type=OpenAI,
        path='meta-llama/Llama-3.2-3B-Instruct',
        key=
        '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',  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
        # meta_template=api_meta_template,
        query_per_second=1,
        max_out_len=2048,
        max_seq_len=4096,
        openai_api_base='http://localhost:8988/v1/chat/completions', #'https://api.groq.com/openai/v1',
        batch_size=8),
]
```