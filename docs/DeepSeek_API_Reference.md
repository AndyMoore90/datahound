# DeepSeek API – Python Request Structure

DeepSeek uses the OpenAI-compatible API. Use the `openai` client with `base_url`.

## Setup

```python
from openai import OpenAI

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
api_key = "your-api-key"

client = OpenAI(
    api_key=api_key,
    base_url=DEEPSEEK_BASE_URL
)
```

---

## Chat Model (`deepseek-chat`)

```python
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ],
    max_tokens=8192,
    temperature=0.0,
    stream=False
)

content = response.choices[0].message.content
```

---

## Reasoner Model (`deepseek-reasoner`)

```python
response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {"role": "system", "content": "You are an expert analyst. Provide responses in valid JSON format only."},
        {"role": "user", "content": "Analyze the following data and return a JSON object..."}
    ],
    max_tokens=12000,
    temperature=0.1,
    stream=False
)

content = response.choices[0].message.content
```

---

## Model Comparison

| Model              | Use Case                             | Typical Settings          |
|--------------------|--------------------------------------|---------------------------|
| `deepseek-chat`    | General analysis, structured output   | temperature 0.0, 8192 tok |
| `deepseek-reasoner`| Complex reasoning, JSON extraction   | temperature 0.1, 12000 tok|

---

## Minimal Examples

**Chat:**

```python
from openai import OpenAI

client = OpenAI(api_key="sk-...", base_url="https://api.deepseek.com")
r = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": "Hi"}], max_tokens=1024)
print(r.choices[0].message.content)
```

**Reasoner:**

```python
from openai import OpenAI

client = OpenAI(api_key="sk-...", base_url="https://api.deepseek.com")
r = client.chat.completions.create(model="deepseek-reasoner", messages=[{"role": "user", "content": "What is 2+2?"}], max_tokens=1024)
print(r.choices[0].message.content)
```
