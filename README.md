# GPT-2 Fine-Tuned Mental Health Chatbot

![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model%20Hosted-orange?logo=huggingface)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Transformers](https://img.shields.io/badge/Huggingface-Transformers-yellow?logo=transformers)
![License](https://img.shields.io/github/license/TheCarBun/GPT2-mental-health)

## 🧠 Overview

This repository contains a **fine-tuned GPT-2 model** designed for a **mental health chatbot** that provides supportive and empathetic responses to users expressing distress. The model was trained on conversational data to generate comforting and meaningful replies.

## 📂 Repository Structure

```
├── fine_tuned_model/
│   ├── config.json
│   ├── generation_config.json
│   ├── merges.txt
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── training_args.bin
│   ├── vocab.json
├── intents.json   # Dataset used for training
├── train.csv      # Processed training data
├── test.csv       # Processed test data
├── chatbot_data.txt # Raw conversation data
├── README.md      # Documentation
```

## 🚀 Model Details

- **Base Model:** GPT-2 (small)
- **Training Data:** Custom dataset containing mental health conversations
- **Objective:** Generate empathetic responses based on user input
- **Training Framework:** Hugging Face `transformers`
- **Fine-tuned on:** Google Colab (CUDA GPU)

## 🔧 Installation & Setup

### 1️⃣ Install Dependencies

```bash
pip install transformers torch datasets huggingface_hub
```

### 2️⃣ Load the Fine-Tuned Model

```python
from transformers import pipeline

chat_model = pipeline(
    "text-generation",
    model="TheCarBun/GPT-2-fine-tuned-mental-health",
    tokenizer="TheCarBun/GPT-2-fine-tuned-mental-health",
    truncation=True
)

input_text = "User: I'm feeling very down today."
output = chat_model(input_text, max_length=50)
print(output[0]['generated_text'])
```

## 🛠 Training Process

### 1️⃣ Preprocessing

- Cleaned conversational data from `intents.json`
- Tokenized text using GPT-2 tokenizer

### 2️⃣ Training

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="fine_tuned_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)
trainer.train()
```

## 📊 Results & Performance

| Epoch | Training Loss | Validation Loss |
| ----- | ------------- | --------------- |
| 1     | 1.1932        | 1.0248          |
| 2     | 0.7532        | 0.7870          |
| 3     | 0.7520        | 0.6927          |
| 4     | 0.6018        | 0.6579          |
| 5     | 0.5192        | 0.6403          |

## 🎯 Future Improvements

- Deploy as a **REST API**
- Fine-tune with **larger datasets**
- Add **sentiment analysis** for improved response generation

## 📜 License

This project is licensed under the **Apache 2.0 License**.

## 🙌 Contributing

Contributions are welcome! Feel free to submit a PR or open an issue.

---

📝 _Maintained by [TheCarBun](https://github.com/TheCarBun)_
