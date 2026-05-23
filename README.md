# 🧠 Claude Thinking Dataset: 20 Model Fine-Tuning Notebooks

This repository contains **20 complete Jupyter notebooks** that fine-tune a wide range of machine learning models on a unique dataset of **Claude AI conversations** – including the assistant’s internal `<think>` reasoning chains.

The dataset (`claude-opus-4_5-250x_jsonl.txt`) consists of ~250 user–assistant pairs, where each assistant response contains a hidden reasoning block (`<think>...</think>`) followed by the final answer. We extract both parts and use them for various NLP, audio, vision, and agent tasks.
- [Download Dataset] [claude-opus-4_5-250x_jsonl.txt]( https://huggingface.co/datasets/[YOUR_USERNAME]/claude-thinking-dataset/resolve/main/claude-opus-4_5-250x_jsonl.txt)
---

## 📁 Repository Structure

```
├── claude-opus-4_5-250x_jsonl.txt     # Dataset (required)
├── 01_gpt2_language_model.ipynb
├── 02_t5_seq2seq_qa.ipynb
├── 03_lora_cot_distillation.ipynb
├── 04_bert_topic_classification.ipynb
├── 05_sentence_transformers_embeddings.ipynb
├── 06_bart_summarization.ipynb
├── 07_reward_model_rlhf.ipynb
├── 08_clip_vision_text.ipynb
├── 09_whisper_audio_asr.ipynb
├── 10_longformer_sliding_window.ipynb
├── 11_knowledge_distillation.ipynb
├── 12_codet5_code_generation.ipynb
├── 13_bert_ner.ipynb
├── 14_multilabel_classification.ipynb
├── 15_t5_question_generation.ipynb
├── 16_agent_react_tooluse.ipynb
├── 17_vit_t5_vqa.ipynb
├── 18_cnn_audio_classification.ipynb
├── 19_dpo_preference_optimization.ipynb
└── 20_full_agent_rag_pipeline.ipynb
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/claude-thinking-dataset.git
cd claude-thinking-dataset
```

### 2. Install base dependencies
All notebooks install required packages automatically (`transformers`, `datasets`, `torch`, etc.).  
We recommend a **Python 3.10+** environment with GPU support (CUDA).

```bash
pip install --upgrade pip
```

### 3. Dataset
Place the `claude-opus-4_5-250x_jsonl.txt` file in the root directory.  
Each line is a JSON object with a `"messages"` list containing `user` and `assistant` roles.  
The assistant content includes `<think>...</think>` tags.

### 4. Run notebooks
Open any notebook in Jupyter Lab/Notebook or VS Code.  
Execute cells sequentially. Most notebooks train a model in **2–10 minutes** on a single GPU (e.g., T4).

---

## 📚 Notebook Overview

| # | Notebook | Task | Model | Modality |
|---|----------|------|-------|-----------|
| 01 | GPT‑2 Language Model | Generate Claude‑style responses | GPT‑2 | Text Gen |
| 02 | T5 Seq2Seq QA | User → Response | Flan‑T5‑small | Text2Text |
| 03 | LoRA CoT Distillation | Chain‑of‑Thought reasoning | TinyLlama + LoRA | Text (CoT) |
| 04 | BERT Topic Classification | Classify user questions | BERT‑base | Text Class |
| 05 | Sentence Transformers | Semantic search (Q↔A) | all‑MiniLM‑L6‑v2 | Embeddings |
| 06 | BART Summarization | Thinking → concise answer | BART‑base | Summarization |
| 07 | Reward Model (RLHF) | Score response quality | DistilBERT | Regression |
| 08 | CLIP Vision‑Text | Concept image alignment | CLIP‑ViT‑B/32 | Vision‑Text |
| 09 | Whisper ASR | TTS → ASR on Claude responses | Whisper‑tiny | Speech |
| 10 | Longformer | Sliding window on long reasoning | Longformer‑base‑4096 | Long Context |
| 11 | Knowledge Distillation | BERT → BiLSTM student | BERT + LSTM | Distillation |
| 12 | CodeT5 | Code generation from natural language | CodeT5‑small | Code Gen |
| 13 | BERT NER | Extract entities from responses | dslim/bert‑base‑NER | Token Class |
| 14 | Multi‑label Classification | 8 skill tags per response | DistilBERT | Multi‑label |
| 15 | T5 Question Generation | Response → Question | Flan‑T5‑base | Reverse QA |
| 16 | ReAct Agent | Tool‑use (Thought→Action) | Flan‑T5‑small | Agent |
| 17 | ViT‑T5 VQA | Visual knowledge cards | ViT + T5 | VQA |
| 18 | CNN Audio Classification | MFCC on TTS audio | Custom CNN | Audio Class |
| 19 | DPO Preference Optimization | Align with thorough reasoning | TinyLlama + DPO | Preference |
| 20 | Full Agent RAG | ChromaDB + tool router + T5 | Mixed | RAG Agent |

---

## 🔧 Key Features

- **Real reasoning data** – Claude’s hidden `<think>` blocks provide rich chain‑of‑thought supervision.
- **From NLP to multimodal** – Covers text, speech, images, and code.
- **Production‑ready techniques** – LoRA, DPO, distillation, RAG, ReAct agents.
- **Minimal dependencies** – Each notebook is self‑contained and installs its own packages.
- **GPU‑friendly** – All models optionally use `fp16` and are tested on Colab/T4.

---

## 📊 Example Results

| Notebook | Sample Output |
|----------|----------------|
| GPT‑2 | `"<USER>: What is deep learning? <ASSISTANT>: Deep learning is a subset of machine learning that uses neural networks with many layers..."` |
| T5 QA | `"Neural networks learn by adjusting weights through backpropagation to minimize a loss function."` |
| BART Summary | `"Backpropagation computes gradients of the loss with respect to weights using the chain rule."` |
| CNN Audio | Topic classification accuracy ~82% on 5 categories (coding, science, math, philosophy, general). |
| ReAct Agent | `Question: Calculate 25*18+100` → `Thought: I need to compute. Action: calculator → Observation: 550 → Final Answer: 550` |

---

## 📦 Requirements (global)

- Python 3.10+
- CUDA-capable GPU (recommended, but many notebooks run on CPU)
- ~10 GB disk space for all models and cached data

All Python dependencies are installed inside each notebook via `!pip install`.

---

## 🤝 Contributing

Feel free to open issues or pull requests to improve notebooks, fix bugs, or add new tasks (e.g., multilingual, larger models, etc.).

---

## 📄 License

This project is for **educational and research purposes**. The dataset consists of synthetic/anonymized interactions. Please respect Anthropic’s acceptable use policy when using Claude‑generated content.

---

## ⭐ Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft), [TRL](https://github.com/huggingface/trl)
- [Sentence‑Transformers](https://www.sbert.net/)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Librosa](https://librosa.org/), [ChromaDB](https://www.trychroma.com/)

---

**Enjoy exploring how a single dataset can power 20 different ML tasks!**
