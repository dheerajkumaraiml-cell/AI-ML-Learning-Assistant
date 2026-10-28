# 🤖 AI/ML Learning Assistant

An AI-powered learning assistant project designed to help students and developers learn Artificial Intelligence, Machine Learning, Deep Learning, Natural Language Processing (NLP), Python programming, and practical coding concepts.

The goal is to develop a lightweight, domain-specific Transformer-based assistant that can answer educational questions, explain technical concepts, and assist with coding and debugging.

This project is also a hands-on learning journey focused on understanding the complete AI/ML development lifecycle, from data collection and preprocessing to model fine-tuning, evaluation, inference, and application integration.

## 🎯 Project Objectives

- Build an AI assistant focused on AI/ML education and programming.
- Provide clear explanations of technical concepts.
- Support Python programming, coding, and debugging questions.
- Learn how Transformer-based language models work.
- Prepare high-quality datasets for model fine-tuning.
- Implement a reproducible training and evaluation pipeline.
- Integrate the trained model into an application through a REST API.
- Apply software engineering, testing, and security best practices.

## 📚 Learning Domains

The assistant is intended to support questions related to:

- Machine Learning
- Deep Learning
- Natural Language Processing (NLP)
- Python Programming
- PyTorch and TensorFlow
- Data Science
- AI Fundamentals
- Coding and Debugging
- Model Training and Evaluation

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Primary programming language |
| PyTorch | Deep learning and model training |
| Hugging Face Transformers | Pretrained models and tokenization |
| PEFT / LoRA / QLoRA | Planned parameter-efficient fine-tuning |
| FastAPI | Planned REST API |
| SQLite | Planned database for users, API keys, and request logs |
| Pydantic | Planned request and response validation |
| Pytest | Planned automated testing |

*The technology stack and implementation status may evolve during development.*

## 📊 Dataset Preparation

Dataset preparation is one of the major completed milestones of this project.

The master dataset currently contains **37,089 records** after data cleaning, topic normalization, dataset merging, duplicate checks, and quality analysis.

The preparation pipeline includes:

1. Data collection from multiple sources.
2. Cleaning invalid and poor-quality records.
3. Normalizing topic categories.
4. Preparing additional Python, coding/debugging, and ML evaluation datasets.
5. Merging datasets and removing exact duplicate questions.
6. Analyzing dataset quality and category distributions.
7. Detecting and grouping potential near-duplicate questions.
8. Creating group-aware training, validation, and test splits.
9. Auditing cross-split overlap to reduce data leakage.

### Dataset Split

| Split | Records | Percentage |
|---|---:|---:|
| Training | 33,380 | 90% |
| Validation | 1,854 | 5% |
| Test | 1,855 | 5% |
| **Total** | **37,089** | **100%** |

The split passed the implemented exact-question overlap checks and character-level TF-IDF similarity audit at a threshold of 0.90. No cross-split near-duplicate candidates were detected by that audit. These checks reduce the risk of data leakage but cannot guarantee detection of every semantic duplicate.

### Dataset Files

```text
dataset/
├── merged_dataset.jsonl
├── cleaned_dataset.jsonl
├── normalized_dataset.jsonl
├── additional/
├── raw/
├── leakage_reports/
└── splits/
    ├── train.jsonl
    ├── validation.jsonl
    └── test.jsonl
```

The `merged_dataset.jsonl` file is the master dataset, while the `splits/` directory contains the accepted training, validation, and test files.

## 🏗️ Planned System Architecture

The intended end-to-end workflow is:

```text
Data Collection
      ↓
Data Cleaning and Preprocessing
      ↓
Dataset Merging and Quality Checks
      ↓
Train / Validation / Test Splitting
      ↓
Tokenization
      ↓
Base Model Selection
      ↓
LoRA / QLoRA Fine-Tuning
      ↓
Validation and Test Evaluation
      ↓
Model Saving and Loading
      ↓
Inference Pipeline
      ↓
FastAPI REST API
      ↓
API-Key Authentication
      ↓
SQLite Database and Request Logging
      ↓
Testing and Documentation
```

## 🚧 Development Status

### Completed

- [x] Dataset collection and preparation
- [x] Main dataset cleaning
- [x] Topic normalization
- [x] Additional dataset preparation
- [x] Dataset merging
- [x] Exact duplicate removal and checks
- [x] Dataset quality analysis
- [x] Group-aware train/validation/test splitting
- [x] Initial cross-split leakage audit

### In Progress / Planned

- [ ] Base model and tokenizer selection
- [ ] Tokenization and sequence-length analysis
- [ ] Model configuration and parameter-efficient fine-tuning
- [ ] Training and validation
- [ ] Test-set evaluation
- [ ] Checkpoint saving and loading
- [ ] Inference pipeline
- [ ] FastAPI integration
- [ ] API-key authentication
- [ ] SQLite integration
- [ ] Automated testing
- [ ] Final documentation and application interface

## 🧠 Learning Goals

This project aims to develop practical understanding of:

- Tokenization, token IDs, embeddings, and context windows.
- Transformer architecture and self-attention.
- Query, Key, and Value (Q/K/V) representations.
- Causal attention and next-token prediction.
- Training loss, backpropagation, and optimization.
- Learning rates, batch sizes, epochs, and checkpoints.
- Overfitting, generalization, and model evaluation.
- Inference, API design, authentication, and database integration.
- Testing, debugging, reproducibility, and maintainable code.

## ⚠️ Limitations

The model's final accuracy, response quality, and performance have not yet been established through completed training and evaluation.

The dataset has undergone automated quality and leakage checks, but automated similarity methods cannot identify every possible semantic duplicate. The assistant's reliability will need to be evaluated through suitable test cases and practical experiments.

## 🌱 Future Improvements

- Improve dataset diversity and answer quality.
- Evaluate model responses for correctness and relevance.
- Improve coding and debugging performance.
- Optimize memory usage and inference speed.
- Add API-key revocation, expiration, and secure request logging.
- Develop automated tests and comprehensive documentation.
- Build a user-friendly interface for interacting with the assistant.

## 🎯 Project Vision

Build a lightweight, domain-specific AI/ML Learning Assistant that combines educational explanations, practical programming support, and reliable software engineering.

The primary goal is not just to create a working application, but to understand, evaluate, explain, and improve every major component of the system.

**Status:** Under active development.