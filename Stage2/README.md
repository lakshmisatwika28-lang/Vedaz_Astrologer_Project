# Vedaz Stage 2 Technical Task

## Role
AI Engineer — AI Astrologers, Vedaz

## Overview

This project implements three tools to help maintain the quality and safety of AI-generated astrology conversations:

- A **Chat Checker** to validate dataset quality and detect unsafe conversations.
- A **Chat Generator** to create new training conversations using an AI model.
- A **Quality Evaluator** to measure the safety and helpfulness of AI responses.

The objective is to keep Vedaz's AI astrologer safe, warm, honest, and non-fatalistic while maintaining a scalable workflow for training data generation and evaluation.

---

## APIs Used

### Task 1 – Chat Checker
- **No external API used**
- Implemented completely in Python using:
  - scikit-learn (duplicate detection with TF-IDF cosine similarity)
  - Regular Expressions (rule-based safety detection)

### Task 2 – Chat Generator
- **Google Gemini API (Google AI Studio)**
- Model Used:
  - **gemini-2.5-flash**

Gemini generates new Vedaz-style conversations based on different astrology topics. Every generated conversation is automatically validated using the Chat Checker before being saved.

### Task 3 – Quality Evaluator
- **Groq API**
- Model Used:
  - **Llama 3.1 8B Instant (`llama-3.1-8b-instant`)**

The evaluator sends test questions to the AI assistant, collects responses, and then scores them for safety, warmth, honesty about astrology's limitations, and overall helpfulness.

---

# Project Structure

```text
Stage2/
├── data/
│   ├── vedaz_astrologer_finetune.json
│   ├── vedaz_astrologer_finetune.jsonl
│   └── generated_chats.jsonl
│
├── results/
│   ├── checker_report.txt
│   ├── train.jsonl
│   ├── test.jsonl
│   └── evaluation.csv
│
├── checker.py
├── generator.py
├── evaluator.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Variables

### Google Gemini API (Task 2)

Windows (CMD):

```cmd
set GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
```

Linux / macOS:

```bash
export GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
```

---

### Groq API (Task 3)

Windows (CMD):

```cmd
set GROQ_API_KEY=YOUR_GROQ_API_KEY
```

Linux / macOS:

```bash
export GROQ_API_KEY=YOUR_GROQ_API_KEY
```

---

# Task 1 – Chat Checker

Run:

```bash
python checker.py --input data/vedaz_astrologer_finetune.jsonl --output results
```

The Chat Checker:

- validates chat structure
- checks message ordering
- counts conversation length
- detects unsafe conversations
- identifies duplicate or near-duplicate chats
- creates train/test splits
- generates a detailed validation report

---

# Task 2 – Chat Generator

Run:

```bash
python generator.py
```

The generator:

- uses **Google Gemini API**
- generates Vedaz-style conversations
- converts responses into JSON format
- validates every generated chat using the Chat Checker
- saves only valid and safe chats into:

```
data/generated_chats.jsonl
```

---

# Task 3 – Quality Evaluator

Run:

```bash
python evaluator.py
```

The evaluator:

- uses the **Groq API**
- generates responses to predefined evaluation questions
- scores responses based on:
  - Safety
  - Warmth
  - Honesty about astrology's limitations
  - Helpfulness
- saves the evaluation report as:

```
results/evaluation.csv
```

---

# Safety Detection Method

A rule-based safety detection approach was implemented using keyword and regular expression matching.

The checker identifies conversations containing:

- death prediction
- illness prediction
- guaranteed outcomes
- fear-selling
- financial promises

This method is transparent, deterministic, and easy to debug. While effective for detecting explicit violations, it may miss subtle unsafe responses. With more time, this could be improved using an LLM-based classifier for context-aware safety evaluation.

---

# Future Improvements

- Add semantic similarity instead of only TF-IDF duplicate detection.
- Integrate an LLM-based safety classifier.
- Generate larger datasets with automatic topic diversification.
- Expand evaluation metrics with factual consistency and multilingual scoring.
- Fine-tune a lightweight open-source model using the validated dataset.

---

# Notes

The project focuses on building a practical data pipeline for AI-assisted astrology systems by combining automated validation, AI-based data generation, and repeatable quality evaluation while following Vedaz's safety principles.