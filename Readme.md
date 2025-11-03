# 🧠 Smart Study Assistant

An **AI-powered Study Assistant** that reads your study notes, summarizes them, and generates quiz questions for quick self-revision.  
Built as part of the **Data Science Internship Assignment**.

---

## 👤 Author

**Name:** Anurag Sain  
**University:** IIT Roorkee  
**Department:** Electrical Engineering  

---

## 🎯 Project Overview

This project automates the manual process of revising study notes using a fine-tuned Large Language Model (LLM).  
It performs three key tasks:

1. Summarizes long text into concise key points.  
2. Generates quiz questions based on the summary.  
3. Calculates a **Coverage Score**, indicating how well the summary captures the key ideas.

---

## 🧩 Architecture


### Components:
- **Planner:** Parses user input and defines tasks (summarization + quiz generation).  
- **Executor:** Runs the AI model (`google/flan-t5-base`, optionally fine-tuned using LoRA).  
- **Evaluator:** Measures quality through keyword overlap (Coverage Score).  

---

## 🧠 Models and Tools Used

| Component | Library / Tool | Description |
|------------|----------------|--------------|
| **Base Model** | `google/flan-t5-base` | Pretrained T5 model for text generation |
| **Fine-Tuning** | `PEFT (LoRA)` | Parameter-efficient fine-tuning |
| **Frameworks** | `transformers`, `datasets`, `evaluate`, `accelerate`, `peft` | Model, data & evaluation tools |
| **UI (Optional)** | `gradio` | Web interface for testing |

---


