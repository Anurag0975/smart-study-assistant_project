──────────────────────────────────────────────
                DATA SCIENCE INTERNSHIP PROJECT
──────────────────────────────────────────────
                     Project Title:
               🧠 SMART STUDY ASSISTANT
──────────────────────────────────────────────

Submitted by:   Anurag Sain  
Institute:      Indian Institute of Technology, Roorkee  
Department:     Electrical Engineering  
Submission Date: November 2025  

──────────────────────────────────────────────
1. INTRODUCTION
──────────────────────────────────────────────
The "Smart Study Assistant" is an AI-driven application designed to help students revise and understand study material efficiently. The system uses advanced Natural Language Processing (NLP) and transformer-based models to automatically summarize notes and generate quiz questions. 

The project was developed as part of the Data Science Internship Assignment, aimed at applying practical AI concepts such as model fine-tuning, evaluation, and deployment.

──────────────────────────────────────────────
2. OBJECTIVE
──────────────────────────────────────────────
The main objective of this project is to build an intelligent assistant that can:
1. Summarize lengthy study notes into concise key points.
2. Generate relevant quiz questions based on the provided text.
3. Evaluate the summarization quality using a "Coverage Score" metric.
4. Provide both a command-line and web-based user interface.

──────────────────────────────────────────────
3. TOOLS AND TECHNOLOGIES USED
──────────────────────────────────────────────
- **Programming Language:** Python  
- **Frameworks and Libraries:**
  - Transformers (Hugging Face)
  - Datasets
  - Evaluate
  - Accelerate
  - PEFT (LoRA)
  - Gradio (for UI)
  - Torch (PyTorch backend)
- **Model Used:** google/flan-t5-base  
- **Fine-tuning Method:** Parameter Efficient Fine-Tuning (LoRA)
- **Development Environment:** Visual Studio Code (VS Code)

──────────────────────────────────────────────
4. METHODOLOGY
──────────────────────────────────────────────

The Smart Study Assistant follows a modular pipeline:

Step 1: **Data Input**
   - The user pastes their study notes in the terminal or web interface.

Step 2: **Planning Stage**
   - The planner identifies the type of content (concept, process, or fact) and defines sub-tasks.

Step 3: **Model Execution**
   - The executor uses the FLAN-T5 model (or its fine-tuned variant) to generate:
       • A concise summary
       • 2–3 quiz questions

Step 4: **Evaluation Stage**
   - The evaluator computes a "Coverage Score" — how well the summary covers key concepts from the input.

Step 5: **Output Generation**
   - The summarized text, generated questions, and coverage score are displayed in the terminal or UI.

──────────────────────────────────────────────
5. MODEL AND TRAINING DETAILS
──────────────────────────────────────────────

Base Model: **FLAN-T5 (Google Research)**  
- Type: Encoder-Decoder Transformer  
- Pretrained on: Instruction-following tasks  
- Fine-tuned using: LoRA (Low-Rank Adaptation)

Fine-tuning involved using a small dataset containing pairs of text → summary/question examples.  
The model was optimized for lightweight deployment without heavy GPU requirements.

──────────────────────────────────────────────
6. FEATURES
──────────────────────────────────────────────
✅ Automated summarization of study notes  
✅ Quiz question generation for self-revision  
✅ Quantitative evaluation (Coverage Score)  
✅ Lightweight fine-tuning using LoRA  
✅ Optional web interface built using Gradio  

──────────────────────────────────────────────
7. RESULTS AND OUTPUT
──────────────────────────────────────────────
Example Input:
"Photosynthesis is the process by which green plants use sunlight to synthesize food from carbon dioxide and water."

Model Output:
Summary:
"Photosynthesis converts sunlight, carbon dioxide, and water into glucose and oxygen."

Quiz:
1. What pigment is essential for photosynthesis?
2. Which gas is released as a byproduct of this process?
3. What are the main inputs for photosynthesis?

Coverage Score: 92%

──────────────────────────────────────────────
8. EVALUATION METRIC
──────────────────────────────────────────────
The **Coverage Score (%)** measures how much of the original content’s key points appear in the generated summary.  
Formula:
Coverage Score = (Matched Keywords / Total Keywords) × 100

This ensures objective quality evaluation of the AI-generated summaries.

──────────────────────────────────────────────
9. CHALLENGES FACED
──────────────────────────────────────────────
- Compatibility issues with Python 3.14 and Pydantic-core during library installation.
- Managing dependency versions across Transformers and PEFT.
- Model loading time on CPU-only systems.
- Limited dataset for fine-tuning (handled by using few-shot examples).

──────────────────────────────────────────────
10. LEARNINGS
──────────────────────────────────────────────
- Understanding of Transformer-based NLP models.
- Practical knowledge of LoRA fine-tuning for resource-efficient customization.
- Experience in deploying AI models through both CLI and Gradio web interfaces.
- Exposure to evaluation metrics for text summarization and question generation.

──────────────────────────────────────────────
11. CONCLUSION
──────────────────────────────────────────────
The Smart Study Assistant successfully demonstrates the potential of AI in enhancing the learning process.  
By combining summarization, quiz generation, and automated evaluation, it serves as a digital revision partner for students.

This project integrates multiple data science concepts, from NLP model usage to evaluation and deployment, fulfilling the internship assignment objectives.

──────────────────────────────────────────────
12. FUTURE ENHANCEMENTS
──────────────────────────────────────────────
- Integration of voice-based input and text-to-speech output.
- Support for multiple subjects and topic classification.
- Expansion to multilingual summarization.
- Integration with note-taking platforms like Notion or Google Docs.

──────────────────────────────────────────────
13. REFERENCES
──────────────────────────────────────────────
- Hugging Face Transformers Documentation  
- Google Research: FLAN-T5 Paper  
- PEFT: Parameter Efficient Fine-Tuning Library  
- Gradio: Open Source Web UI for ML Models  
- Python Official Documentation  

──────────────────────────────────────────────
END OF REPORT
──────────────────────────────────────────────
