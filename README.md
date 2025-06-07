# ✨ PromptForge – A Prompt Optimization Playground with RAG

### Live Demo

[streamlit-app-2025-06-07-20-06-60 (1).webm](https://github.com/user-attachments/assets/dfb78337-d685-4baa-8c5d-6124c7712d59)

---

## Overview

**PromptForge** is a simple tool for experimenting with prompt optimization for LLMs. It tackles the common issue of getting better outputs from language models without having to fine-tune endlessly by hand. Under the hood, it uses a Retrieval-Augmented Generation (RAG) approach and lets you choose from a few strategies to refine your original prompt. After that, it compares the original and refined outputs using a scoring system to estimate how much the output improved.

---

## Features

* **RAG-Based Prompt Refinement**
  Pulls examples from a small knowledge base of expert-level prompts to help rework your input.

* **Selectable Refinement Strategies**
  Choose how you want your prompt adjusted—e.g., for creative writing, technical explanations, concise replies, or step-by-step reasoning.

* **Output Comparison and Scoring**
  Automatically scores both original and refined outputs on a 1–10 scale and gives an “uplift” value to show improvement (if any).

* **Dual LLM Setup**
  One model handles prompt rewriting, another evaluates the responses to reduce bias.

* **Session History**
  Keeps a temporary history of prompts and results during your session.

---

## Tech Stack

* **Frontend/UI**: Streamlit
* **LLMs**: Google Gemini API: `gemini-2.0-flash` for internal system logic
* **Vector Search**: FAISS (`faiss-cpu`)
* **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
* **Environment Config**: `python-dotenv`
* **Logging**: Python’s built-in `logging` module

---

## Getting Started (Local Setup)

1. **Clone the Repo**

   ```bash
   git clone https://github.com/YOUR_USERNAME/prompt-forge.git
   cd prompt-forge
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Your API Key**
   Create a `.env` file with your Gemini API key:

   ```env
   GOOGLE_API_KEY="your-gemini-api-key"
   ```

5. **Build the RAG Index**

   ```bash
   python knowledge_base.py
   ```

6. **Run the App**

   ```bash
   streamlit run app.py
   ```

---

## Future Plans

* **Switch to FastAPI**
  Decouple backend logic from the UI and offer a scalable API.

* **Add More LLMs**
  Include support for models from OpenAI and Anthropic.

* **Smart Strategy Selection**
  Automatically suggest the best refinement approach based on the input.

* **Persistent History and Accounts**
  Store sessions and enable logins using something like Firestore.

* **Custom Knowledge Bases**
  Let users upload their own documents or datasets for RAG.

---
