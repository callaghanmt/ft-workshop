# Fine-Tuning Small Language Models with Huggingface Transformers and Unsloth
These workshop materials are for the worksho run at the Open University on 2025-11-24

## Creating Stoic Gemma

This is the repository for the **Stoic Gemma** fine-tuning workshop. This is designed to guide participants through the end-to-end process of taking a small, efficient Large Language Model (Google's Gemma 2 2B) and steering its behaviour to adopt a specific persona; in this case, a Stoic philosopher in the style of Marcus Aurelius.

We use the **Unsloth** library to ensure the training is optimised for speed and memory efficiency, making it suitable for the free tier of Google Colab using a T4 GPU.

## 📂 The Notebooks

The workshop is broken down into three stages. It is recommended to run them in the following order:

### 1. `fine_tuning_1.ipynb` — Exploring Base & Instruction Models
**Goal:** Understand the starting point.
Before we start tinkering, we need to understand the tools. This notebook compares the "Base" version of Gemma 2 (which behaves like an autocomplete engine) against the "Instruction Tuned" (IT) version (which behaves like a chatbot).
*   **Key Concepts:** Tokenisation, Base vs. IT models, Prompting strategies.
*   **Outcome:** You will see why a base model cannot simply "follow instructions" without further training.

### 2. `fine_tuning_2.ipynb` — Fine-Tuning with Unsloth
**Goal:** The training loop.
This is the main part of the workshop. We take the Instruction Tuned model and apply **QLoRA** (Quantised Low-Rank Adaptation) to fine-tune it on a synthetic dataset of "Stoic Wisdom".
*   **Key Concepts:** 4-bit Quantisation, LoRA adapters, Synthetic Data generation, Training with the `SFTTrainer`.
*   **Bonus:** This notebook includes **CodeCarbon** integration to measure the energy footprint of a training run.
*   **Outcome:** You will save a set of "adapter weights" that turn the generic assistant into a philosopher or upload to Huggingface Hub.

### 3. `fine_tuning_3.ipynb` — Domain-Specific Evaluation
**Goal:** Marking the 'homework'.
How do we know if the training actually worked? Standard benchmarks (like maths tests) won't tell us if the model is more "Stoic". Here, we build a custom heuristic metric—the "Stoic Score".
*   **Key Concepts:** Inference with adapters, A/B testing (Base vs. Fine-Tuned), Keyword heuristic evaluation.
*   **Outcome:** A generated report comparing the "before and after" responses to life scenarios (e.g., losing your keys, getting fired).

---

## 🚀 Getting Started

### Prerequisites
*   A **Google Account** (to access Google Colab).
*   A **Hugging Face Account** (to access the gated Gemma 2 models).
*   A **Hugging Face Access Token** (Write permissions).

### Setup Instructions

1.  **Open in Colab:** Upload these notebooks to your Google Drive or open them directly from GitHub into Google Colab.
2.  **Select the Correct Runtime:**
    *   Go to `Runtime` > `Change runtime type`.
    *   Select **T4 GPU**. This is essential; the code will not run on a CPU.
3.  **Secrets Management:**
    *   In the Colab sidebar (left-hand menu), look for the **Key** icon (Secrets).
    *   Add a new secret named `HF_TOKEN` and paste your Hugging Face access token there.
    *   Enable "Notebook access" for this secret.

## ⚠️ Important Notes

*   **Licence Agreement:** Gemma 2 is a _gated model_. You must visit the model page on Hugging Face and accept the licence terms before the code will work.
*   **Session Management:** If you are on the free tier of Colab, resources are not guaranteed. If you run out of memory, try restarting the session via `Runtime` > `Disconnect and delete runtime`.
*   **British Spelling:** Please note that while the code relies on standard Python libraries (which use US English), the documentation and outputs may favour British spelling (e.g., *optimisation*, *behaviour*, *tokenisation*).

## 🤝 Acknowledgements

*   **Model:** Google Gemma 2 (2B)
*   **Library:** Unsloth AI (for making fine-tuning 2x faster and 60% less memory-intensive)
*   **Carbon Tracking:** CodeCarbon
