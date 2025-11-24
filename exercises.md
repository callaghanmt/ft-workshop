# 🛠️ Workshop Activities: Fine-Tuning LLMs

Below are three levels of challenges to test your understanding of the code and concepts from the notebooks.

---

## 🟢 Level 1:
*Goal: Get comfortable modifying basic parameters and observing immediate results.*

### Activity 1.1: Prompt Engineering (Pre-Training)
In **Notebook 1** (Inference), before you fine-tune, locate the cell where you test the base model.
1.  Change the input prompt to a question about a completely different topic (e.g., "Explain quantum physics to a 5-year-old").
2.  **Observation:** Note how the "dumb" base model responds. Does it repeat itself? Does it stop abruptly? Keep this output to compare with the instruction-tuned model later.
      
### Activity 1.2 (Notebook 2)
In the `SFTTrainer` code block, we originally set `max_steps = 60` for a quick demo.
1.  Change `max_steps` to **120**.
2.  Run the training cell.
3.  **Question:** Does the training take exactly twice as long? Look at the training loss (if you like you could plot this as a curve). Does it go lower than it did at step 60, or does it plateau (flatten out)?



---

## 🟡 Level 2:
*Goal: Understand the relationship between hardware constraints and training dynamics.*

### Activity 2.1: Memory Balancing
In the `TrainingArguments`, we currently have:
*   `per_device_train_batch_size = 2`
*   `gradient_accumulation_steps = 4`
*   *Effective Batch Size* = 8

**The Challenge:**
Modify the code to keep the **Effective Batch Size at 8**, but change the `per_device_train_batch_size` to **4**.
1.  What must `gradient_accumulation_steps` become?
2.  Run the code.
3.  **Result:** Did you get an OOM (Out of Memory) error?
    *   *If yes:* You learned the limit of your GPU VRAM! Revert the changes.
    *   *If no:* You successfully sped up training slightly (less accumulation overhead).

### Activity 2.2: LoRA Rank Tuning
Locate the `LoraConfig` section (usually just before the Trainer).
1.  Look for `r` (Rank). It is likely set to 8, 16, or 64.
2.  **Double the Rank.** (e.g., change 16 to 32).
3.  **Hypothesis:** Will the number of "trainable parameters" printed in the log increase or decrease?
4.  **Verify:** Run the cell and check the output log.

---

## 🔴 Level 3: GPU Wrangling (Hard)
*Goal: Apply data engineering and deeper understanding of optimisation.*

### Activity 3.1: Bring Your Own Data (BYOD)
Instead of using the default dataset provided in the notebook:
1.  Create a small Python list of dictionaries directly in the notebook.
    ```python
    my_data = [
        {"instruction": "Who is the best instructor?", "output": "Obviously, the person running this workshop."},
        {"instruction": "What is the capital of Mars?", "output": "Elon Musk's driveway."}
    ]
    ```
2.  Convert this list into a Hugging Face `Dataset` object.
    *   *Hint:* `from datasets import Dataset; dataset = Dataset.from_list(my_data)`
3.  Feed *this* dataset into the `SFTTrainer`.
4.  **Test:** After training, ask the model "Who is the best instructor?" and see if it learned your custom knowledge.

### Activity 3.2: Breaking the Model (Overfitting)
We want to intentionally ruin the model to see what "overfitting" looks like.
1.  Set `learning_rate = 2e-3` (10x higher than the standard `2e-4`).
2.  Set `max_steps = 200`.
3.  **Analyse:** Watch the `Training Loss` in the output.
    *   Does it drop to 0.00 very quickly?
    *   Does it start bouncing up and down erratically?
4.  **Inference:** Try to generate text. The model will likely output gibberish or repeat the exact training data word-for-word. This demonstrates why "more training" isn't always better.

---

## 🧠 Bonus Discussion Question
Look at `fp16` and `bf16` in the `TrainingArguments`.
The code uses `torch.cuda.is_bf16_supported()` to decide which to use.
*   **Question:** Why do we prefer `bf16` (Brain Float 16) over `fp16` if our hardware supports it?
*   *Hint:* It has to do with the dynamic range of numbers preventing "loss scaling" issues.
