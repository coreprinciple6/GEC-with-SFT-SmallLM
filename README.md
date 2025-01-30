# 📝 Fine-Tuning SmolLM-135M for Grammatical Error Correction (GEC)  

## 📌 Overview  

This repository contains the implementation of **Supervised Fine-Tuning (SFT)** of the [SmolLM-135M model](https://huggingface.co/HuggingFaceTB/SmolLM-135M) for the **Grammatical Error Correction (GEC) task**. The fine-tuning is performed using the **CoEdIT dataset**, which consists of input sentences with grammatical errors and their corrected versions. The data is in json format. here is a sample from the dataset

```
{
  '_id': 1,
  'task': "gec",
  'src': "Improve the grammaticality: As the number of people grows, the need of habitable environment is unquestionably essential.",
  'tgt': "As the number of people grows, the need for a habitable environment is unquestionably increasing."
}
```

## 🎯 Task  

The goal is to train the SmolLM-135M model to effectively correct grammatical errors in input sentences. The model is evaluated using **BLEU** to measure its performance.  

## 🚀 Execution Plan  

### 🔹 1. Data Preparation  
- Download the **SmolLM-135M** model  
- Load the **CoEdIT dataset** and split it into **train** and **validation** sets  
- Format data as **source-target pairs**  
- Add **special tokens** for error correction  

### 🔹 2. Model Setup  
- Initialize from the **SmolLM-135M checkpoint**  
- Configure **hyperparameters** (learning rate, batch size, etc.)  
- Set up the **tokenizer** with appropriate **padding/truncation**  
- Prepare **data collator** for batching  

### 🔹 3. Training Loop  
- Implement **supervised fine-tuning** using **HuggingFace Trainer**  
- Monitor **training loss**  
- Save model **checkpoints**  

### 🔹 4. Evaluation Pipeline  
- Implement **BLEU scoring**
- Log **metrics** using `Weights & Biases (W&B)`  

### 🔹 5. Optimization  
- Tune **batch size** and **learning rate**  
- Apply **gradient accumulation** if necessary  
- Monitor **GPU memory usage**  
- Implement **mixed precision training** for efficiency  

## 📊 Expected Metrics  and Results

| Metric        | Target Score | Achieved Score |
|--------------|-------------|---------------|
| BLEU Score   | >45 (BEA-2019 baseline) | **0.47** (after 2 epochs) |
| Loss | >50 (CoNLL-2014 benchmark) | TBD |

## ⚙️ Training Configuration  

```python
training_args = SFTConfig(
    output_dir="./sft_model",
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=100,  
    gradient_accumulation_steps=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    learning_rate=3e-5,
    num_train_epochs=2,
    weight_decay=0.1,
    save_steps=100,
    max_seq_length=MAX_LENGTH,
    use_cpu=False,
    report_to=["wandb"],
)
```

## 📌 Post-Training Steps  

- Generate **inference pipeline**  
- Create **evaluation report**  
- Document **training configuration**  
- Save **model artifacts and metrics**  

## 📂 Model Checkpoints  

The trained model is available on [Hugging Face 🤗](#) (link to be added).  
