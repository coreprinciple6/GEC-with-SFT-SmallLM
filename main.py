import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_metric
import numpy as np

# set device to mps if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

def prepare_dataset(examples, tokenizer, max_length=128):
    sources = [src.replace("Improve the grammaticality: ", "") for src in examples['src']]
    targets = examples['tgt']
    
    # For causal LM, concatenate source and target with a separator
    concatenated = [f"{src} => {tgt}" for src, tgt in zip(sources, targets)]
    
    model_inputs = tokenizer(
        concatenated,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    model_inputs['labels'] = model_inputs['input_ids'].clone()
    return model_inputs

def compute_metrics(eval_pred):
    bleu = load_metric("sacrebleu")
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    
    # Generate sequences from logits
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Extract only the target part after '=>'
    decoded_preds = [pred.split('=>')[-1].strip() if '=>' in pred else pred for pred in decoded_preds]
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_labels = [label.split('=>')[-1].strip() if '=>' in label else label for label in decoded_labels]
    
    return {"bleu": bleu.compute(predictions=decoded_preds, references=[[ref] for ref in decoded_labels])["score"]}


# Load dataset
full_train_ds = load_dataset("grammarly/coedit", split="train")
full_test_ds = load_dataset("grammarly/coedit", split="validation")
train_ds = full_train_ds.filter(lambda example: example['task'] == 'gec')
test_ds = full_test_ds.filter(lambda example: example['task'] == 'gec')

print(f"Train dataset size: {len(train_ds)}")
print(f"Test dataset size: {len(test_ds)}")

# Take subset for testing
train_ds = train_ds.select(range(5000))
test_ds = test_ds.select(range(300))

# Initialize model and tokenizer
model_name = "HuggingFaceTB/SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.decoder_start_token_id = tokenizer.cls_token_id

# Check and set special tokens
if tokenizer.bos_token is None:
    tokenizer.bos_token = "<BOS>"
if tokenizer.eos_token is None:
    tokenizer.eos_token = "<EOS>"
if tokenizer.pad_token is None:
    tokenizer.pad_token = "<PAD>"

# Prepare datasets
train_ds = train_ds.map(
    lambda x: prepare_dataset(x, tokenizer),
    batched=True,
    remove_columns=train_ds.column_names
)
test_ds = test_ds.map(
    lambda x: prepare_dataset(x, tokenizer),
    batched=True,
    remove_columns=test_ds.column_names
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./gec_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics
)

# Train and evaluate
trainer.train()
eval_results = trainer.evaluate()
print(f"BLEU Score: {eval_results['eval_bleu']}")

# Save model
trainer.save_model("./gec_model_final")



import torch

def correct_grammar(model, tokenizer, text, max_length=128):
    # Format the input as a prompt
    prompt = f"Input: {text} Output:"
    
    # Move model to device
    device = next(model.parameters()).device
    
    # Tokenize with attention mask
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True,
        add_special_tokens=True
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Ensure pad token and attention mask are properly set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # Generate with proper parameters
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        num_beams=5,
        length_penalty=0.6,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    
    # Decode and extract the correction
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the correction part after "Output:"
    try:
        corrected_text = decoded_output.split("Output:")[-1].strip()
    except:
        corrected_text = decoded_output
    
    return corrected_text

# Example usage
device = torch.device("cpu")
model = model.to(device)

# Test the model
text = "As the number of people grows, the need of habitable environment is unquestionably essential."
corrected = correct_grammar(model, tokenizer, text)
print(f"Original: {text}")
print(f"Corrected: {corrected}")

# Test with a few more examples
test_cases = [
    "She don't like pizza.",
    "The cats is sleeping on the bed.",
    "I have went to the store yesterday."
]

print("\nMore test cases:")
for test in test_cases:
    corrected = correct_grammar(model, tokenizer, test)
    print('-'*50)
    print(f"\nOriginal: {test}")
    print(f"Corrected: {corrected}")