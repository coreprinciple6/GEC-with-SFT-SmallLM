import torch, evaluate, psutil, wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from tqdm import tqdm

wandb.login()
wandb.init(
    project="gec-with-sft",
    name="epoch2-fulldata",
    tags = ["mps"],
    config={
        "model": "HuggingFaceTB/SmolLM-135M",
        "task": "Grammarly GEC",
        "epochs": 2,
        "lr": 3e-5,
        "warmup_ratio": 0.1,
        "weight_decay": 0.1,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 256,
    }
)
## ~~~~~~~~~ CONFIGURATIONS ~~~~~~~~~~~~ ##

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
MAX_LENGTH = 256
BATCH_SIZE = 32
NUM_BEAMS = 2
EPOCHS = 2

## ~~~~~~~~~ HELPER FUNCTIONS ~~~~~~~~~~~~ ##
def print_memory_usage():
    '''Prints the current memory usage.'''
    process = psutil.Process()
    print(f"RAM Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")

# Define task and response templates
task_template = ""
response_template = "Corrected:" 

def format_data(example):
    '''Formats the data into a single text prompt.'''
    text = f"{task_template}\n{example['src']}\n{response_template}\n{example['tgt']}{tokenizer.eos_token}"
    return {"text": text}

def prompt_func_eval(batch):
    '''Formats the data into a single text prompt for evaluation.'''
    prompts = []
    for src in batch['src']:
        formatted = f"{task_template}\n{src}\n{response_template}\n"
        prompts.append(formatted)
    return prompts

## ~~~~~~~~~ MODEL CINFIGURATIONS ~~~~~~~~~~~~ ##

# Initialize model and tokenizer
model_name = "HuggingFaceTB/SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set padding token to eos token as its not set by default
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'
model = model.to(DEVICE)

## ~~~~~~~~~ DATA PROCESSING ~~~~~~~~~~~~ ##
def prepare_dataset(examples, tokenizer, max_length=MAX_LENGTH):
    '''Tokenizes the input text and returns the input_ids and attention_mask.'''
    inputs = tokenizer(
        examples["text"],
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        return_attention_mask=True 
    )
    return inputs

# Load dataset
print("Loading dataset...")
train_ds_original = load_dataset("grammarly/coedit", split="train").filter(
     lambda x: x['task'] == 'gec' ).select(range(19823))
test_ds_original = load_dataset("grammarly/coedit", split="validation").filter(
    lambda x: x['task'] == 'gec').select(range(485))

print(f"Train dataset size: {len(train_ds_original)}")
print(f"Test dataset size: {len(test_ds_original)}")

# Apply formatting to the datasets
train_ds = train_ds_original.map(format_data, remove_columns=train_ds_original.column_names)
test_ds = test_ds_original.map(format_data, remove_columns=test_ds_original.column_names)

# Prepare datasets
train_ds = train_ds.map(
    lambda x: prepare_dataset(x, tokenizer), batched=True, remove_columns=train_ds.column_names
)
test_ds = test_ds.map(
    lambda x: prepare_dataset(x, tokenizer), batched=True, remove_columns=test_ds.column_names
)

class CustomDataCollatorForLanguageModeling(DataCollatorForCompletionOnlyLM):
    '''Custom data collator for language modeling that replaces -100 labels with eos token.'''
    def __call__(self, examples):
        batch = super().__call__(examples)
        labels = batch['labels']
        eos_token_id = self.tokenizer.eos_token_id
        labels[labels == -100] = eos_token_id # Replace -100 with eos token
        batch['labels'] = labels # Update labels
        return batch
    
collator = CustomDataCollatorForLanguageModeling(response_template, tokenizer=tokenizer)
print_memory_usage()

## ~~~~~~~~~ TRAINING ~~~~~~~~~~~~ ##
print("Training...")
# TrainingArguments
training_args = SFTConfig(
    output_dir="./smolLM_model",
    eval_strategy="steps",
    eval_steps=100, # Evaluate every 100 steps
    logging_steps=100, 
    gradient_accumulation_steps=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    learning_rate=3e-5,
    num_train_epochs=EPOCHS,
    weight_decay=0.1,
    save_steps=100,
    max_seq_length=MAX_LENGTH,
    use_cpu=False, # uses GPU
    report_to=["wandb"],
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
)
# Train and evaluate
trainer.train()
# Save model
trainer.save_model("./sft_model_final")
print("Model saved.")
tokenizer.save_pretrained("./sft_model_final")
print_memory_usage()

## ~~~~~~~~~ EVALUATION ~~~~~~~~~~~~ ##
DEVICE = torch.device("cpu")
def evaluate_model(model, tokenizer, dataset, batch_size=BATCH_SIZE):
    '''Evaluates the model on the test dataset and returns the BLEU score.'''
    model.eval()
    bleu = evaluate.load('bleu')
    all_preds = []
    all_refs = []

    # Iterate over the dataset in batches
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch = dataset[i:i+batch_size] 
        src = batch['src']
        tgt = batch['tgt']

        # Format prompts
        formatted_prompts = prompt_func_eval({
            'src': src,
            'tgt': tgt
        })

        # Tokenize the prompts
        inputs = tokenizer(
            formatted_prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=False
        ).to(DEVICE)

        # Generate predictions
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_LENGTH,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                num_beams=NUM_BEAMS,
                early_stopping=True
            )

        # Decode the generated outputs
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Extract the corrected text
        truncated_preds = []
        for pred in preds:
            split_pred = pred.split(response_template)
            if len(split_pred) >= 2:
                corrected_text = split_pred[1].strip()
            else:
                corrected_text = pred.strip()
            truncated_preds.append(corrected_text)

        # Append predictions and references
        all_preds.extend(truncated_preds)
        all_refs.extend(tgt)

    # Compute BLEU score
    results = bleu.compute(predictions=all_preds, references=all_refs)
    return results['bleu']

print("Evaluating...")
bleu_score = evaluate_model(model.to(DEVICE), tokenizer, test_ds_original, batch_size=BATCH_SIZE)
print("BLEU Score:", round(bleu_score, 3))

## ~~~~~~~~~ INFERENCE ~~~~~~~~~~~~ ##
print("Inference...")
def correct_grammar(model, tokenizer, text, max_length=MAX_LENGTH):
    '''Corrects the grammar of the input text using the model.'''
    prompt = f"{task_template}\n{text}\n{response_template}\n"

    # Tokenize with attention mask
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=MAX_LENGTH,
        truncation=True,
        padding=True,
        add_special_tokens=False
    ).to(DEVICE)
    
    # Ensure pad token and attention mask are properly set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # Generate with proper parameters
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=MAX_LENGTH,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
    # Decode and extract the correction
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the correction part after "tgt:"
    try:
        corrected_text = decoded_output.split(response_template)[-1].strip()
    except:
        corrected_text = decoded_output
    
    return corrected_text

# Test the model
test_cases = [
    "Improve the grammaticality: As the number of people grows, the need of habitable environment is unquestionably essential.",
    "Improve the grammaticality: She don't like pizza.",
    "Improve the grammaticality: The cats is sleeping on the bed.",
    "Improve the grammaticality: I have went to the store yesterday."
]
for test in test_cases:
    corrected = correct_grammar(model, tokenizer, test)
    print('-'*50)
    print(f"\nOriginal: {test.split(':')[-1].strip()}")
    print(f"\nCorrected:  {corrected}")