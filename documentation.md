# Technical Documentation: E-Commerce Q&A with LoRA

## Table of Contents

1. [Architecture Overview](#architecture)
2. [Data Pipeline](#data)
3. [Training Methodology](#training)
4. [Evaluation Framework](#evaluation)
5. [Code Documentation](#code)
6. [Reproducibility Guide](#reproducibility)

---

## 1. Architecture Overview {#architecture}

### Model Selection: BLOOMZ-560M

**Specifications**:
- Parameters: 559 million
- Architecture: Causal transformer decoder
- Pretraining: Multilingual instruction tuning
- Context length: 2048 tokens

**Why BLOOMZ?**
- **Instruction-tuned**: Pretrained on QA-like tasks
- **Size**: Fits in 16GB GPU memory
- **Performance**: Comparable to larger models for specific tasks
- **Open-source**: Reproducible research

### LoRA Configuration

**Mathematical Foundation**:

LoRA decomposes weight updates as low-rank matrices:
```
W = W₀ + ΔW = W₀ + BA
```

Where:
- `W₀` = Frozen pretrained weights (559M params)
- `B, A` = Trainable rank-r matrices
- `r << d` (16 vs 4096)

**Implementation**:
```python
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # Rank: controls adapter capacity
    lora_alpha=32,           # Scaling factor
    lora_dropout=0.1,        # Regularization
    target_modules=["query_key_value"],  # BLOOMZ attention layers
    bias="none"
)
```

**Efficiency Gains**:
- Trainable params: 1.57M (0.28%)
- Frozen params: 557.6M (99.72%)
- Memory reduction: ~50%
- Training speed: 3x faster than full fine-tuning

---

## 2. Data Pipeline

### Dataset: Amazon Product Q&A

**Source**: `sentence-transformers/amazon-qa`
- Total examples: 2.5 million
- Format: Real customer Q&A pairs
- Coverage: 100+ product categories

### Preprocessing Steps

1. **Quality Filtering**:
```python
def clean_example(example):
    q, a = example["query"], example["answer"]
    
    # Type validation
    if not (isinstance(q, str) and isinstance(a, str)):
        return False
    
    # Length thresholds
    if len(q.strip()) < 10 or len(a.strip()) < 10:
        return False  # Remove too-short entries
    
    # Outlier removal
    if len(q.strip()) > 500 or len(a.strip()) > 1000:
        return False  # Remove extremely long entries
    
    return True
```

**Results**: 2.41M examples retained (95.2% of original)

2. **Subsampling Strategy**:
```python
cleaned = cleaned.shuffle(seed=42)
cleaned = cleaned.select(range(15000))
```

**Rationale**:
- Balances quality vs. compute
- Prevents overfitting on massive dataset
- Reduces training from 4h to 60min

3. **Train/Val/Test Split**:
```python
# 70% / 15% / 15% split
split_1 = cleaned.train_test_split(test_size=0.30, seed=42)
split_2 = split_1["test"].train_test_split(test_size=0.5, seed=42)

dataset_splits = {
    "train": split_1["train"],           # 10,500
    "validation": split_2["train"],      # 2,250
    "test": split_2["test"]              # 2,250
}
```

4. **Data Formatting**:
```python
def format_example(example):
    return {
        "input_text": f"Question: {example['query']}\nAnswer:",
        "target_text": example["answer"]
    }
```

**Example**:
```
Input: "Question: Is this laptop compatible with Windows 11?\nAnswer:"
Target: "Yes, it supports Windows 11 Pro and has all required specs."
```

### Tokenization
```python
def tokenize_function(examples):
    all_input_ids, all_labels = [], []
    
    for inp, tgt in zip(examples["input_text"], examples["target_text"]):
        full_text = inp + " " + tgt
        encoded = tokenizer(
            full_text, 
            truncation=True, 
            max_length=128, 
            padding="max_length"
        )
        
        all_input_ids.append(encoded["input_ids"])
        
        # Mask padding tokens in labels
        labels = [
            tid if tid != tokenizer.pad_token_id else -100 
            for tid in encoded["input_ids"]
        ]
        all_labels.append(labels)
    
    return {
        "input_ids": all_input_ids, 
        "attention_mask": encoded["attention_mask"],
        "labels": all_labels
    }
```

**Key Parameters**:
- `max_length=128`: Balance between context and memory
- `padding="max_length"`: Fixed-size batches for efficiency
- Label masking: Prevents training on padding tokens

---

## 3. Training Methodology

### Hyperparameter Grid Search

**Configurations Tested**:

| Config | LoRA Rank | LR    | Alpha | Purpose                 |
|--------|-----------|-------|-------|-------------------------|
| 1      | 8         | 3e-4  | 32    | Baseline standard LoRA  |
| 2      | 16        | 3e-4  | 32    | Test higher capacity    |
| 3      | 4         | 5e-4  | 16    | Test efficiency tradeoff|

### Training Arguments
```python
TrainingArguments(
    output_dir="./lora_config2",
    eval_strategy="epoch",           
    save_strategy="epoch",           
    logging_steps=200,               
    learning_rate=3e-4,              
    per_device_train_batch_size=4,  
    gradient_accumulation_steps=2,  
    num_train_epochs=2,              
    weight_decay=0.01,               
    max_grad_norm=1.0,               
    fp16=False,                      
    load_best_model_at_end=True, 
    metric_for_best_model="eval_loss"
)
```

### Training Loop
```python
def train_lora_config(config_name, lr, epochs, lora_r, lora_alpha, output_dir):
    # 1. Load base model
    model = AutoModelForCausalLM.from_pretrained(
        "bigscience/bloomz-560m", 
        torch_dtype=torch.float32
    )
    
    # 2. Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=["query_key_value"]
    )
    
    # 3. Apply LoRA adapters
    model = get_peft_model(model, peft_config)
    
    # 4. Train with Hugging Face Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"]
    )
    
    trainer.train()
    
    # 5. Save LoRA weights only
    model.save_pretrained(output_dir)
    
    return trainer.state.log_history
```

---

## 4. Evaluation Framework

### Metrics: ROUGE

**ROUGE-1** (Unigram F1):
```
Precision = (matching_unigrams / total_pred_unigrams)
Recall = (matching_unigrams / total_ref_unigrams)
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**ROUGE-L** (Longest Common Subsequence):
- Measures structural similarity
- Order-aware matching
- More robust than ROUGE-1

### Evaluation Code
```python
from evaluate import load

rouge = load("rouge")

def evaluate_model(model, test_data):
    predictions, references = [], []
    
    for example in test_data:
        pred = generate_answer(example["query"], model, tokenizer)
        predictions.append(pred)
        references.append(example["answer"])
    
    scores = rouge.compute(
        predictions=predictions, 
        references=references
    )
    
    return scores
```

---

## 5. Code Documentation

### Key Functions

#### `generate_answer(question, model, tokenizer, max_new_tokens=50)`

**Purpose**: Generate answer for a given product question

**Parameters**:
- `question` (str): Customer question
- `model`: LoRA fine-tuned model
- `tokenizer`: BLOOMZ tokenizer
- `max_new_tokens` (int): Max answer length

**Returns**: (str) Generated answer

**Implementation**:
```python
def generate_answer(question, model, tokenizer, max_new_tokens=50):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,          # Sampling randomness
            do_sample=True,           # Enable sampling
            top_p=0.9,                # Nucleus sampling
            repetition_penalty=1.2    # Discourage repetition
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_response.split("Answer:")[-1].strip()
    return answer
```

#### `ProductQAAssistant` Class

**Purpose**: Production-ready inference pipeline
```python
class ProductQAAssistant:
    def __init__(self, model, tokenizer, device):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device
    
    def preprocess_question(self, question):
        # Add question mark if missing
        question = question.strip()
        if not question.endswith('?'):
            question += '?'
        return question
    
    def generate_answer(self, question, max_new_tokens=50):
        question = self.preprocess_question(question)
        return generate_answer(
            question, 
            self.model, 
            self.tokenizer, 
            max_new_tokens
        )
```

**Usage**:
```python
assistant = ProductQAAssistant(model, tokenizer, "cuda")
answer = assistant.generate_answer("Is this waterproof?")
```

---

## 6. Reproducibility Guide

### Exact Environment
```bash
# Python version
Python 3.10.12

# CUDA version
CUDA 11.8

# GPU
Tesla T4 (15GB VRAM)
```

### Random Seed Control

All randomness is controlled for reproducibility:
```python
import torch
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
```

### Deterministic Training
```python
# In TrainingArguments
seed=42,
data_seed=42
```

### Expected Outputs

**Config 2 Results** (r=16, lr=3e-4):
- Training loss: ~3.74 (final epoch)
- Validation loss: 3.5821
- ROUGE-L: 0.1429

**Variance**: Â±0.05 due to non-deterministic GPU operations



## 7. Known Limitations

1. **Response Quality**:
   - Coherence issues (responses mix unrelated concepts)
   - High question echoing rate (95%)
   - Needs more training data (15K → 50K+)

2. **ROUGE Limitations**:
   - Doesn't measure semantic coherence
   - Can be gamed by keyword matching
   - Production needs human evaluation

3. **Model Constraints**:
   - 560M params → limited world knowledge
   - Max length 128 tokens → truncates long contexts
   - Single language (primarily English)

---

## 8. Future Improvements

1. **Scale training**:
   - 15K → 50K examples
   - 2 → 5 epochs
   - Expect 20-30% ROUGE improvement

2. **Architecture**:
   - Apply LoRA to MLP layers too
   - Test r=32 for complex queries
   - Experiment with QLoRA (4-bit quantization)

3. **Evaluation**:
   - Add human evaluation (50 samples)
   - Implement BLEU, METEOR metrics
   - A/B test with production data

4. **Deployment**:
   - Quantize to INT8 (2x inference speedup)
   - Deploy with FastAPI + Docker
   - Add caching for common questions

---