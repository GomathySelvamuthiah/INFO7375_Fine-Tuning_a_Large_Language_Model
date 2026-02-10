# E-Commerce Product Q&A Assistant with LoRA Fine-Tuning

Fine-tuning BLOOMZ-560M for product question answering using Parameter-Efficient Fine-Tuning (LoRA).

## 📋 Project Overview

This project demonstrates efficient LLM fine-tuning for e-commerce question answering using:
- **Model**: BLOOMZ-560M (559M parameters)
- **Technique**: LoRA (Low-Rank Adaptation)
- **Dataset**: Amazon Product Q&A (2.5M examples)
- **Training**: Only 0.28% of parameters trained (1.57M/559M)

## 🎯 Key Results

- **Best Configuration**: r=16, α=32, lr=3e-4
- **Validation Loss**: 3.58 (4.8% better than r=8 baseline)
- **ROUGE-L Score**: 0.142 (+904% vs untrained baseline)
- **Training Time**: ~60 minutes on Tesla T4 GPU
- **Efficiency**: 99.72% of model parameters frozen

## 🔧 Environment Setup

### Requirements
```bash
Python 3.8+
CUDA 11.0+ (for GPU acceleration)
16GB+ RAM recommended
GPU: Tesla T4 or equivalent (15GB VRAM)
```

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/GomathySelvamuthiah/E-Commerce_Product_QA_Assistant.git
cd E-Commerce_Product_QA_Assistant
```

2. **Create virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Google Colab Setup

If running in Google Colab:

1. Upload `amazon_qa_finetuning.ipynb`
2. Enable GPU runtime: `Runtime > Change runtime type > GPU > T4`
3. Run all cells sequentially


## 🚀 Quick Start

### Running the Notebook

1. **Open Jupyter**:
```bash
jupyter notebook amazon_qa_finetuning.ipynb
```

2. **Run cells sequentially**:
   - Setup (Cell 1-2)
   - Data Preparation (Cell 3-6)
   - Model Loading (Cell 7-8)
   - Training (Cell 9-10)
   - Evaluation (Cell 11-14)
   - Inference Demo (Cell 15)

### Expected Runtime

- **Total**: ~90 minutes on Tesla T4
  - Data loading: 2 min
  - Tokenization: 3 min
  - Training (3 configs): 60 min
  - Evaluation: 10 min
  - Error analysis: 5 min

### Using the Trained Model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load model
base_model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
model = PeftModel.from_pretrained(base_model, "./results/lora_config2")
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")

# Generate answer
question = "Is this laptop compatible with Windows 11?"
prompt = f"Question: {question}\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)
```

## 📊 Evaluation Metrics

### ROUGE Scores (Fine-tuned vs Baseline)

| Metric   | Baseline | Fine-tuned | Improvement |
|----------|----------|------------|-------------|
| ROUGE-1  | 0.0144   | 0.2025     | +1303%      |
| ROUGE-2  | 0.0006   | 0.0306     | +5256%      |
| ROUGE-L  | 0.0142   | 0.1429     | +905%       |

### Hyperparameter Comparison

| Config | LoRA Rank | LR    | Eval Loss | Trainable Params |
|--------|-----------|-------|-----------|------------------|
| 1      | 8         | 3e-4  | 3.7633    | 786K (0.14%)     |
| **2**  | **16**    | **3e-4** | **3.5821** | **1.57M (0.28%)** |
| 3      | 4         | 5e-4  | 4.3861    | 393K (0.07%)     |

## ⚠️ Important Findings

### The ROUGE-Quality Gap

We discovered a critical limitation of automatic metrics:

**ROUGE scores improved dramatically (+900%) but actual response quality is moderate.**

**Why?**
- ROUGE counts word overlap, not semantic coherence
- Baseline was extremely weak (only yes/no answers)
- Model learned to echo questions rather than answer them (95% echo rate)

**Lesson**: Production systems need **human evaluation**, not just ROUGE!

## 🔍 Error Analysis Insights

Three main patterns identified:

1. **No short responses** (0/100 <3 words) ✓
2. **Heavy question echoing** (95/100 echo input)
3. **Consistent across question lengths**

## 🛠️ Reproducing Results

### Step-by-Step

1. **Environment setup** (5 min):
```bash
pip install -r requirements.txt
```

2. **Run training** (60 min):
```python
# All training code is in the notebook
# Just run cells 1-10 sequentially
```

3. **Evaluate results** (10 min):
```python
# Run cells 11-14 for evaluation
# Results saved to results/ folder
```

4. **Test inference** (5 min):
```python
# Run cell 15 for live demo
```



## 📚 References

- [LoRA Paper](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
- [BLOOMZ Model](https://huggingface.co/bigscience/bloomz-560m)
- [Amazon QA Dataset](https://huggingface.co/datasets/sentence-transformers/amazon-qa)
- [ROUGE Metrics](https://aclanthology.org/W04-1013/) (Lin, 2004)

## 🤝 Contributing

Issues and pull requests welcome!

## 📄 License

MIT License - see LICENSE file

## 👤 Author

Gomathy Selvamuthiah  
February 2026

## 🎥 Demo Video

See [Video Presentation Link](https://drive.google.com/file/d/1sTF2sMaCoFiCw38jvJev58kOWgPy--fi/view?usp=sharing) for live demonstration of:
- Training process
- Hyperparameter comparison
- Error analysis
- Real-time inference

