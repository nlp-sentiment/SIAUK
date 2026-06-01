

# SIAUK: Soft Injection of Action Unit Knowledge for Facial Expression Recognition



## Project Structure

```text
SIAUK/
│
├── data/
│   ├── FER2013/
│   └── CK+/
│
├── prompts/
│   ├── full_au_prompt.py
│   ├── au_only_prompt.py
│   └── basic_prompt.py
│
├── models/
│   ├── vlm_backbone.py
│   ├── lora_adapter.py
│   ├── visual_pooling.py
│   └── classifier.py
│
├── datasets/
│   ├── fer2013_dataset.py
│   └── ckplus_dataset.py
│
├── trainer/
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── utils/
│   ├── metrics.py
│   ├── logger.py
│   └── seed.py
│
├── configs/
│   └── config.yaml
│
├── checkpoints/
│
├── outputs/
│
└── requirements.txt
```

---

## Environment

Python Version:

```bash
Python >= 3.10
```

Main Dependencies:

```bash
torch
transformers
peft
accelerate
numpy
pandas
scikit-learn
opencv-python
Pillow
tqdm
matplotlib
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

### FER2013

```text
data/
└── FER2013/
    ├── train/
    ├── val/
    └── test/
```

### CK+

```text
data/
└── CK+/
    ├── train/
    ├── val/
    └── test/
```

The training, validation, and testing sets follow an 80/10/10 subject-independent split.

---

## Pretrained VLM

The framework uses:

```text
Qwen2-VL-7B-Instruct
```

Download the model and specify its path in:

```yaml
configs/config.yaml
```

Example:

```yaml
model_name: Qwen2-VL-7B-Instruct
model_path: /path/to/Qwen2-VL-7B-Instruct
```

---

## Training

Start training:

```bash
python trainer/train.py
```

Training configuration:

```yaml
epochs: 5
batch_size: 8

learning_rate_lora: 5e-5
learning_rate_classifier: 1e-4

label_smoothing: 0.1

lora_rank: 16
lora_alpha: 32

dropout: 0.05
```

---

## Evaluation

Evaluate a trained checkpoint:

```bash
python trainer/evaluate.py \
    --checkpoint checkpoints/best_model.pt
```

Output metrics:

```text
Accuracy
Macro-F1
Precision
Recall
Confusion Matrix
```

---

## Inference

Single image prediction:

```bash
python trainer/inference.py \
    --image sample.jpg \
    --checkpoint checkpoints/best_model.pt
```

Output:

```text
Predicted Emotion: Happy
Confidence: 0.93
```

---

## Prompt Types

### Full AU Prompt

Contains:

* Task description
* Facial region guidance
* AU semantic prior knowledge
* Output constraints

### AU-only Prompt

Contains:

* AU semantic descriptions
* Emotion label set

### Basic Prompt

Contains:

* Emotion labels only

Prompt templates are located in:

```text
prompts/
```

---

## LoRA Configuration

Supported LoRA target modules:

```text
Q,V
Q,K,V
Q,K,V,O
```

Recommended setting:

```yaml
target_modules:
  - q_proj
  - v_proj
```

This configuration provides the best trade-off between performance and parameter efficiency.

---

## Visual Token Pooling

Supported pooling strategies:

```text
1. Mean Pooling
2. Last Token Pooling
3. Visual Token Mean Pooling
```

Recommended:

```yaml
pooling: visual_mean
```

Visual-token mean pooling focuses on expression-related visual regions and suppresses irrelevant textual noise.

---

## Checkpoints

Best checkpoints are automatically saved to:

```text
checkpoints/
```

Training logs and prediction results are stored in:

```text
outputs/
```

---

## Reproducibility

Random seed:

```yaml
seed: 42
```

Hardware:

```text
Ubuntu 20.04
PyTorch 2.2.2
NVIDIA GPU (24GB)
```

---
