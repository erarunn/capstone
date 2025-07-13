# SmolLM2 Training Scripts
[![Run on Gradient](https://img.shields.io/badge/Run_on-Gradient-blue?logo=paperspace)](https://gradient.paperspace.com/) [![Hugging Face Datasets](https://img.shields.io/badge/Datasets-HuggingFace-orange?logo=huggingface)](https://huggingface.co/docs/datasets/) [![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red?logo=pytorch)](https://pytorch.org/) [![Transformers](https://img.shields.io/badge/Lib-Transformers-blue?logo=transformers)](https://huggingface.co/docs/transformers/) [![Google Colab](https://img.shields.io/badge/Notebook-Google_Colab-blue?logo=googlecolab)](https://colab.research.google.com/) [![Python](https://img.shields.io/badge/Language-Python-3776AB?logo=python)](https://www.python.org/) [![CUDA](https://img.shields.io/badge/Acceleration-CUDA-orange?logo=nvidia)](https://developer.nvidia.com/cuda-zone)


---
## Model checkpoints for all three trainings:

### 1. 
- [Checkpoint 1](https://drive.google.com/drive/folders/1MB1E-oIC1ZG8UwJm2tbutNX_b3iYfBS6?usp=sharing)  
- [Checkpoint 2](https://drive.google.com/drive/folders/1nUT6Tn9RS_AnVB2pZC1s6ZYknu6PrsIO?usp=sharing)  
- [Checkpoint 3](https://drive.google.com/drive/folders/1k7yNF3PmBbZx-7KXW9TfEvOl_Z7K1NlC?usp=sharing)

### Training Proof.
		`cd ./proof`
---

## Overview of Scripts

### 1. Basic SmolLM2 Script

- **Description:**  
  Implements a minimal transformer model with a simple training loop. Loads streaming dataset, tokenizes, and trains without gradient accumulation or mixed precision.

- **Pros:**  
  - Easy to understand and modify  
  - Minimal dependencies and simpler debugging  
  - Good starting point for experimentation

- **Cons:**  
  - No gradient accumulation limits batch size  
  - No learning rate scheduling, which can harm training stability  
  - No mixed precision, slower training and higher memory use  
  - Less efficient checkpoint management

---

### 2. Intermediate SmolLM2 Script

- **Description:**  
  Adds gradient accumulation to enable effective larger batch training on limited GPU memory. Uses cosine learning rate scheduler with warmup. Introduces mixed precision training (AMP) with GradScaler for faster, more memory-efficient training.

- **Pros:**  
  - Larger effective batch size with gradient accumulation  
  - Cosine LR scheduler stabilizes learning and can improve convergence  
  - Mixed precision speeds up training and reduces memory footprint  
  - Improved checkpoint saving/loading logic

- **Cons:**  
  - Still relatively small model size  
  - Slightly more complex code, harder for beginners  
  - No token padding or ignore_index handling in loss (may lead to wasted compute on padding tokens)

---

### 3. Advanced SmolLM2 Script

- **Description:**  
  Further increases model capacity (hidden size and intermediate size). Adds padding token ignore index in loss function to avoid computing loss on padding tokens. Enhanced logging for input samples and tokens. Refined checkpoint loading to load latest checkpoint by max step. Weight tying between embedding and head weights is noted but commented out.

- **Pros:**  
  - Larger model capacity allows learning more complex patterns  
  - Proper loss masking on padding tokens improves training signal quality  
  - Improved debugging prints help trace tokenization and inputs  
  - More robust checkpoint loading and saving  
  - Compatible with mixed precision and gradient accumulation

- **Cons:**  
  - Larger model requires more GPU memory and longer training time  
  - Weight tying not enabled, missing potential parameter efficiency and regularization  
  - Still uses simple greedy decoding in generation (can limit output diversity)  

---

## Comparison of the Three Scripts

| Feature                        | Script 1 (Basic) | Script 2 (Intermediate) | Script 3 (Advanced)   |
|-------------------------------|------------------|------------------------|----------------------|
| Model size                    | Small (576 hidden)| Medium (576 hidden + accumulation + AMP) | Larger (768 hidden, 3072 intermediate) |
| Gradient Accumulation          | No               | Yes                    | Yes                  |
| Mixed Precision (AMP)          | No               | Yes                    | Yes                  |
| Learning Rate Scheduler        | No               | Cosine with warmup     | Cosine with warmup   |
| Loss padding mask (ignore_index) | No            | No                     | Yes                  |
| Checkpoint handling            | Basic            | Improved               | Robust (latest checkpoint load) |
| Logging/debug prints           | Minimal          | Moderate               | Detailed (input tokens, sample text) |
| Weight tying                  | No               | No                     | Not enabled, commented |
| Generation decoding            | Greedy           | Greedy                 | Greedy               |

---

## Recommendations to Improve Accuracy

1. **Increase Model Capacity:**  
   Larger hidden sizes, intermediate layers, and more attention heads usually improve model expressivity.

2. **Enable Weight Tying:**  
   Sharing embedding and output projection weights reduces parameters and can improve generalization.

3. **Use Better Token Padding & Masking:**  
   Mask padding tokens in loss and attention to avoid noisy gradients and wasted compute.

4. **Use Advanced Decoding Strategies:**  
   Replace greedy decoding with beam search or sampling (top-k, nucleus) during evaluation for richer generation.

5. **Experiment with Optimizers and LR Schedules:**  
   Try AdamW variants, layer-wise learning rate decay, or other schedulers to stabilize training.

6. **Add Regularization:**  
   Incorporate dropout layers, label smoothing, or tune weight decay to reduce overfitting and improve generalization.  
   [![Run on Gradient](https://img.shields.io/badge/Run_on-Gradient-blue?logo=paperspace)](https://gradient.paperspace.com/)

7. **Increase Training Data or Fine-tuning:**  
   Augment the dataset or fine-tune the model on domain-specific corpora to boost accuracy. Streaming dataset loading supports large-scale data efficiently.  
   [![Hugging Face Datasets](https://img.shields.io/badge/Datasets-HuggingFace-orange?logo=huggingface)](https://huggingface.co/docs/datasets/)

8. **Mixed Precision with Loss Scaling:**  
   Ensure stable mixed precision training with appropriate GradScaler configuration.

9. **Gradient Checkpointing:**  
   Allows training larger models by trading compute for memory.

10. **Hyperparameter Tuning:**  
    Learning rates, batch sizes, sequence lengths, and accumulation steps impact final accuracy.

---

If you want, I can help you implement any of these improvements or create a unified script incorporating the best features from all three!
