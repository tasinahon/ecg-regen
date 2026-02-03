# ECG-ReGen: Electrocardiogram Report Generation via Retrieval-Augmented Self-Supervised Learning

Implementation of the paper "Electrocardiogram Report Generation and Question Answering via Retrieval-Augmented Self-Supervised Modeling" for PTB-XL dataset.

## Overview

ECG-ReGen is a retrieval-based approach for ECG-to-text report generation that leverages:
- **Self-supervised pre-training** with Masked ECG Modeling (MEM), Masked Language Modeling (MLM), and ECG-Text Matching (ETM)
- **Multi-modal architecture** combining ECG encoder (Conv1D + Transformer) and BERT text encoder
- **FAISS-based retrieval** for efficient similarity search and report generation
- **Zero-shot LLM integration** for question answering

## Target Performance (PTB-XL Dataset)

| Metric | Target Score |
|--------|--------------|
| BLEU-1 | 0.801 |
| BLEU-2 | 0.768 |
| BLEU-3 | 0.737 |
| BLEU-4 | 0.700 |
| BERTScore | 0.920 |
| Meteor | 0.836 |
| Rouge | 0.836 |

## Installation

```bash
pip install -r requirements.txt
```

For GPU support with FAISS (recommended):
```bash
pip install faiss-gpu
```

## Data Preparation

Your PTB-XL data should be organized as follows:
```
data/
├── ptbxl_train_proper.jsonl
├── ptbxl_val_proper.jsonl
├── ptbxl_test_proper.jsonl
└── ptbxl/
    ├── ptbxl_database.csv
    └── records500/
```

JSONL format:
```json
{
  "dataset": "ptbxl",
  "id": "ptbxl_12345",
  "ecg_path": "path/to/ecg_file",
  "messages": [
    {"role": "user", "content": ""},
    {"role": "assistant", "content": "sinus rhythm normal ekg"}
  ]
}
```

## Training

Train the model with self-supervised pre-training (MEM + MLM + ETM):

```bash
python main.py --mode train \
    --train_data data/ptbxl_train_proper.jsonl \
    --val_data data/ptbxl_val_proper.jsonl \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 5e-5 \
    --mem_mask_ratio 0.75 \
    --mlm_mask_ratio 0.15 \
    --output_dir outputs
```

### Key Hyperparameters:
- `--mem_mask_ratio 0.75`: Mask 75% of ECG patches for reconstruction
- `--mlm_mask_ratio 0.15`: Mask 15% of text tokens for language modeling
- `--hidden_dim 768`: Embedding dimension (matches BERT base)
- `--batch_size 32`: Batch size (adjust based on GPU memory)

## Evaluation

Evaluate on test set with retrieval-based report generation:

```bash
python main.py --mode evaluate \
    --train_data data/ptbxl_train_proper.jsonl \
    --test_data data/ptbxl_test_proper.jsonl \
    --checkpoint outputs/best_model.pt \
    --batch_size 32 \
    --output_dir outputs
```

This will:
1. Build FAISS index from training set embeddings
2. Retrieve top-1 nearest neighbor reports for test samples
3. Evaluate using BLEU, BERTScore, Meteor, and Rouge metrics
4. Compare results with paper baseline scores

## Model Architecture

### ECG Encoder
- 1D Convolutional layers for local feature extraction
- Transformer encoder (6 layers, 12 heads, 768 hidden dim)
- Positional encoding
- Global max pooling for embeddings

### Multi-Modal Fusion
- Cross-attention between ECG and text features
- Feed-forward networks with GELU activation
- Layer normalization

### Pre-training Tasks
1. **MEM (Masked ECG Modeling)**: Reconstruct 75% masked ECG patches
2. **MLM (Masked Language Modeling)**: Predict 15% masked text tokens
3. **ETM (ECG-Text Matching)**: Binary classification for aligned pairs

## Project Structure

```
ecg-regen/
├── main.py              # Training and evaluation pipeline
├── model.py             # ECG-ReGen architecture
├── dataset.py           # PTB-XL data loader
├── trainer.py           # Self-supervised pre-training
├── retrieval.py         # FAISS indexing and retrieval
├── evaluation.py        # NLG metrics (BLEU, BERTScore, etc.)
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## Results

After evaluation, results are saved to:
- `outputs/results.json`: All metric scores
- `outputs/predictions.json`: Generated vs ground truth reports
- `outputs/faiss_index.bin`: FAISS index for future use
- `outputs/faiss_metadata.pkl`: Report corpus metadata

## Tips for Achieving Target Scores

1. **Data Quality**: Ensure ECG signals are properly loaded and normalized
2. **Training Duration**: Train for sufficient epochs (50+ recommended)
3. **Batch Size**: Use larger batches if GPU memory allows (improves stability)
4. **Learning Rate**: Fine-tune between 5e-5 and 1e-4
5. **FAISS Index**: Larger training set = better retrieval quality
6. **Post-processing**: Clean retrieved reports (remove artifacts, standardize format)

## Lab Computer Setup

For setting up on lab computers with high VRAM GPUs, see [SETUP_LAB.md](SETUP_LAB.md) for:
- Optimized batch sizes for different GPU configurations
- Training time estimates
- tmux/screen usage for long-running jobs
- Monitoring tools and best practices

## Citation

```bibtex
@article{tang2024ecgregen,
  title={Electrocardiogram Report Generation and Question Answering via Retrieval-Augmented Self-Supervised Modeling},
  author={Tang, Jialu and Xia, Tong and Lu, Yuan and Mascolo, Cecilia and Saeed, Aaqib},
  year={2024}
}
```

## License

This implementation is for research purposes only.
