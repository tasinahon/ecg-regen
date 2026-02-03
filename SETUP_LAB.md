# Lab PC Setup Instructions

## Quick Setup on Lab Computer

### 1. Clone Repository

```bash
git clone https://github.com/tasinahon/ecg-regen.git
cd ecg-regen
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows
```

### 3. Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 4. Download PTB-XL Data

You need to manually download and place the PTB-XL ECG records:

```bash
# Download from: https://physionet.org/content/ptb-xl/1.0.3/
# Extract records500/ folder to data/ptbxl/

# Expected structure:
# data/
#   ptbxl/
#     records500/
#       00000/
#       01000/
#       ...
```

### 5. Verify Setup

```bash
python setup.py
```

## Training with High VRAM

With more GPU memory, you can increase batch size:

```bash
# For 24GB VRAM (e.g., RTX 3090, RTX 4090)
python main.py --mode train \
    --batch_size 32 \
    --num_workers 4 \
    --epochs 50 \
    --output_dir outputs

# For 40GB+ VRAM (e.g., A100)
python main.py --mode train \
    --batch_size 64 \
    --num_workers 8 \
    --epochs 50 \
    --output_dir outputs
```

## Monitoring Training

### Terminal 1: Training
```bash
python main.py --mode train --batch_size 32 --epochs 50
```

### Terminal 2: Monitor (every 10 epochs)
```bash
watch -n 600 python monitor_training.py --checkpoint outputs/best_model.pt
```

## Expected Training Time

- **Batch size 8**: ~20 hours/epoch
- **Batch size 16**: ~10 hours/epoch  
- **Batch size 32**: ~5 hours/epoch
- **Batch size 64**: ~2.5 hours/epoch

**Total for 50 epochs** (batch_size=32): ~10 days

## Tips

1. **Use tmux or screen** to keep training running:
   ```bash
   tmux new -s ecg-train
   python main.py --mode train --batch_size 32
   # Ctrl+B then D to detach
   # tmux attach -t ecg-train to reattach
   ```

2. **Monitor GPU usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **Save checkpoints regularly** (already set to save every 10 epochs)

4. **Check progress samples** every 5% during training
